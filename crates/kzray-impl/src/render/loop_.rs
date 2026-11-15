//! 渲染主循环。
//!
//! 按行并行（`rayon`）。每行 clone 一个 `Sampler`——同一行内像素顺序
//! 处理，不同行完全并行。
//!
//! # 为什么每行 clone 而不是每线程 clone
//!
//! `rayon::map_init` 可以每线程 clone 一次，但需要 `Sync` 的 init 闭包。
//! 每行 clone 更简单——`PCGSampler` 的 clone 是 `Pcg64Mcg`（`u128`）+ `u64`
//! 的拷贝，开销可忽略（600 行 × ~50ns = 30μs，相比 18s 渲染时间可忽略）。
//!
//! # 种子独立性
//!
//! 每行 clone 的 `PCGSampler` 字段相同（同 `seed`），但 `start_pixel_sample`
//! 内部用 `(seed, pixel, sample_index)` hash，不同像素产生不同随机数序列。
//! 跨行不相关。跨像素不相关。
//!
//! # 维度
//!
//! 相机采样消耗维度 0-4（像素 2 + 透镜 2 + 时间 1），`li` 从维度 5 开始。
//!
//! # 并发
//!
//! `Film::add_sample` 用 `&self`，内部 `Mutex` 保护。多线程竞争锁——第一版
//! 先这样，profile 后如瓶颈改分块。

use glam::{UVec2, Vec2};
use rayon::prelude::*;

use kzray_abs::camera::Camera;
use kzray_abs::film::Film;
use kzray_abs::integrator::Integrator;
use kzray_abs::light::{InfiniteLight, Light, LightSampler};
use kzray_abs::sampler::Sampler;
use kzray_abs::scene::Scene;
use kzray_abs::spectrum::SpectralAlgebra;

use kzray_data::Image;

// ═══════════════════════════════════════════════════════════════
// RenderConfig
// ═══════════════════════════════════════════════════════════════

/// 渲染配置。
#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    /// 每像素采样数。
    pub samples_per_pixel: u32,
    /// 采样器全局种子。相同种子 + 相同场景 + 相同线程数 → 相同结果。
    pub seed: u64,
}

impl RenderConfig {
    pub fn new(samples_per_pixel: u32, seed: u64) -> Self {
        RenderConfig {
            samples_per_pixel,
            seed,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// render
// ═══════════════════════════════════════════════════════════════

/// 渲染一帧。按行并行。
///
/// # 泛型约束
///
/// - `SM: Sampler + Clone + Sync`：`Sampler` 已要求 `Send`，这里加 `Clone`
///   （每行 clone）和 `Sync`（共享引用 `&SM` 跨线程捕获）。
/// - `C: Camera<S>` / `SC: Scene<...>` / `I: Integrator<...>` / `F: Film<...>`：
///   都已 `Send + Sync`（契约层要求），无需额外约束。
///
/// # 种子
///
/// `base_sampler` 按值传入，`render` 拥有它。每行 `base_sampler.clone()`，
/// 克隆后的实例字段相同（同 seed），但 `start_pixel_sample(pixel, ...)`
/// 内部用 pixel hash，不同像素产生不同序列。
///
/// # 线程数
///
/// 由 `rayon` 全局线程池决定（`RAYON_NUM_THREADS` 环境变量或 CPU 核心数）。
/// 调用方可以通过 `rayon::ThreadPoolBuilder::new().num_threads(n).build()`
/// 自定义线程池。
pub fn render<S, LS, C, F, SC, L, IL, I, SM>(
    camera: &C,
    scene: &SC,
    integrator: &I,
    film: &F,
    base_sampler: SM,
    config: RenderConfig,
    wavelengths: &S::WavelengthContext,
) -> Image
where
    S: SpectralAlgebra,
    LS: LightSampler,
    C: Camera<S>,
    F: Film<S, Output = Image>,
    SC: Scene<S, LS, L, IL>,
    L: Light<S>,
    IL: InfiniteLight<S>,
    I: Integrator<S, LS, SC, L, IL>,
    SM: Sampler + Clone + Sync,
{
    let res = film.resolution();
    let width_f = res.x as f32;
    let height_f = res.y as f32;
    let spp = config.samples_per_pixel;
    assert!(spp > 0, "samples_per_pixel must be > 0");

    // 按行并行。每行 clone 一个 sampler。
    (0..res.y).into_par_iter().for_each(|y| {
        let mut sampler = base_sampler.clone();

        for x in 0..res.x {
            for s in 0..spp {
                sampler.start_pixel_sample(UVec2::new(x, y), s);

                // 像素抖动
                let (jx, jy) = sampler.next_2d(0);
                let pixel_sample = Vec2::new(
                    (x as f32 + jx) / width_f,
                    (y as f32 + jy) / height_f,
                );

                // 透镜采样
                let lens_sample = sampler.next_2d(2);

                // 时间采样
                let time = camera.sample_time(sampler.next_f32(4));

                // 生成主光线
                let cam_sample =
                    match camera.gen_ray(
                        pixel_sample,
                        Vec2::from(lens_sample),
                        time, wavelengths) {
                            Some(cs) => cs,
                            None => continue,
                        };

                // 相机采样消耗维度 0-4，li 从维度 5 开始
                let radiance =
                    integrator.li(scene, &cam_sample.ray, &mut sampler, 5, wavelengths);

                // 乘相机权重 W_e
                let weighted = radiance.mul(&cam_sample.weight);

                // 累加到胶片
                film.add_sample(cam_sample.film_position, weighted, wavelengths, 1.0);
            }
        }
    });

    film.resolve()
}
