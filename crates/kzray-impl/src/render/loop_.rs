//! 渲染主循环。
//!
//! 单线程遍历所有像素 × 样本，串联相机 → 积分器 → 胶片。
//!
//! # 采样维度
//!
//! 每个像素的每个样本：
//!
//! ```text
//! 像素抖动  next_2d(0)  → 维度 0, 1
//! 透镜样本  next_2d(2)  → 维度 2, 3
//! 时间样本  next_f32(4) → 维度 4
//! ```
//!
//! `PathIntegrator::li` 内部的 `dim` 从 0 开始，与相机采样的维度 0-4
//! **重叠**。
//!
//! **当前不影响结果**：`PCGSampler` 忽略 `dimension` 参数——每次
//! `next_f32` / `next_2d` 独立生成随机数，与维度索引无关。所以维度
//! 重叠只影响 Sobol / Halton 等低差异采样器。
//!
//! **将来加 Sobol 时必须修复**：两个方案：
//!
//! - `Integrator::li` 加 `start_dim: u32` 参数，渲染循环传 `5`；
//! - 或 `PathIntegrator::li` 内部用固定偏移（如 `dim = 16` 开始）。
//!
//! 后者不干净但无需改契约；前者需要改 `Integrator` trait。留待加
//! Sobol 时决策。
//!
//! # 并发
//!
//! 单线程。阶段 6 用 `rayon` 按行并行，每线程自己的 `PCGSampler`。
//! `Film::add_sample` 已支持 `&self` 并发写入。

use glam::{UVec2, Vec2};

use kzray_abs::camera::Camera;
use kzray_abs::film::Film;
use kzray_abs::integrator::Integrator;
use kzray_abs::light::{InfiniteLight, Light, LightSampler};
use kzray_abs::sampler::Sampler;
use kzray_abs::scene::Scene;
use kzray_abs::spectrum::SpectralAlgebra;

use kzray_data::Image;

use crate::sampler::PCGSampler;

// ═══════════════════════════════════════════════════════════════
// RenderConfig
// ═══════════════════════════════════════════════════════════════

/// 渲染配置。
#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    /// 每像素采样数。
    pub samples_per_pixel: u32,
    /// 采样器全局种子。相同种子 + 相同场景 → 相同结果。
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

/// 渲染一帧。
///
/// 遍历所有像素 × 样本，每样本：
///
/// 1. `sampler.start_pixel_sample(pixel, s)` 重置采样器；
/// 2. 采样像素抖动 `(u, v)`、透镜样本、时间；
/// 3. `camera.gen_ray` 生成主光线；
/// 4. `integrator.li` 计算辐射亮度；
/// 5. 乘 `W_e` 权重后 `film.add_sample`。
///
/// 最后 `film.resolve()` 得到 `Image`。
///
/// # 参数
///
/// - `camera`：相机。`gen_ray` 的 `pixel_sample` 是归一化屏幕坐标
///   `[0, 1]²`，渲染循环负责把 `(x + jitter, y + jitter) / (W, H)`
///   转换好传入；
/// - `scene`：场景只读视图；
/// - `integrator`：积分器；
/// - `film`：胶片。`Output` 必须是 `Image`；
/// - `config`：采样数与种子；
/// - `wavelengths`：波长上下文。
pub fn render<S, LS, C, F, SC, L, IL, I, SM>(
    camera: &C,
    scene: &SC,
    integrator: &I,
    film: &F,
    sampler: &mut SM,
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
    SM: Sampler,
{
    let res = film.resolution();
    let width_f = res.x as f32;
    let height_f = res.y as f32;
    let spp = config.samples_per_pixel;
    assert!(spp > 0, "samples_per_pixel must be > 0");


    for y in 0..res.y {
        for x in 0..res.x {
            for s in 0..spp {
                sampler.start_pixel_sample(UVec2::new(x, y), s);

                // 像素抖动：[0, 1)²
                let (jx, jy) = sampler.next_2d(0);
                // 归一化屏幕坐标
                let pixel_sample = Vec2::new(
                    (x as f32 + jx) / width_f,
                    (y as f32 + jy) / height_f,
                );

                // 透镜采样
                let lens_sample = sampler.next_2d(2);

                // 时间采样
                let time_sample = sampler.next_f32(4);
                let time = camera.sample_time(time_sample);

                // 生成主光线
                let cam_sample = match camera.gen_ray(pixel_sample,
                    Vec2::from(lens_sample), time, wavelengths)
                {
                    Some(cs) => cs,
                    None => continue,
                };

                // 追踪
                // 相机采样消耗维度 0-4，li 从维度 5 开始
                let radiance = integrator.li(
                    scene,
                    &cam_sample.ray,
                    sampler,
                    5,
                    wavelengths,
                );

                // 乘相机权重 W_e
                let weighted = radiance.mul(&cam_sample.weight);

                // 累加到胶片。weight=1.0，filter 权重在 film 内部归一化。
                film.add_sample(cam_sample.film_position, weighted, wavelengths, 1.0);
            }
        }
    }

    film.resolve()
}
