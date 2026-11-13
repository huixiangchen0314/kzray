//! 胶片契约。
//!
//! `Film` 是数据流的终点：积分器把样本累加到胶片，最后解析为输出。
//! 它是**状态容器**而非**采样器**——没有 `sample` / `pdf`，只有
//! `add_sample` / `splat` / `resolve`。
//!
//! # 职责边界
//!
//! 本模块只定义"像素累加 + 解析"的契约。以下职责**不在**本模块：
//!
//! - **重建滤波器（`Filter`）**：独立模块（`kzray-math/src/filter.rs`）。
//!   Film 实现层可以在内部持有 Filter 做 splat 权重分配，契约层不暴露。
//! - **光谱→RGB（`PixelSensor`）**：独立模块（`crate::sensor`）。Film 的
//!   `resolve` 返回 `Output`，输出形态由实现层决定；RGB 转换由实现层
//!   持有的具体 `PixelSensor` 完成，不在契约层。
//! - **波长采样（`SampledWavelengths`）**：spectrum 模块。Film 不负责
//!   决定采样哪些波长，只接收调用方给出的波长上下文。
//! - **图像 I/O**：`Output` 的职责，不是 Film 的。写出 EXR / PNG 是
//!   实现层 `Output` 类型的方法。
//! - **GBuffer 扩展**：能力 trait [`FilmWithVisibleSurface`]。
//! - **自适应采样**：能力 trait [`AdaptiveFilm`]。
//!
//! # `wavelengths` 参数
//!
//! `add_sample` / `splat` 接收 `&S::WavelengthContext`，因为：
//!
//! - `RGBFilm<Spectrum<N>>` 在累加时需要用 `CieSensor::process(value, wavelengths)`
//!   把光谱转成 RGB，再存到像素缓冲；
//! - `RGBFilm<SpectrumRgb>` 的 `wavelengths` 是 `&NoWavelengths`，零成本；
//! - `SpectralFilm` 存原始光谱数据，`wavelengths` 用于记录或后续转换。
//!
//! 如果 Film 实现**不**在累加时转换（如纯光谱存储），可以忽略 `wavelengths`。
//! 契约层不强制使用，但签名提供它是必要的——否则光谱渲染模式的 RGB
//! 转换无法在累加时完成。
//!
//! # 并发
//!
//! 多线程渲染时，多个线程并行 `add_sample`。`Film` 必须 `Send + Sync`，
//! `add_sample` 用 `&self`（不是 `&mut self`），内部可变性由实现层
//! 处理（`Mutex<Vec<Pixel>>` / 分块原子 / 无锁队列等）。
//!
//! # 与 `Camera` 的对称
//!
//! `Camera` 生成 `CameraSample`，其中 `film_position` 是胶片上的连续
//! 坐标；`Film::add_sample` 接受这个坐标。整数化和 filter 重建由
//! `Film` 内部处理——契约层只声明"连续位置 + 光谱值 + 波长上下文 + 权重"。
//!
//! # 与 pbrt `Film` 的差异
//!
//! pbrt 的 `Film` 是"所有与输出相关的东西的容器"（像素存储 + Filter +
//! PixelSensor + 波长采样 + 图像 I/O + GBuffer），因为 C++ 的
//! `TaggedPointer` 不允许把 Filter、PixelSensor 独立成可组合的组件。
//! 我们把每个职责放回它该在的模块——**胶片只负责像素存储**，重建在
//! Filter，光谱转换在 PixelSensor，图像格式在 Output。
//!
//! # 与 `light` / `bsdf` 的维度管理
//!
//! Film 不做采样，没有 `consumed` 或 `max_sample_dimensions`。`Camera`
//! 按固定维度预算取样本（像素 2D + 透镜 2D + 时间 1D），`add_sample` /
//! `splat` 不消耗维度。

use glam::{UVec2, Vec2};

use crate::spectrum::SpectralAlgebra;
use kzray_data::VisibleSurface;

// ═══════════════════════════════════════════════════════════════
// Film trait
// ═══════════════════════════════════════════════════════════════

/// 胶片契约。
///
/// 极窄接口：累加、splat、解析、分辨率。filter 重建、tile 管理、光谱到
/// RGB 的转换、图像 I/O 都是实现层细节，不在契约层暴露。
///
/// `Output` 是解析结果的类型，由实现层决定（RGB 图像、光谱数据、
/// GBuffer、`Vec<Pixel>` 等）。契约层不感知具体形态。
///
/// # 泛型参数 `S`
///
/// `S: SpectralAlgebra` 是样本的光谱类型（`Rgb` 或 `Spectrum<N>`）。
/// 实现层按 `S` 决定 `Output` 的形态：
///
/// - `S = Rgb`：`Output` 通常是 `Image`（线性 RGB 图像）；
/// - `S = Spectrum<N>`：`Output` 可能是光谱数据或经 `CieSensor` 转换后的
///   `Image`。
pub trait Film<S: SpectralAlgebra>: Send + Sync {
    /// 解析结果的类型。实现层决定。
    type Output;

    /// 累加一个样本。
    ///
    /// - `pixel`：胶片上的连续位置（像素坐标，含亚像素），范围
    ///   `[0, width] × [0, height]`。来自 `CameraSample::film_position`。
    /// - `value`：样本的光谱值，通常是路径吞吐量乘入射辐射亮度。
    /// - `wavelengths`：采样波长上下文。`RGBFilm<Spectrum<N>>` 用它做
    ///   光谱→RGB 转换；`RGBFilm<SpectrumRgb>` 的上下文是 `NoWavelengths`，
    ///   零成本。
    /// - `weight`：MIS 权重或采样权重（标量）。MIS 权重由积分器
    ///   按 `pdf` 计算后传入；采样权重通常是 `1 / sample_count`。
    ///
    /// filter 重建（按 filter 权重 splat 到邻域像素）、tile 归属、
    /// 光谱到 RGB 的转换（如果需要）都是实现层细节。实现层可以在内部
    /// 持有 `Filter` 和 `PixelSensor`，契约层不感知。
    ///
    /// 并发安全：`&self` 而非 `&mut self`，多线程可并行调用。
    fn add_sample(
        &self,
        pixel: Vec2,
        value: S,
        wavelengths: &S::WavelengthContext,
        weight: f32,
    );

    /// splat 一个样本。
    ///
    /// 与 `add_sample` 的区别：`add_sample` 是"有明确像素归属"的样本
    /// （从相机采样的 primary ray、BDPT 的端点连接），`splat` 是"没有
    /// 明确像素归属"的样本（光子映射的光子、BDPT 的某些路径）。
    /// splat 的样本按 filter 分布到邻域像素。
    ///
    /// 默认实现：退化为 `add_sample`（权重 1.0）。需要真正 splat
    /// 语义的实现覆盖此方法。
    fn splat(&self, pixel: Vec2, value: S, wavelengths: &S::WavelengthContext) {
        self.add_sample(pixel, value, wavelengths, 1.0);
    }

    /// 解析为最终输出。
    ///
    /// 消耗性操作但用 `&self`：多线程渲染完成后，调用方先 join 所有
    /// 线程，再调用 `resolve`。内部可变性由实现层处理。
    ///
    /// 返回 `Self::Output`，类型由实现层决定：
    ///
    /// - RGB 图像：`Image`（kzray-data）；
    /// - 光谱数据：`SpectralImage`（实现层类型）；
    /// - GBuffer：`GBuffer`（实现层类型）；
    /// - 原始像素数组：`Vec<Pixel>`。
    fn resolve(&self) -> Self::Output;

    /// 胶片分辨率。
    ///
    /// 与 `Camera::film_size` 应一致；调用方可以用两者校验。
    fn resolution(&self) -> UVec2;
}

// ═══════════════════════════════════════════════════════════════
// FilmWithVisibleSurface：GBuffer 能力
// ═══════════════════════════════════════════════════════════════

/// 胶片接受可见表面数据的能力。
///
/// 仅 GBuffer / AOV 类胶片实现本 trait。基础 `Film::add_sample` 不携带
/// `VisibleSurface`，因为 PT 的普通输出不需要；需要深度、法线、物体 ID
/// 等 AOV 时，调用方检查胶片是否实现本能力。
///
/// # 与 `light.rs` 的哲学一致
///
/// 抽象层不提供"从 `dyn Film` 找回 `FilmWithVisibleSurface`"的能力
/// 查询；持有具体实例的实现层自行枚举并 match。这与 `light.rs` 的
/// `AreaLight` / `InfiniteLight` 拆分同一原则——基础 trait 只放共性，
/// 可选能力拆能力 trait，不当心能力在类型层面不存在。
///
/// # 使用场景
///
/// 积分器在 `add_sample` 时如果要写 AOV，需要判断当前胶片是否支持。
/// 实现层通常会有一个 `FilmKind` enum（`RGB` / `GBuffer` / `Spectral`），
/// `match` 分派：
///
/// ```text
/// match film_kind {
///     FilmKind::GBuffer(f) => f.add_sample_with_visible(p, v, wl, w, &visible),
///     FilmKind::RGB(f) => f.add_sample(p, v, wl, w),
///     _ => {}
/// }
/// ```
pub trait FilmWithVisibleSurface<S: SpectralAlgebra>: Film<S> {
    /// 累加一个带可见表面数据的样本。
    ///
    /// `visible` 是求交产生的可见表面信息，由
    /// [`kzray_data::surface::VisibleSurface`] 提供。GBuffer 胶片用它
    /// 填充深度、法线、物体 ID 等通道。
    ///
    /// `value` / `wavelengths` / `weight` 的语义与 [`Film::add_sample`]
    /// 相同；`visible` 是附加数据，不参与光谱值计算。
    fn add_sample_with_visible(
        &self,
        pixel: Vec2,
        value: S,
        wavelengths: &S::WavelengthContext,
        weight: f32,
        visible: &VisibleSurface,
    );
}

// ═══════════════════════════════════════════════════════════════
// AdaptiveFilm：自适应采样能力
// ═══════════════════════════════════════════════════════════════

/// 胶片支持像素重置的能力。
///
/// 仅自适应采样（adaptive sampling）需要。基础 `Film` 不提供，因为
/// 普通 PT 的 `add_sample` 是单调累加，不需要重置。
///
/// # 使用场景
///
/// 自适应采样在判断某像素已收敛后，可以重置它（如果后续还要用），
/// 或者用于 tile 重渲染。普通 PT 不调用。
///
/// # 完整契约在实现层
///
/// 自适应采样的完整逻辑（方差估计、采样预算分配、tile 收敛判断）
/// 属于积分器或独立的自适应采样模块，不在本 trait 上。本 trait 只
/// 声明"可重置像素"这一能力。
pub trait AdaptiveFilm<S: SpectralAlgebra>: Film<S> {
    /// 重置指定像素的累积值。
    ///
    /// `pixel` 是整数像素坐标（不是连续位置），因为重置的粒度是像素。
    /// `pixel.x < resolution().x && pixel.y < resolution().y` 由调用方
    /// 保证；越界行为由实现层定义（panic 或忽略）。
    fn reset_pixel(&self, pixel: UVec2);
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use kzray_data::EntityId;

use super::*;
    use crate::spectrum::{NoWavelengths, Rgb as SpectrumRgb, SampledWavelengths, Spectrum};
    use std::sync::atomic::{AtomicU32, Ordering};

    // ─────────── 测试桩 ───────────

    struct CountingFilm {
        samples: AtomicU32,
        splats: AtomicU32,
        resolution: UVec2,
    }

    impl CountingFilm {
        fn new(width: u32, height: u32) -> Self {
            CountingFilm {
                samples: AtomicU32::new(0),
                splats: AtomicU32::new(0),
                resolution: UVec2::new(width, height),
            }
        }
    }

    impl Film<SpectrumRgb> for CountingFilm {
        type Output = (u32, u32);

        fn add_sample(
            &self,
            _pixel: Vec2,
            _value: SpectrumRgb,
            _wavelengths: &NoWavelengths,
            _weight: f32,
        ) {
            self.samples.fetch_add(1, Ordering::Relaxed);
        }

        fn splat(&self, _pixel: Vec2, _value: SpectrumRgb, _wavelengths: &NoWavelengths) {
            self.splats.fetch_add(1, Ordering::Relaxed);
        }

        fn resolve(&self) -> Self::Output {
            (
                self.samples.load(Ordering::Relaxed),
                self.splats.load(Ordering::Relaxed),
            )
        }

        fn resolution(&self) -> UVec2 {
            self.resolution
        }
    }

    struct DefaultSplatFilm {
        samples: AtomicU32,
        resolution: UVec2,
    }

    impl DefaultSplatFilm {
        fn new() -> Self {
            DefaultSplatFilm {
                samples: AtomicU32::new(0),
                resolution: UVec2::new(1, 1),
            }
        }
    }

    impl Film<SpectrumRgb> for DefaultSplatFilm {
        type Output = u32;

        fn add_sample(
            &self,
            _pixel: Vec2,
            _value: SpectrumRgb,
            _wavelengths: &NoWavelengths,
            _weight: f32,
        ) {
            self.samples.fetch_add(1, Ordering::Relaxed);
        }

        fn resolve(&self) -> Self::Output {
            self.samples.load(Ordering::Relaxed)
        }

        fn resolution(&self) -> UVec2 {
            self.resolution
        }
    }

    struct TestGbufferFilm {
        with_visible: AtomicU32,
        resolution: UVec2,
    }

    impl TestGbufferFilm {
        fn new() -> Self {
            TestGbufferFilm {
                with_visible: AtomicU32::new(0),
                resolution: UVec2::new(1, 1),
            }
        }
    }

    impl Film<SpectrumRgb> for TestGbufferFilm {
        type Output = u32;

        fn add_sample(
            &self,
            _pixel: Vec2,
            _value: SpectrumRgb,
            _wavelengths: &NoWavelengths,
            _weight: f32,
        ) {
        }

        fn resolve(&self) -> Self::Output {
            self.with_visible.load(Ordering::Relaxed)
        }

        fn resolution(&self) -> UVec2 {
            self.resolution
        }
    }

    impl FilmWithVisibleSurface<SpectrumRgb> for TestGbufferFilm {
        fn add_sample_with_visible(
            &self,
            _pixel: Vec2,
            _value: SpectrumRgb,
            _wavelengths: &NoWavelengths,
            _weight: f32,
            _visible: &VisibleSurface,
        ) {
            self.with_visible.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct TestAdaptiveFilm {
        resets: AtomicU32,
        resolution: UVec2,
    }

    impl TestAdaptiveFilm {
        fn new() -> Self {
            TestAdaptiveFilm {
                resets: AtomicU32::new(0),
                resolution: UVec2::new(1, 1),
            }
        }
    }

    impl Film<SpectrumRgb> for TestAdaptiveFilm {
        type Output = u32;

        fn add_sample(
            &self,
            _pixel: Vec2,
            _value: SpectrumRgb,
            _wavelengths: &NoWavelengths,
            _weight: f32,
        ) {
        }

        fn resolve(&self) -> Self::Output {
            self.resets.load(Ordering::Relaxed)
        }

        fn resolution(&self) -> UVec2 {
            self.resolution
        }
    }

    impl AdaptiveFilm<SpectrumRgb> for TestAdaptiveFilm {
        fn reset_pixel(&self, _pixel: UVec2) {
            self.resets.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct SpectralCountingFilm {
        count: AtomicU32,
    }

    impl Film<Spectrum<4>> for SpectralCountingFilm {
        type Output = u32;

        fn add_sample(
            &self,
            _pixel: Vec2,
            _value: Spectrum<4>,
            _wavelengths: &SampledWavelengths<4>,
            _weight: f32,
        ) {
            self.count.fetch_add(1, Ordering::Relaxed);
        }

        fn resolve(&self) -> u32 {
            self.count.load(Ordering::Relaxed)
        }

        fn resolution(&self) -> UVec2 {
            UVec2::new(1, 1)
        }
    }

    // ─────────── 基础契约 ───────────

    #[test]
    fn add_sample_counts() {
        let film = CountingFilm::new(100, 100);
        film.add_sample(Vec2::new(10.5, 20.5), SpectrumRgb::ONE, &NoWavelengths, 1.0);
        film.add_sample(Vec2::new(30.5, 40.5), SpectrumRgb::ONE, &NoWavelengths, 0.5);
        assert_eq!(film.resolve(), (2, 0));
    }

    #[test]
    fn splat_default_delegates_to_add_sample() {
        let film = DefaultSplatFilm::new();
        film.splat(Vec2::new(0.5, 0.5), SpectrumRgb::ONE, &NoWavelengths);
        film.splat(Vec2::new(0.5, 0.5), SpectrumRgb::ONE, &NoWavelengths);
        assert_eq!(film.resolve(), 2);
    }

    #[test]
    fn splat_override_does_not_call_add_sample() {
        let film = CountingFilm::new(100, 100);
        film.splat(Vec2::new(10.5, 20.5), SpectrumRgb::ONE, &NoWavelengths);
        film.splat(Vec2::new(30.5, 40.5), SpectrumRgb::ONE, &NoWavelengths);
        let (samples, splats) = film.resolve();
        assert_eq!(samples, 0);
        assert_eq!(splats, 2);
    }

    #[test]
    fn resolution_query() {
        let film = CountingFilm::new(1920, 1080);
        assert_eq!(film.resolution(), UVec2::new(1920, 1080));
    }

    // ─────────── 并发安全 ───────────

    #[test]
    fn add_sample_is_thread_safe() {
        use std::sync::Arc;
        use std::thread;

        let film = Arc::new(CountingFilm::new(100, 100));
        let mut handles = vec![];

        for _ in 0..4 {
            let film_clone = Arc::clone(&film);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    film_clone.add_sample(
                        Vec2::new(i as f32, i as f32),
                        SpectrumRgb::ONE,
                        &NoWavelengths,
                        1.0,
                    );
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let (samples, _) = film.resolve();
        assert_eq!(samples, 400);
    }

    // ─────────── GBuffer 能力 ───────────

    #[test]
    fn gbuffer_film_add_sample_with_visible() {
        let film = TestGbufferFilm::new();
        let visible = VisibleSurface::new(
            glam::Vec3::ZERO,
            glam::Vec3::Y,
            glam::Vec2::ZERO,
            1.0,
            kzray_data::common::InstanceId::from(EntityId::from_name("42")),
            0,
        );
        film.add_sample_with_visible(
            Vec2::new(0.5, 0.5),
            SpectrumRgb::ONE,
            &NoWavelengths,
            1.0,
            &visible,
        );
        assert_eq!(film.resolve(), 1);
    }

    #[test]
    fn gbuffer_film_is_also_film() {
        fn takes_film<F: Film<SpectrumRgb>>(_f: &F) {}
        let film = TestGbufferFilm::new();
        takes_film(&film);
    }

    // ─────────── 自适应能力 ───────────

    #[test]
    fn adaptive_film_reset_pixel() {
        let film = TestAdaptiveFilm::new();
        film.reset_pixel(UVec2::new(5, 10));
        film.reset_pixel(UVec2::new(0, 0));
        assert_eq!(film.resolve(), 2);
    }

    #[test]
    fn adaptive_film_is_also_film() {
        fn takes_film<F: Film<SpectrumRgb>>(_f: &F) {}
        let film = TestAdaptiveFilm::new();
        takes_film(&film);
    }

    // ─────────── object-safety ───────────

    #[test]
    fn film_is_object_safe() {
        let film: Box<dyn Film<SpectrumRgb, Output = (u32, u32)>> =
            Box::new(CountingFilm::new(1, 1));
        assert_eq!(film.resolution(), UVec2::new(1, 1));
    }

    // ─────────── Send + Sync ───────────

    #[test]
    fn film_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CountingFilm>();
    }

    // ─────────── Spectrum<N> 支持 ───────────

    #[test]
    fn film_generic_over_spectrum() {
        let film = SpectralCountingFilm {
            count: AtomicU32::new(0),
        };
        let value = Spectrum::new([0.1, 0.2, 0.3, 0.4]);
        let wl = SampledWavelengths::new([500.0; 4], [0.25; 4]);
        film.add_sample(Vec2::new(0.5, 0.5), value, &wl, 1.0);
        assert_eq!(film.resolve(), 1);
    }
}
