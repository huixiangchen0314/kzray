//! 积分器契约。
//!
//! 积分器是渲染器的**算法核心**——给定场景、光线、采样器，计算沿光线
//! 到达的辐射亮度。渲染方程的全部求解逻辑（路径追踪、BDPT、光子映射、
//! MLT……）都在这一层。
//!
//! # 为什么是窄接口
//!
//! 契约层只定义**入口**，不定义**过程**。`li` 一个方法足够：
//!
//! - **算法差异全在实现层**：PT 的路径状态机、BDPT 的子路径连接、
//!   PM 的光子映射、MLT 的马尔可夫链——这些是算法，不是契约。
//! - **契约层不假设算法结构**：如果契约层定义 `sample_path` / `connect` /
//!   `mnee` 之类的方法，就等于把某种算法结构强加给所有实现。
//! - **配置是实现的属性**：最大深度、RR 阈值、光源采样策略、MIS 权重——
//!   这些是积分器的构造参数，不是 `li` 的输入。
//!
//! 对比 `Light` / `BSDF` / `Camera`：它们的契约有多个方法，因为它们的
//! 行为是**协议的多个面**（采样 / 求值 / PDF），而积分器的行为是
//! **一个整体计算**——给定输入，返回输出，中间的算法过程不对外暴露。
//!
//! # 泛型参数
//!
//! - `S: SpectralAlgebra`：光谱类型；
//! - `LS: LightSampler`：光源采样器类型。与 [`crate::scene::Scene`] 的
//!   `LS` 一致——`LightSampler` 不可 dyn，所以通过泛型参数传递。
//!
//! 积分器本身是 **dyn compatible** 的：`li` 不是泛型方法，`S` / `LS` 是
//! trait 参数，`Box<dyn Integrator<S, LS>>` 可用。这允许实现层在运行时
//! 切换积分器（如 `--integrator pt` / `--integrator bdpt`）。
//!
//! # `&dyn Scene<S, LS>` 而非 `&impl Scene<S, LS>`
//!
//! `Integrator` 可能被 `Box<dyn Integrator<S, LS>>` 持有（运行时切换）。
//! 如果 `li` 接收 `&impl Scene<S, LS>`，`Integrator` 就不再 dyn compatible
//! ——每个 `Scene` 实现会单态化出一份 `Integrator`，`Box<dyn>` 无法表达。
//!
//! 代价是 `scene.intersect` / `scene.material` 等调用走虚分派。每次求交
//! 一次虚分派，相比 BVH 遍历（O(log n) 次盒求交）可忽略。这与
//! [`crate::scene::Scene`] 本身是 `&dyn Scene` 的取舍一致——`Scene` 的
//! 方法不是泛型方法，所以 `Scene` 可以 dyn；`LightSampler` 的 `sample`
//! 用了 `&mut impl Sampler`，所以 `LightSampler` 不可 dyn，通过 `LS`
//! 泛型参数传递。
//!
//! # 维度管理
//!
//! `li` 接收 `&mut dyn Sampler`，内部按**维度追踪**模型推进基址：
//!
//! ```text
//! → LightSampler::sample(ctx, sampler)      消耗 c1 维
//! → sampler.next_2d(dim1) 取 2D 样本       消耗 2 维
//! → BSDF::sample_f(sampler, dim2, mode)     消耗 c2 维
//! → sampler.next_f32(dim3) 取 RR 样本       消耗 1 维
//! ```
//!
//! 具体流程见 [`crate::conventions`] 的"维度流程"文档。积分器实现负责
//! 按这个流程推进，契约层不约束内部顺序——BDPT 的维度流程和 PT 不同，
//! 但都遵守"追踪实际消耗"的原则。
//!
//! # 与 `Camera` 的关系
//!
//! `Camera` 生成 primary ray，`Integrator::li` 消费它。积分器**不持有**
//! `Camera`——相机由积分器外部的调用方持有，`li` 只接收已生成的 ray。
//! 这是"数据流"的设计：相机 → ray → 积分器 → 辐射亮度 → 胶片。
//!
//! BDPT 的相机侧子路径由积分器内部调用 `Camera::sample_wi`（如果实现了
//! `SampleWiCamera`），但那是积分器的实现细节，不是契约层的输入。
//!
//! # 与 `Film` 的关系
//!
//! 积分器**不直接写 Film**。`li` 返回辐射亮度 `S`，调用方（渲染循环）
//! 把它累加到 `Film::add_sample`。这样积分器不需要知道 Film 的存在，
//! Film 也不需要知道积分器的算法——两者通过 `S` 解耦。
//!
//! 唯一的例外是 BDPT / PM 的 splat：`Film::splat` 由积分器在特定路径
//! 下调用（如光子映射的光子沉积）。这是实现层的选择，契约层不约束
//! ——`Integrator` 的实现可以持有 `&dyn Film`，也可以不持有。

use crate::light::LightSampler;
use crate::ray::Ray;
use crate::sampler::Sampler;
use crate::scene::Scene;
use crate::spectrum::SpectralAlgebra;

// ═══════════════════════════════════════════════════════════════
// Integrator
// ═══════════════════════════════════════════════════════════════

/// 积分器契约。
///
/// 给定场景、光线、采样器、波长上下文，计算沿光线到达的辐射亮度。
/// 这是渲染方程的**唯一入口**——所有算法差异在实现层。
///
/// # 泛型参数
///
/// - `S: SpectralAlgebra`：光谱类型；
/// - `LS: LightSampler`：光源采样器类型，与 `Scene<S, LS>` 一致。
///
/// # Object-safety
///
/// `Integrator<S, LS>` 是 object-safe 的：`li` 不是泛型方法。
/// `Box<dyn Integrator<S, LS>>` / `Arc<dyn Integrator<S, LS>>` 可用。
///
/// # 生命周期
///
/// `li` 不持有 `scene` / `ray` / `sampler` 的引用，只读/借用它们。
/// 积分器内部可以创建临时状态（路径栈、子路径、MIS 权重数组），
/// 但生命周期不超出 `li` 的调用。
pub trait Integrator<S, LS>: Send + Sync
where
    S: SpectralAlgebra,
    LS: LightSampler,
{
    /// 计算沿 `ray` 到达的辐射亮度。
    ///
    /// 参数：
    /// - `scene`：场景只读视图。求交、查材质、查光源都通过它；
    /// - `ray`：光线。`ray.time` 已由相机填充；`ray.origin` 和
    ///   `ray.dir` 是相机生成的 primary ray，或 BDPT 的相机侧子路径；
    /// - `sampler`：采样器。积分器内部按维度追踪推进基址；
    /// - `wavelengths`：波长上下文。只读。
    ///
    /// 返回沿光线到达的辐射亮度 `S`。调用方把它累加到 `Film`。
    ///
    /// # 语义
    ///
    /// - **从相机出发**（`Radiance` 传输方向）：返回的是沿 `ray.dir`
    ///   看到的辐射亮度。PT / BDPT 的 eye subpath 用这个语义。
    /// - **从光源出发**（`Importance` 传输方向）：BDPT 的 light subpath
    ///   需要不同的入口——当前 `li` 只定义 `Radiance` 方向。`Importance`
    ///   方向的子路径追踪是积分器的内部实现，不是契约层的独立方法。
    /// - **不累加自发光**：`li` 返回的是**反射/透射后的辐射亮度**，
    ///   不是射线直接命中的自发光。自发光由 `Light::sample_li` 或
    ///   `InfiniteLight::emitted_radiance` 在直接光照计算中处理。
    ///   BDPT 的端点连接会加入自发光，但那是算法内部的事。
    /// - **不乘相机权重**：`W_e` 由相机侧处理（`CameraSample::weight`），
    ///   `li` 返回的是场景辐射亮度，调用方乘 `W_e` 得到最终贡献。
    ///
    /// # 失败
    ///
    /// 返回 `S::zero()` 表示无贡献（如光线逃逸到无环境光的无限远、
    /// 路径被 RR 终止、采样失败）。`li` 不返回 `Option`——零贡献是
    /// 合法结果，调用方统一累加。
    ///
    /// # 维度
    ///
    /// `sampler` 的维度基址由调用方设置（`start_pixel_sample`）。`li`
    /// 内部按维度追踪推进，返回后调用方不需要知道消耗了多少维度——
    /// 下一次 `li` 调用前会重新 `start_pixel_sample`。
    fn li(
        &self,
        scene: &dyn Scene<S, LS>,
        ray: &Ray,
        sampler: &mut dyn Sampler,
        wavelengths: &S::WavelengthContext,
    ) -> S;
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::light::{LightSampleContext, LightSelection};
    use crate::material::Material;
    use crate::ray::SurfaceHit;
    use crate::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
    use kzray_data::common::LightId;
    use kzray_data::geometry::Aabb;
    use std::ops::Range;

    // ─────────── 测试桩 ───────────

    struct EmptyLightSampler;

    impl LightSampler for EmptyLightSampler {
        fn sample(
            &self,
            _ctx: &LightSampleContext,
            _sampler: &mut impl Sampler,
        ) -> Option<LightSelection> {
            None
        }

        fn pdf(&self, _ctx: &LightSampleContext, _light_id: LightId) -> f32 {
            0.0
        }

        fn max_sample_dimensions(&self) -> u32 {
            1
        }
    }

    struct EmptyScene;

    impl Scene<SpectrumRgb, EmptyLightSampler> for EmptyScene {
        fn bounds(&self) -> Aabb {
            Aabb::empty()
        }

        fn intersect(&self, _ray: &Ray, _t_range: Range<f32>) -> Option<SurfaceHit> {
            None
        }

        fn intersect_p(&self, _ray: &Ray, _t_range: Range<f32>) -> bool {
            false
        }

        fn material(&self, _hit: &SurfaceHit) -> Option<&Material<SpectrumRgb>> {
            None
        }

        fn light(&self, _id: LightId) -> Option<&dyn crate::light::Light<SpectrumRgb>> {
            None
        }

        fn light_sampler(&self) -> Option<&EmptyLightSampler> {
            None
        }

        fn infinite_light(&self) -> Option<&dyn crate::light::InfiniteLight<SpectrumRgb>> {
            None
        }
    }

    struct TestSampler;

    impl Sampler for TestSampler {
        fn next_f32(&mut self, _dimension: u32) -> f32 {
            0.5
        }
    }

    // ─────────── 测试用积分器 ───────────

    /// 常数积分器：无论输入如何，返回固定值。
    /// 用于验证签名、object-safety、`Send + Sync`。
    struct ConstantIntegrator {
        value: SpectrumRgb,
    }

    impl Integrator<SpectrumRgb, EmptyLightSampler> for ConstantIntegrator {
        fn li(
            &self,
            _scene: &dyn Scene<SpectrumRgb, EmptyLightSampler>,
            _ray: &Ray,
            _sampler: &mut dyn Sampler,
            _wavelengths: &NoWavelengths,
        ) -> SpectrumRgb {
            self.value
        }
    }

    /// 零积分器：永远返回零。用于验证"无贡献"路径。
    struct ZeroIntegrator;

    impl Integrator<SpectrumRgb, EmptyLightSampler> for ZeroIntegrator {
        fn li(
            &self,
            _scene: &dyn Scene<SpectrumRgb, EmptyLightSampler>,
            _ray: &Ray,
            _sampler: &mut dyn Sampler,
            _wavelengths: &NoWavelengths,
        ) -> SpectrumRgb {
            SpectrumRgb::ZERO
        }
    }

    // ─────────── 契约测试 ───────────

    #[test]
    fn constant_integrator_returns_value() {
        let integrator = ConstantIntegrator {
            value: SpectrumRgb::new(0.5, 0.5, 0.5),
        };
        let scene = EmptyScene;
        let ray = Ray::new(glam::Vec3::ZERO, glam::Vec3::X, 0.0);
        let mut sampler = TestSampler;
        let result = integrator.li(&scene, &ray, &mut sampler, &NoWavelengths);
        assert_eq!(result, SpectrumRgb::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn zero_integrator_returns_zero() {
        let integrator = ZeroIntegrator;
        let scene = EmptyScene;
        let ray = Ray::new(glam::Vec3::ZERO, glam::Vec3::X, 0.0);
        let mut sampler = TestSampler;
        let result = integrator.li(&scene, &ray, &mut sampler, &NoWavelengths);
        assert_eq!(result, SpectrumRgb::ZERO);
    }

    #[test]
    fn integrator_is_object_safe() {
        // Integrator<S, LS> 是 object-safe 的
        let _: Box<dyn Integrator<SpectrumRgb, EmptyLightSampler>> =
            Box::new(ConstantIntegrator {
                value: SpectrumRgb::ONE,
            });
    }

    #[test]
    fn integrator_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ConstantIntegrator>();
        assert_send_sync::<ZeroIntegrator>();
    }

    #[test]
    fn integrator_works_with_dyn_scene() {
        // 验证 &dyn Scene<S, LS> 作为参数可用
        let integrator: Box<dyn Integrator<SpectrumRgb, EmptyLightSampler>> =
            Box::new(ConstantIntegrator {
                value: SpectrumRgb::new(1.0, 0.0, 0.0),
            });
        let scene: Box<dyn Scene<SpectrumRgb, EmptyLightSampler>> = Box::new(EmptyScene);
        let ray = Ray::new(glam::Vec3::ZERO, glam::Vec3::X, 0.0);
        let mut sampler = TestSampler;
        let result = integrator.li(scene.as_ref(), &ray, &mut sampler, &NoWavelengths);
        assert_eq!(result, SpectrumRgb::new(1.0, 0.0, 0.0));
    }
}
