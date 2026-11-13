//! 场景只读视图。
//!
//! `Scene` 是**已构建好的可渲染场景**的只读接口。它不是数据层
//! [`kzray_data::input::RenderInput`] 的抽象，也不是构造契约——构建层
//! （实现层）负责把 `RenderInput` 转换成 `Scene` 的实现（BVH、材质求值、
//! 光源实现等）。
//!
//! # 与数据层的关系
//!
//! | 数据层 | `Scene` |
//! |---|---|
//! | `RenderInput.meshes` + `instances` | `intersect` 内部的 BVH |
//! | `RenderInput.materials` | `material(hit) -> Option<&Material>` |
//! | `RenderInput.lights` | `light(id) -> Option<&dyn Light>` |
//! | `LightType::Background` | `infinite_light() -> Option<&dyn InfiniteLight>` |
//! | `WorldData.bounds` | `bounds() -> Aabb` |
//! | `RenderInput.camera` | **不在 `Scene` 里**，由积分器独立持有 |
//!
//! 相机不在 `Scene` 里：PT 的 primary ray 生成独立于场景，BDPT 的
//! 相机侧采样也是独立路径。`Scene` 是"可被光线求交和查询的世界"，
//! 相机是"观察这个世界的设备"。
//!
//! # 为什么 `Scene` 带 `LS: LightSampler` 泛型参数
//!
//! Rust 的 trait dyn 兼容性规则：`LightSampler::sample` 用了
//! `&mut impl Sampler`（单态化以换取性能），使 `LightSampler` **不是
//! dyn compatible**——`&dyn LightSampler` 不可用。
//!
//! `Scene` 需要暴露光源采样器，两个选项：
//!
//! 1. **改 `LightSampler` 为 dyn compatible**：`sample` 用 `&mut dyn Sampler`，
//!    失去单态化。每次 `sample` 内部取 2-3 个 `next_2d` / `next_f32`
//!    会走虚分派，相比 BVH 遍历可忽略——但这是**性能**取舍。
//! 2. **`Scene` 带 `LS` 泛型参数**：`Scene<S, LS>` 保留 `LightSampler`
//!    的完整能力，`light_sampler() -> Option<&LS>` 返回具体类型引用。
//!    代价是 `Scene` 多了泛型参数，`Box<dyn Scene<S, LS>>` 需要指定 `LS`。
//!
//! **当前选择方案 2**：`Scene` 是"最大的上下文承载点"，把不能 dyn 的
//! 类型参数化到它身上，符合"场景承载所有运行时组件"的设计。这是
//! Rust 的固有限制——性能和少泛型只能选其一。
//!
//! 未来如果还有其他 trait 不可 dyn（如某个新的采样器、某个带 `impl Trait`
//! 参数的组件），同样处理：把泛型参数加到 `Scene` 上，而不是把 trait
//! 改成 dyn compatible。**`Scene` 承载所有运行时组件的具体类型**，
//! 这是"上下文"的本质。
//!
//! # Object-safety
//!
//! `Scene<S, LS>` **是** object-safe 的：所有方法都不是泛型方法，
//! `&LS` 是具体类型引用（`LS` 是 trait 泛型参数，编译时确定，不参与
//! 方法级泛型）。`Box<dyn Scene<S, LS>>` / `Arc<dyn Scene<S, LS>>` 可用。
//!
//! 积分器持有 `&dyn Scene<S, LS>`，虚分派每次求交一次，相比 BVH 遍历
//! 可忽略。

use std::ops::Range;

use kzray_data::common::LightId;
use kzray_data::geometry::Aabb;

use crate::light::{InfiniteLight, Light, LightSampler};
use crate::material::Material;
use crate::ray::{Ray, SurfaceHit};
use crate::spectrum::SpectralAlgebra;

/// 场景只读视图。
///
/// 已构建好场景的只读接口：求交、查材质、查光源。构建层把数据层
/// `RenderInput` 转换成 `Scene` 的实现。
///
/// # 泛型参数
///
/// - `S: SpectralAlgebra`：光谱类型；
/// - `LS: LightSampler`：光源采样器类型。因为 `LightSampler` 不可 dyn，
///   所以通过泛型参数传入。见模块文档"为什么 `Scene` 带 `LS` 泛型参数"。
///
/// # 与 `Camera` 的关系
///
/// 相机**不在** `Scene` 里。PT 的 primary ray 由相机生成，积分器持有
/// `&dyn Camera<S>` 和 `&dyn Scene<S, LS>` 两个独立引用。
///
/// # `Option` 返回值的含义
///
/// - `material` / `light` / `light_sampler` / `infinite_light` 返回
///   `Option`：构建层可能有无材质图元、无光源场景。`None` 是正常情况，
///   调用方按需处理（如跳过光源采样、用默认材质）。
pub trait Scene<S, LS>: Send + Sync
where
    S: SpectralAlgebra,
    LS: LightSampler,
{
    /// 世界包围盒。BVH 和裁剪用。
    ///
    /// 构建层从 `WorldData.bounds` 或 instances 聚合得到。空场景返回
    /// `Aabb::empty()`（`is_empty()` 为 `true`）。
    fn bounds(&self) -> Aabb;

    /// 求交。返回最近命中的表面。
    ///
    /// `t_range` 是光线参数的可行区间，通常 `0.0..f32::INFINITY`
    /// （primary ray）或 `epsilon..distance`（阴影射线）。
    ///
    /// 返回 `None` 表示未命中。
    fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<SurfaceHit>;

    /// 可见性查询。返回是否有命中，不构造 `SurfaceHit`。
    ///
    /// 用于阴影射线——比 `intersect` 快，不需要构造完整命中信息。
    /// 返回 `true` 表示**有遮挡**（光线被挡住），与 `intersect().is_some()`
    /// 语义一致。
    fn intersect_p(&self, ray: &Ray, t_range: Range<f32>) -> bool;

    /// 从命中查到运行时材质。
    ///
    /// 返回 `&Material<S>`（枚举），调用方按需 match 或调 `build_bsdf`。
    /// 返回 `None` 表示该图元无材质（构建层的容错，如空材质槽）。
    ///
    /// 用 `&Material<S>` 而非 `&dyn BSDFBuilder<S>`：枚举封装不同类型的
    /// 运行时材质（当前只有 BSDF，未来加 BSSRDF），签名稳定。
    fn material(&self, hit: &SurfaceHit) -> Option<&Material<S>>;

    /// 按 ID 查光源。
    ///
    /// `LightId` 是数据层 `RenderInput.lights` 的键。返回 `None` 表示
    /// ID 无效（构建层应保证所有 `LightId` 可达，此情况是数据错误）。
    ///
    /// `Light<S>` 是 dyn compatible 的（方法无泛型参数），所以返回
    /// `&dyn Light<S>` 而非泛型参数。
    fn light(&self, id: LightId) -> Option<&dyn Light<S>>;

    /// 光源选择器。
    ///
    /// 场景只有一个选择器（策略在构建时固定）。返回 `None` 表示场景
    /// 无光源，积分器跳过光源采样。
    ///
    /// 返回 `&LS`（具体类型）而非 `&dyn LightSampler`：`LightSampler`
    /// 不可 dyn（见模块文档）。`LS` 是 `Scene` 的泛型参数，编译时确定。
    fn light_sampler(&self) -> Option<&LS>;

    /// 无限远光源（环境光 / 天空光）。
    ///
    /// 对应数据层的 `LightType::Background`。构建层应校验至多一个
    /// `Background` 光源；多个时取第一个或报错，由构建层决定。
    ///
    /// 返回 `None` 表示场景无环境光。
    ///
    /// `InfiniteLight<S>` 是 dyn compatible 的（方法无泛型参数），
    /// 所以返回 `&dyn InfiniteLight<S>` 而非泛型参数。
    fn infinite_light(&self) -> Option<&dyn InfiniteLight<S>>;
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::light::{LightSampleContext, LightSelection};
    use crate::sampler::Sampler;
    use crate::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
    use glam::Vec3;

    // ─────────── 测试用 LightSampler ───────────

    /// 空采样器：总是返回 `None`（无光源）。
    struct EmptyLightSampler;

    impl LightSampler for EmptyLightSampler {
        fn sample(
            &self,
            _ctx: &LightSampleContext,
            _sampler: &mut impl Sampler,
            _dim: u32
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

    // ─────────── 测试用 Scene ───────────

    /// 空场景：无几何、无光源、无材质。
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

        fn light(&self, _id: LightId) -> Option<&dyn Light<SpectrumRgb>> {
            None
        }

        fn light_sampler(&self) -> Option<&EmptyLightSampler> {
            None
        }

        fn infinite_light(&self) -> Option<&dyn InfiniteLight<SpectrumRgb>> {
            None
        }
    }

    // ─────────── 契约测试 ───────────

    #[test]
    fn empty_scene_bounds_is_empty() {
        let scene = EmptyScene;
        assert!(scene.bounds().is_empty());
    }

    #[test]
    fn empty_scene_never_hits() {
        let scene = EmptyScene;
        let ray = Ray::new(Vec3::ZERO, Vec3::X, 0.0);
        assert!(scene.intersect(&ray, 0.0..f32::INFINITY).is_none());
        assert!(!scene.intersect_p(&ray, 0.0..f32::INFINITY));
    }

    #[test]
    fn empty_scene_has_no_lights() {
        let scene = EmptyScene;
        assert!(scene.light(LightId::from(0)).is_none());
        assert!(scene.light_sampler().is_none());
        assert!(scene.infinite_light().is_none());
    }

    // ─────────── object-safety ───────────

    #[test]
    fn scene_is_object_safe() {
        let _: Box<dyn Scene<SpectrumRgb, EmptyLightSampler>> = Box::new(EmptyScene);
    }

    #[test]
    fn scene_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EmptyScene>();
    }

    // 抑制 unused import 警告
    #[test]
    fn _use_imports() {
        let _: Option<NoWavelengths> = None;
    }
}
