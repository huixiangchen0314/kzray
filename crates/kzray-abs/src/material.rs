//! 运行时材质：`BSDFBuilder` / `BXDFBuilder` trait + `Material` 枚举。
//!
//! # 与数据层的关系
//!
//! 数据层 [`kzray_data::shading::Material`] 是**纯数据**：
//!
//! ```text
//! Material { bsdf: FieldExpr }    // 可序列化，无工厂
//! ```
//!
//! 本模块的 [`Material<S>`] 是**运行时材质**：构建层（实现层）把数据层
//! `Material` 的 `FieldExpr` 求值成具体 [`BSDFBuilder`] 实现，再包装成
//! [`Material::BSDF`]。这个转换**不在契约层**，不在数据层——是构建层
//! 的职责。契约层只定义"运行时材质是什么形态"和"如何构造 BSDF"。
//!
//! # 为什么用枚举而不是 trait
//!
//! 运行时材质的"种类"是**封闭域**（BSDF、BSSRDF、未来可能的其他）。
//! 用枚举表达封闭域：
//!
//! - **签名稳定**：`Scene::material` 返回 `Option<&Material<S>>`，加新
//!   变体不改签名；
//! - **穷尽匹配**：调用方 match 时编译器强制处理所有变体；
//! - **无虚分派开销**：`Material` 的分派是 enum 的 `match`，不是 vtable。
//!
//! 对比"`Scene::material` 返回 `&dyn BSDFBuilder<S>`"：如果未来加
//! BSSRDF，签名需要改或加平行方法（`material_bssrdf`），破坏接口。
//!
//! # 为什么叫 `BSDFBuilder` 而不是 `Material`
//!
//! `Material` 这个名字把语义锁死在 BSDF 上。未来加 BSSRDF 时：
//!
//! - `Material` → 语义模糊（是 BSDF 材质还是 BSSRDF 材质？）
//! - `BSDFBuilder` → 明确构造 BSDF；`BSSRDFBuilder` 自然扩展
//!
//! 数据层 `Material` 仍然是 `Material`（"材质"的概念在数据层是清晰的：
//! 一个包含 BSDF 表达式的实体）；契约层的 trait 是"构造什么"的接口，
//! 名字应反映构造目标。运行时材质用 [`Material<S>`] 枚举封装不同 builder，
//! 是"材质"概念在运行时的统一入口。

use std::sync::Arc;

use crate::bsdf::{BxDF, BSDF};
use crate::ray::SurfaceHit;
use crate::sampler::Sampler;
use crate::spectrum::SpectralAlgebra;

// ═══════════════════════════════════════════════════════════════
// BSDFBuilder
// ═══════════════════════════════════════════════════════════════

/// 完整 BSDF 的构建器。对应数据层"BSDF 类节点"。
///
/// 从表面命中 `hit` 构造 [`BSDF<S>`]。`BSDF` 是多个 [`BxDF`] 的组合，
/// 由构建器决定组合哪些分量、权重如何分配。
///
/// # 运行时工厂，不可序列化
///
/// 本 trait 是**运行时接口**——实现持有求值后的纹理、常数、节点图结果，
/// 不参与序列化。数据层的 [`kzray_data::shading::Material`] 才是可序列化
/// 的"材质声明"，由构建层转换成 `BSDFBuilder` 实现。
///
/// # 每个着色点一次
///
/// `build` 每个着色点调用一次，不是热路径。`sampler` 用 `&mut dyn Sampler`
/// 而非 `&mut impl Sampler`：`Scene::material` 返回 `&Material`，`Material`
/// 内部持有 `Arc<dyn BSDFBuilder>`，`BSDFBuilder` 必须 object-safe。
/// 虚分派开销相比 BSDF 求值可忽略。
///
/// # Object-safety
///
/// `BSDFBuilder` 是 object-safe 的：`build` 不是泛型方法。
/// `Box<dyn BSDFBuilder<S>>` / `Arc<dyn BSDFBuilder<S>>` 可用。
pub trait BSDFBuilder<S: SpectralAlgebra>: Send + Sync {
    /// 从表面命中构造 BSDF。
    ///
    /// 参数：
    /// - `hit`：表面命中信息（位置、法线、UV、instance_id 等）；
    /// - `sampler`：采样器。需要随机选择的材质（如随机选层的混合材质）
    ///   从这里取样本；纯确定性材质忽略它；
    /// - `wavelengths`：波长上下文，只读取。
    ///
    /// 返回的 `BSDF<S>` 已构造完成，调用方可以直接 `sample_f` / `eval` /
    /// `pdf`。
    fn build(
        &self,
        hit: &SurfaceHit,
        sampler: &mut dyn Sampler,
        wavelengths: &S::WavelengthContext,
    ) -> BSDF<S>;
}

// ═══════════════════════════════════════════════════════════════
// BXDFBuilder
// ═══════════════════════════════════════════════════════════════

/// 单个 BxDF 的构建器。对应数据层"BxDF 类节点"。
///
/// 从表面命中 `hit` 构造单个 [`BxDF<S>`]。返回 `Arc<dyn BxDF<S>>` 而非
/// 具体类型：`BSDF::new` 需要 `Vec<(Arc<dyn BxDF<S>>, f32)>`，构建器
/// 直接产出 `Arc` 便于组合。
///
/// # 与 `BSDFBuilder` 的分工
///
/// - `BXDFBuilder` 产生**单个分量**（漫反射、微表面、镜面）；
/// - `BSDFBuilder` 产生**组合**（一个或多个 BxDF + 权重）。
///
/// 组合 BSDF 的实现（如 `PrincipledBSDF`）内部持有多个 `BXDFBuilder`，
/// 在 `build` 时逐个构造 BxDF 再组合。契约层不强制这个关系——一个
/// `BSDFBuilder` 实现可以直接构造单一 BxDF 包装成 `BSDF`，跳过
/// `BXDFBuilder`。
///
/// # Object-safety
///
/// `BXDFBuilder` 是 object-safe 的：`build` 不是泛型方法。
/// `Box<dyn BXDFBuilder<S>>` / `Arc<dyn BXDFBuilder<S>>` 可用。
pub trait BXDFBuilder<S: SpectralAlgebra>: Send + Sync {
    /// 从表面命中构造单个 BxDF。
    ///
    /// 参数与 [`BSDFBuilder::build`] 相同；返回 `Arc<dyn BxDF<S>>` 便于
    /// `BSDF::new` 直接组合。
    fn build(
        &self,
        hit: &SurfaceHit,
        sampler: &mut dyn Sampler,
        wavelengths: &S::WavelengthContext,
    ) -> Arc<dyn BxDF<S>>;
}

// ═══════════════════════════════════════════════════════════════
// Material 枚举
// ═══════════════════════════════════════════════════════════════

/// 运行时材质。
///
/// 封装不同类型的材质构建器。当前只有 `BSDF` 变体；未来加 BSSRDF 时
/// 新增变体，`Scene::material` 的签名不变——这是用枚举而非
/// `&dyn BSDFBuilder` 的原因。
///
/// # 内部持有 `Arc`
///
/// 每个变体持有 `Arc<dyn ...Builder<S>>`：材质是场景级数据，被多个
/// 着色点共享；`Arc` 保证引用计数开销低且线程安全。`Scene::material`
/// 返回 `&Material<S>`，不需要克隆 `Arc`。
///
/// # Send + Sync
///
/// `Arc<dyn BSDFBuilder<S>>` 要求 `BSDFBuilder: Send + Sync`（已满足），
/// 所以 `Material` 自动 `Send + Sync`。`Scene` 要求 `Send + Sync`，
/// `Material` 满足这个约束。
pub enum Material<S: SpectralAlgebra> {
    /// BSDF 材质。构造完整 [`BSDF`]。
    BSDF(Arc<dyn BSDFBuilder<S>>),
    // 未来扩展：
    // BSSRDF(Arc<dyn BSSRDFBuilder<S>>),
    // 或者更一般的混合材质：
    // Mixed { bsdf: Option<Arc<dyn BSDFBuilder<S>>>, bssrdf: ... },
}

impl<S: SpectralAlgebra> Material<S> {
    /// 从 `BSDFBuilder` 构造 BSDF 材质。
    pub fn bsdf(builder: Arc<dyn BSDFBuilder<S>>) -> Self {
        Material::BSDF(builder)
    }

    /// 如果是 BSDF 材质，返回 `&dyn BSDFBuilder<S>`。
    ///
    /// 用于需要直接访问构建器的场景（如 BDPT 需要按材质类型分派）。
    /// 普通 PT 用 [`Self::build_bsdf`] 即可。
    ///
    /// 当前只有 `BSDF` 变体，所以总是返回 `Some`。未来加 BSSRDF 时，
    /// 对 BSSRDF 变体返回 `None`。
    pub fn as_bsdf_builder(&self) -> Option<&dyn BSDFBuilder<S>> {
        match self {
            Material::BSDF(b) => Some(b.as_ref()),
        }
    }

    /// 构造 BSDF。如果不是 BSDF 材质，返回 `None`。
    ///
    /// 普通 PT 只处理 BSDF 材质；BSSRDF 材质的路径追踪需要不同处理
    /// （次表面散射），由积分器按材质类型分派。
    ///
    /// 当前只有 `BSDF` 变体，所以总是返回 `Some`。未来加 BSSRDF 时，
    /// 对 BSSRDF 变体返回 `None`。
    pub fn build_bsdf(
        &self,
        hit: &SurfaceHit,
        sampler: &mut dyn Sampler,
        wavelengths: &S::WavelengthContext,
    ) -> Option<BSDF<S>> {
        match self {
            Material::BSDF(b) => Some(b.build(hit, sampler, wavelengths)),
        }
    }

    /// 是否是 BSDF 材质。
    ///
    /// BDPT 等需要按材质类型分派的积分器用它做快速判断。
    pub fn is_bsdf(&self) -> bool {
        matches!(self, Material::BSDF(_))
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsdf::{Diffuse, BxDFEntry};
    use crate::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
    use glam::{Vec2, Vec3};
    use kzray_data::common::InstanceId;

    fn dummy_hit() -> SurfaceHit {
        SurfaceHit {
            t: 1.0,
            distance: 1.0,
            position: Vec3::ZERO,
            shading_normal: Vec3::Y,
            geometric_normal: Vec3::Y,
            uv: Vec2::ZERO,
            instance_id: InstanceId::from(0),
            prim_id: 0,
        }
    }

    // ─────────── 测试用 Builder ───────────

    struct ConstantBSDFBuilder {
        albedo: SpectrumRgb,
    }

    impl BSDFBuilder<SpectrumRgb> for ConstantBSDFBuilder {
        fn build(
            &self,
            _hit: &SurfaceHit,
            _sampler: &mut dyn Sampler,
            _wavelengths: &NoWavelengths,
        ) -> BSDF<SpectrumRgb> {
            let bxdf: Arc<dyn BxDF<SpectrumRgb>> = Arc::new(Diffuse::new(self.albedo));
            BSDF::new(vec![BxDFEntry::new (bxdf, 1.0, SpectrumRgb::from_array([1.0, 1.0, 1.0]))],
                Vec3::Y, Vec3::Y)
                .expect("BSDF construction should succeed")
        }
    }

    struct ConstantBXDFBuilder {
        albedo: SpectrumRgb,
    }

    impl BXDFBuilder<SpectrumRgb> for ConstantBXDFBuilder {
        fn build(
            &self,
            _hit: &SurfaceHit,
            _sampler: &mut dyn Sampler,
            _wavelengths: &NoWavelengths,
        ) -> Arc<dyn BxDF<SpectrumRgb>> {
            Arc::new(Diffuse::new(self.albedo))
        }
    }

    struct TestSampler;
    impl Sampler for TestSampler {
        fn next_f32(&mut self, _dimension: u32) -> f32 {
            0.5
        }
    }

    // ─────────── BSDFBuilder ───────────

    #[test]
    fn bsdf_builder_produces_usable_bsdf() {
        let builder = ConstantBSDFBuilder {
            albedo: SpectrumRgb::new(0.8, 0.2, 0.2),
        };
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let bsdf = builder.build(&hit, &mut sampler, &NoWavelengths);
        let wi = Vec3::new(0.0, 1.0, 0.0);
        let value = bsdf.eval(wi, crate::bsdf::TransportMode::Radiance);
        assert!(value.0[0] > 0.0);
    }

    #[test]
    fn bsdf_builder_is_object_safe() {
        let builder = ConstantBSDFBuilder {
            albedo: SpectrumRgb::ONE,
        };
        let _: Box<dyn BSDFBuilder<SpectrumRgb>> = Box::new(builder);
    }

    #[test]
    fn bsdf_builder_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ConstantBSDFBuilder>();
    }

    // ─────────── BXDFBuilder ───────────

    #[test]
    fn bxdf_builder_produces_usable_bxdf() {
        let builder = ConstantBXDFBuilder {
            albedo: SpectrumRgb::new(0.8, 0.2, 0.2),
        };
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let bxdf = builder.build(&hit, &mut sampler, &NoWavelengths);
        assert!(bxdf.flags().is_diffuse());
        assert!(bxdf.flags().is_valid());
    }

    #[test]
    fn bxdf_builder_is_object_safe() {
        let builder = ConstantBXDFBuilder {
            albedo: SpectrumRgb::ONE,
        };
        let _: Box<dyn BXDFBuilder<SpectrumRgb>> = Box::new(builder);
    }

    // ─────────── Material 枚举 ───────────

    #[test]
    fn material_bsdf_constructor() {
        let builder: Arc<dyn BSDFBuilder<SpectrumRgb>> = Arc::new(ConstantBSDFBuilder {
            albedo: SpectrumRgb::ONE,
        });
        let material = Material::bsdf(builder);
        assert!(material.is_bsdf());
    }

    #[test]
    fn material_as_bsdf_builder_returns_some_for_bsdf() {
        let material = Material::bsdf(Arc::new(ConstantBSDFBuilder {
            albedo: SpectrumRgb::ONE,
        }));
        assert!(material.as_bsdf_builder().is_some());
    }

    #[test]
    fn material_build_bsdf_works() {
        let material = Material::bsdf(Arc::new(ConstantBSDFBuilder {
            albedo: SpectrumRgb::new(0.8, 0.2, 0.2),
        }));
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let bsdf = material
            .build_bsdf(&hit, &mut sampler, &NoWavelengths)
            .expect("BSDF material should build");
        let wi = Vec3::new(0.0, 1.0, 0.0);
        let value = bsdf.eval(wi, crate::bsdf::TransportMode::Radiance);
        assert!(value.0[0] > 0.0);
    }

    #[test]
    fn material_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Material<SpectrumRgb>>();
    }

    #[test]
    fn material_can_be_matched() {
        let material = Material::bsdf(Arc::new(ConstantBSDFBuilder {
            albedo: SpectrumRgb::ONE,
        }));
        match material {
            Material::BSDF(_) => {} // 当前唯一变体
        }
    }
}
