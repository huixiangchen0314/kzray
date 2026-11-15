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
//! # 为什么 `build` 需要 `wo`
//!
//! `BSDF::new(entries, wo_world, normal_world)` 需要 `wo_world`（着色点
//! 出射方向）构造局部坐标系。`SurfaceHit` 不含光线方向——它是纯求交
//! 结果，与光线无关。所以 `build` 显式接收 `wo: Vec3`，由调用方（积分器）
//! 从当前光线计算：`wo = -ray.dir.normalize()`。
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

use glam::Vec3;

use crate::bsdf::{BxDF, BSDF};
use crate::ray::SurfaceHit;
use crate::sampler::Sampler;
use crate::spectrum::SpectralAlgebra;
use crate::library::Library;

// ═══════════════════════════════════════════════════════════════
// BSDFBuilder
// ═══════════════════════════════════════════════════════════════

/// 完整 BSDF 的构建器。对应数据层"BSDF 类节点"。
///
/// 从表面命中 `hit` 和出射方向 `wo` 构造 [`BSDF<S>`]。`BSDF` 是多个
/// [`BxDF`] 的组合，由构建器决定组合哪些分量、权重如何分配。
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
    /// # 参数
    ///
    /// - `hit`：表面命中信息（位置、法线、UV、instance_id 等）；
    /// - `wo`：着色点出射方向（指向相机/上一跳），世界空间，已归一化。
    ///   等于 `-ray.dir.normalize()`；
    /// - `sampler`：采样器，随机材质从这里取样本；
    /// - `wavelengths`：波长上下文，只读；
    /// - `lib`：资产库，材质参数从这里的纹理求值。
    fn build(
        &self,
        hit: &SurfaceHit,
        wo: Vec3,
        sampler: &mut dyn Sampler,
        wavelengths: &S::WavelengthContext,
        lib: &Library<S>,
    ) -> BSDF<S>;

}

// ═══════════════════════════════════════════════════════════════
// BXDFBuilder
// ═══════════════════════════════════════════════════════════════

/// 单个 BxDF 的构建器。对应数据层"BxDF 类节点"。
///
/// 从表面命中 `hit` 和出射方向 `wo` 构造单个 [`BxDF<S>`]。返回
/// `Arc<dyn BxDF<S>>` 而非具体类型：`BSDF::new` 需要
/// `Vec<(Arc<dyn BxDF<S>>, f32)>`，构建器直接产出 `Arc` 便于组合。
///
/// # 为什么需要 `wo`
///
/// 部分 BxDF 依赖 `wo` 做构造期决策——如 Fresnel 混合的两层 BxDF
/// （反射/透射比由 `wo` 与法线夹角决定）、方向依赖的归一化因子。
/// 不接收 `wo` 的 BxDF 忽略它即可。
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
        wo: Vec3,
        sampler: &mut dyn Sampler,
        wavelengths: &S::WavelengthContext,
        lib: &Library<S>,
    ) -> Arc<dyn BxDF<S>>;
}

// ═══════════════════════════════════════════════════════════════
// Material 枚举
// ═══════════════════════════════════════════════════════════════

/// 运行时材质。
///
/// 封装不同类型的材质构建器。当前只有 `BSDF` 变体；未来加 BSSRDF 时
/// 新增变体，`Scene::material` 的签名不变。
pub enum Material<S: SpectralAlgebra> {
    /// BSDF 材质。构造完整 [`BSDF`]。
    BSDF(Arc<dyn BSDFBuilder<S>>),
    // 未来扩展：
    // BSSRDF(Arc<dyn BSSRDFBuilder<S>>),
}

impl<S: SpectralAlgebra> Material<S> {
    /// 从 `BSDFBuilder` 构造 BSDF 材质。
    pub fn bsdf(builder: Arc<dyn BSDFBuilder<S>>) -> Self {
        Material::BSDF(builder)
    }

    /// 如果是 BSDF 材质，返回 `&dyn BSDFBuilder<S>`。
    pub fn as_bsdf_builder(&self) -> Option<&dyn BSDFBuilder<S>> {
        match self {
            Material::BSDF(b) => Some(b.as_ref()),
        }
    }

    /// 构造 BSDF。如果不是 BSDF 材质，返回 `None`。
    ///
    /// `lib` 是资产库，材质参数从这里的纹理求值。
    pub fn build_bsdf(
        &self,
        hit: &SurfaceHit,
        wo: Vec3,
        sampler: &mut dyn Sampler,
        wavelengths: &S::WavelengthContext,
        lib: &Library<S>,
    ) -> Option<BSDF<S>> {
        match self {
            Material::BSDF(b) => Some(b.build(hit, wo, sampler, wavelengths, lib)),
        }
    }

    /// 是否是 BSDF 材质。
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
    use crate::bsdf::{BxDFEntry, Diffuse};
    use crate::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
    use crate::texture::{AlbedoChannel, Texture};
    use glam::{Vec2, Vec3};
    use kzray_data::TextureId;

    // 简化测试纹理——测试只验证接口衔接，不验证纹理实现
    struct FakeAlbedoTexture(SpectrumRgb);
    impl Texture<AlbedoChannel<SpectrumRgb>> for FakeAlbedoTexture {
        fn eval(&self, _uv: Vec2, _d: Option<&crate::texture::Differential>) -> SpectrumRgb {
            self.0
        }
    }

    fn empty_lib_with_albedo(id: TextureId, albedo: SpectrumRgb) -> Library<SpectrumRgb> {
        let mut lib = Library::<SpectrumRgb>::new();
        lib.add_albedo(id, Box::new(FakeAlbedoTexture(albedo)));
        lib
    }

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

    struct ConstantBSDFBuilder {
        albedo: TextureId,
    }

    impl BSDFBuilder<SpectrumRgb> for ConstantBSDFBuilder {
        fn build(
            &self,
            hit: &SurfaceHit,
            wo: Vec3,
            _sampler: &mut dyn Sampler,
            _wavelengths: &NoWavelengths,
            lib: &Library<SpectrumRgb>,
        ) -> BSDF<SpectrumRgb> {
            let albedo = lib.albedo(self.albedo).eval(hit.uv, None);
            let bxdf: Arc<dyn BxDF<SpectrumRgb>> = Arc::new(Diffuse::new(albedo));
            BSDF::new(
                vec![BxDFEntry::new(bxdf, 1.0, SpectrumRgb::from_array([1.0, 1.0, 1.0]))],
                wo,
                Vec3::Y,
            )
            .expect("BSDF construction should succeed")
        }
    }

    struct TestSampler;
    impl Sampler for TestSampler {
        fn start_pixel_sample(&mut self, _pixel: glam::UVec2, _sample_index: u32) {}
        fn next_f32(&mut self, _dimension: u32) -> f32 {
            0.5
        }
    }

    #[test]
    fn bsdf_builder_produces_usable_bsdf() {
        let id = TextureId::from_index(0);
        let lib = empty_lib_with_albedo(id, SpectrumRgb::new(0.8, 0.2, 0.2));
        let builder = ConstantBSDFBuilder { albedo: id };
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let bsdf = builder.build(&hit, Vec3::Y, &mut sampler, &NoWavelengths, &lib);
        let wi = Vec3::new(0.0, 1.0, 0.0);
        let value = bsdf.eval(wi, crate::bsdf::TransportMode::Radiance);
        assert!(value.0[0] > 0.0);
    }

    #[test]
    fn bsdf_builder_is_object_safe() {
        let id = TextureId::from_index(0);
        let builder = ConstantBSDFBuilder { albedo: id };
        let _: Box<dyn BSDFBuilder<SpectrumRgb>> = Box::new(builder);
    }

    #[test]
    fn material_build_bsdf_works() {
        let id = TextureId::from_index(0);
        let lib = empty_lib_with_albedo(id, SpectrumRgb::new(0.8, 0.2, 0.2));
        let material = Material::bsdf(Arc::new(ConstantBSDFBuilder { albedo: id }));
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let bsdf = material
            .build_bsdf(&hit, Vec3::Y, &mut sampler, &NoWavelengths, &lib)
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
}
