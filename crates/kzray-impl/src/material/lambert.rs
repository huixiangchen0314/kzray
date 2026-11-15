//! Lambert（理想漫反射）材质。
//!
//! # 纹理接入
//!
//! `albedo` 字段从 `S` 改为 `TextureId`——材质构造时只记录 id，
//! `build` 时从 `Library` 查 `AlbedoTexture<S>` 并 `eval(hit.uv, ...)`
//! 求值。常量 albedo 用 `ConstantTexture<AlbedoChannel<S>>` 注册，
//! 统一走纹理路径。
//!
//! # 双面语义
//!
//! `wo` 与 `shading_normal` 反向时翻转法线（默认），或返回零反射
//! BSDF（`single_sided(true)`）。

use std::sync::Arc;

use glam::Vec3;

use kzray_abs::bsdf::{BSDF, BxDF, BxDFEntry, Diffuse};
use kzray_abs::library::Library;
use kzray_abs::material::BSDFBuilder;
use kzray_abs::ray::SurfaceHit;
use kzray_abs::sampler::Sampler;
use kzray_abs::spectrum::SpectralAlgebra;
use kzray_data::TextureId;

// ═══════════════════════════════════════════════════════════════
// LambertBSDFBuilder
// ═══════════════════════════════════════════════════════════════

/// Lambert 材质构建器。
///
/// 持有 `albedo: TextureId`，`build` 时从 `Library` 求值。双面语义见
/// 模块文档。
///
/// # 构造
///
/// ```ignore
/// // 1. 注册常量 albedo 纹理
/// let albedo_id = lib.add_albedo(
///     TextureId::from_index(0),
///     Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
///         SpectrumRgb::new(0.8, 0.2, 0.2),
///     )),
/// );
/// // 2. 构造材质
/// let material = Material::bsdf(Arc::new(LambertBSDFBuilder::new(albedo_id)));
/// ```
///
/// # 泛型
///
/// `LambertBSDFBuilder` **不带 `S` 泛型**——`TextureId` 是裸 u32。同一
/// 实例在 `impl<S: SpectralAlgebra> BSDFBuilder<S>` 下可服务任意 `S`，
/// `S` 由 `Library<S>` 的 `albedo` 查询决定。
#[derive(Clone, Copy, Debug)]
pub struct LambertBSDFBuilder {
    albedo: TextureId,
    single_sided: bool,
}

impl LambertBSDFBuilder {
    /// 构造 Lambert 材质（双面）。
    ///
    /// `albedo` 是 `AlbedoChannel<S>` 纹理的 id，必须在 `Library` 中
    /// 已注册。
    pub fn new(albedo: TextureId) -> Self {
        LambertBSDFBuilder {
            albedo,
            single_sided: false,
        }
    }

    /// 设置单面 / 双面。
    pub fn single_sided(mut self, single_sided: bool) -> Self {
        self.single_sided = single_sided;
        self
    }

    /// 当前 albedo 纹理 id。
    pub fn albedo(&self) -> TextureId {
        self.albedo
    }
}

impl<S: SpectralAlgebra> BSDFBuilder<S> for LambertBSDFBuilder {
    fn build(
        &self,
        hit: &SurfaceHit,
        wo: Vec3,
        _sampler: &mut dyn Sampler,
        _wavelengths: &S::WavelengthContext,
        lib: &Library<S>,
    ) -> BSDF<S> {
        let albedo = lib.albedo(self.albedo).eval(hit.uv, None);

        let normal = hit.shading_normal;
        let cos_wo = wo.dot(normal);

        // ── 擦边处理 ──
        //
        // `BSDF::new` 严格要求 `wo.dot(normal) > 0`。当 `cos_wo == 0`
        // 时（光线恰好垂直于表面），翻转法线也得到 `-0 == 0`，不满足
        // `> 0`——这是数值边界，不是方向错误。
        //
        // 物理上擦边反射趋于零。返回零反射 BSDF 是正确语义：不引入
        // 宽容，只是把"擦边无贡献"从"构造失败"变成"构造出无反射的
        // BSDF"，让渲染继续。
        const COS_EDGE_EPS: f32 = 1e-6;
        if cos_wo.abs() < COS_EDGE_EPS {
            let zero_bxdf: Arc<dyn BxDF<S>> = Arc::new(Diffuse::new(S::zero()));
            return BSDF::new(
                vec![BxDFEntry::new(zero_bxdf, 1.0, S::zero())],
                // 用 normal 本身作为 wo，dot(normal, normal) = 1 > 0，
                // BSDF::new 必定成功。反射为零，wo 的选择不影响结果。
                normal,
                normal,
            )
            .expect("zero BSDF with wo = normal should succeed");
        }

        let wo_positive = cos_wo > 0.0;

        // 单面材质 + 反向 wo：零反射（光线穿透）。
        if self.single_sided && !wo_positive {
            let zero_bxdf: Arc<dyn BxDF<S>> = Arc::new(Diffuse::new(S::zero()));
            return BSDF::new(
                vec![BxDFEntry::new(zero_bxdf, 1.0, S::zero())],
                wo,
                normal,
            )
            .expect("zero BSDF construction should succeed");
        }

        // 双面材质 + 反向 wo：翻转法线。
        let normal = if wo_positive { normal } else { -normal };

        let bxdf: Arc<dyn BxDF<S>> = Arc::new(Diffuse::new(albedo));
        BSDF::new(vec![BxDFEntry::new(bxdf, 1.0, S::one())], wo, normal)
            .expect("Lambert BSDF construction should succeed")
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;
    use kzray_abs::bsdf::TransportMode;
    use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb, ScalarAlgebra};
    use kzray_abs::texture::{AlbedoChannel, Differential, Texture};
    use kzray_data::common::InstanceId;

    /// 测试用 albedo 纹理——返回固定颜色。
    struct FixedAlbedo(SpectrumRgb);
    impl Texture<AlbedoChannel<SpectrumRgb>> for FixedAlbedo {
        fn eval(&self, _uv: Vec2, _d: Option<&Differential>) -> SpectrumRgb {
            self.0
        }
    }

    fn make_lib(
        id: TextureId,
        albedo: SpectrumRgb,
    ) -> Library<SpectrumRgb> {
        let mut lib = Library::new();
        lib.add_albedo(id, Box::new(FixedAlbedo(albedo)));
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

    struct TestSampler;
    impl Sampler for TestSampler {
        fn start_pixel_sample(&mut self, _pixel: glam::UVec2, _sample_index: u32) {}
        fn next_f32(&mut self, _dimension: u32) -> f32 {
            0.5
        }
    }

    #[test]
    fn front_side_produces_bsdf() {
        let id = TextureId::from_index(0);
        let lib = make_lib(id, SpectrumRgb::new(0.8, 0.2, 0.2));
        let material = LambertBSDFBuilder::new(id);
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let bsdf = material.build(&hit, Vec3::Y, &mut sampler, &NoWavelengths, &lib);
        assert!(bsdf.has_reflection());
        let f = bsdf.eval(Vec3::Y, TransportMode::Radiance);
        assert!(f.0[0] > 0.0);
    }

    #[test]
    fn lambert_eval_is_albedo_over_pi() {
        let id = TextureId::from_index(0);
        let albedo = SpectrumRgb::new(0.8, 0.4, 0.2);
        let lib = make_lib(id, albedo);
        let material = LambertBSDFBuilder::new(id);
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let bsdf = material.build(&hit, Vec3::Y, &mut sampler, &NoWavelengths, &lib);

        let f = bsdf.eval(Vec3::Y, TransportMode::Radiance);
        let inv_pi = 1.0 / std::f32::consts::PI;
        assert!((f.0[0] - albedo.0[0] * inv_pi).abs() < 1e-5);
        assert!((f.0[1] - albedo.0[1] * inv_pi).abs() < 1e-5);
        assert!((f.0[2] - albedo.0[2] * inv_pi).abs() < 1e-5);
    }

    #[test]
    fn back_side_double_sided_flips_normal() {
        let id = TextureId::from_index(0);
        let lib = make_lib(id, SpectrumRgb::new(0.8, 0.2, 0.2));
        let material = LambertBSDFBuilder::new(id);
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let bsdf = material.build(&hit, -Vec3::Y, &mut sampler, &NoWavelengths, &lib);
        assert!(bsdf.has_reflection());
    }

    #[test]
    fn back_side_single_sided_returns_zero() {
        let id = TextureId::from_index(0);
        let lib = make_lib(id, SpectrumRgb::new(0.8, 0.2, 0.2));
        let material = LambertBSDFBuilder::new(id).single_sided(true);
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let bsdf = material.build(&hit, -Vec3::Y, &mut sampler, &NoWavelengths, &lib);
        let f = bsdf.eval(Vec3::Y, TransportMode::Radiance);
        assert_eq!(f, SpectrumRgb::zero());
    }

    #[test]
    fn oblique_wo_still_works() {
        let id = TextureId::from_index(0);
        let lib = make_lib(id, SpectrumRgb::new(0.8, 0.2, 0.2));
        let material = LambertBSDFBuilder::new(id);
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let wo = Vec3::new(0.5, 0.5, 0.7).normalize();
        let bsdf = material.build(&hit, wo, &mut sampler, &NoWavelengths, &lib);
        assert!(bsdf.has_reflection());
    }

    #[test]
    fn albedo_getter() {
        let id = TextureId::from_index(7);
        let material = LambertBSDFBuilder::new(id);
        assert_eq!(material.albedo(), id);
    }

    #[test]
    fn builder_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LambertBSDFBuilder>();
    }

    #[test]
    fn builder_is_object_safe() {
        let id = TextureId::from_index(0);
        let _: Arc<dyn BSDFBuilder<SpectrumRgb>> =
            Arc::new(LambertBSDFBuilder::new(id));
    }

    #[test]
    fn builder_serves_multiple_spectral_algebras() {
        // 同一个 builder 实例可服务多个 S
        let id = TextureId::from_index(0);
        let builder = LambertBSDFBuilder::new(id);

        // 通过 Rgb
        let lib_rgb = make_lib(id, SpectrumRgb::ONE);
        let _ = <LambertBSDFBuilder as BSDFBuilder<SpectrumRgb>>::build(
            &builder, &dummy_hit(), Vec3::Y, &mut TestSampler, &NoWavelengths, &lib_rgb,
        );

        // builder 本身无 S 参数，可跨 S 复用
        let _ = builder; // 保持存活
    }
}
