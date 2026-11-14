//! Lambert（理想漫反射）材质。
//!
//! 构造单个 `Diffuse` BxDF，包装成 `BSDF`。漫反射与 `wo` 无关（`f` 恒为
//! `albedo / π`），但 `BSDF::new` 需要 `wo` 构造局部坐标系。
//!
//! # 双面语义
//!
//! 如果 `wo` 与 `shading_normal` 反向（光线从表面背面照射），翻转
//! `shading_normal` 使 `wo` 落在法线正向。这是标准漫反射的双面行为——
//! 光线从两面都能照到漫反射面（如纸张、薄片）。
//!
//! 如果要单面材质（背面不可见），构造 `LambertBSDFBuilder` 时用
//! `single_sided(true)`，`build` 时反向 `wo` 返回零 BSDF。

use std::sync::Arc;

use glam::Vec3;

use kzray_abs::bsdf::{BSDF, BxDF, BxDFEntry, Diffuse};
use kzray_abs::material::BSDFBuilder;
use kzray_abs::ray::SurfaceHit;
use kzray_abs::sampler::Sampler;
use kzray_abs::spectrum::SpectralAlgebra;

// ═══════════════════════════════════════════════════════════════
// LambertBSDFBuilder
// ═══════════════════════════════════════════════════════════════

/// Lambert 材质构建器。
///
/// 持有 `albedo`，`build` 时构造 `Diffuse` BxDF 并包装成 `BSDF`。
///
/// # 构造
///
/// ```text
/// let material = LambertBSDFBuilder::<SpectrumRgb>::new(SpectrumRgb::new(0.8, 0.2, 0.2));
/// ```
///
/// # 单面 / 双面
///
/// 默认双面：`wo` 与法线反向时翻转法线。单面模式通过
/// `single_sided(true)` 开启，反向 `wo` 返回零反射的 BSDF（光线穿透）。
pub struct LambertBSDFBuilder<S: SpectralAlgebra> {
    albedo: S,
    single_sided: bool,
}

impl<S: SpectralAlgebra> LambertBSDFBuilder<S> {
    /// 构造 Lambert 材质（双面）。
    ///
    /// `albedo` 是漫反射率，通常在 `[0, 1]` 范围内。可以为负或大于 1
    /// （物理上非现实，但渲染器不禁止——hack 用途）。
    pub fn new(albedo: S) -> Self {
        LambertBSDFBuilder {
            albedo,
            single_sided: false,
        }
    }

    /// 设置单面 / 双面。
    ///
    /// - `false`（默认）：双面。`wo` 与法线反向时翻转法线，光线从两面
    ///   都能照到。
    /// - `true`：单面。`wo` 与法线反向时返回零反射 BSDF。
    pub fn single_sided(mut self, single_sided: bool) -> Self {
        self.single_sided = single_sided;
        self
    }

    /// 当前 albedo。
    pub fn albedo(&self) -> S {
        self.albedo
    }
}

impl<S: SpectralAlgebra> BSDFBuilder<S> for LambertBSDFBuilder<S> {
    fn build(
        &self,
        hit: &SurfaceHit,
        wo: Vec3,
        _sampler: &mut dyn Sampler,
        _wavelengths: &S::WavelengthContext,
    ) -> BSDF<S> {
        let normal = hit.shading_normal;
        let wo_positive = wo.dot(normal) > 0.0;

        // 单面材质 + 反向 wo：返回零反射 BSDF（光线穿透）
        if self.single_sided && !wo_positive {
            // 返回一个"反射率为零"的 BSDF。用 Diffuse::new(S::zero()) 即可。
            // 注意：这里用零 albedo 而不是返回 Err——BSDF::new 要求 wo.z > 0，
            // 所以还是要用正向法线。反射率为零的 Diffuse 对所有方向返回 0。
            let zero_bxdf: Arc<dyn BxDF<S>> = Arc::new(Diffuse::new(S::zero()));
            return BSDF::new(
                vec![BxDFEntry::new(zero_bxdf, 1.0, S::zero())],
                wo,
                normal )
                .expect("zero BSDF construction should succeed");
        }

        // 双面材质 + 反向 wo：翻转法线使 wo 落在正向
        let normal = if wo_positive { normal } else { -normal };

        let bxdf: Arc<dyn BxDF<S>> = Arc::new(Diffuse::new(self.albedo));
        BSDF::new(vec![BxDFEntry::new(bxdf, 1.0, S::one())],
            wo, normal)
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
    use kzray_abs::ScalarAlgebra;
use kzray_abs::bsdf::TransportMode;
    use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
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

    struct TestSampler;
    impl Sampler for TestSampler {
        fn start_pixel_sample(&mut self, _pixel: glam::UVec2, _sample_index: u32) {}
        fn next_f32(&mut self, _dimension: u32) -> f32 {
            0.5
        }
    }

    #[test]
    fn front_side_produces_bsdf() {
        let material =
            LambertBSDFBuilder::<SpectrumRgb>::new(SpectrumRgb::new(0.8, 0.2, 0.2));
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let bsdf = material.build(&hit, Vec3::Y, &mut sampler, &NoWavelengths);
        assert!(bsdf.has_reflection());
        // eval 返回非零
        let wi = Vec3::Y;
        let f = bsdf.eval(wi, TransportMode::Radiance);
        assert!(f.0[0] > 0.0);
    }

    #[test]
    fn lambert_eval_is_albedo_over_pi() {
        let albedo = SpectrumRgb::new(0.8, 0.4, 0.2);
        let material = LambertBSDFBuilder::<SpectrumRgb>::new(albedo);
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let bsdf = material.build(&hit, Vec3::Y, &mut sampler, &NoWavelengths);

        let wi = Vec3::Y;
        let f = bsdf.eval(wi, TransportMode::Radiance);
        let expected_inv_pi = 1.0 / std::f32::consts::PI;
        assert!((f.0[0] - albedo.0[0] * expected_inv_pi).abs() < 1e-5);
        assert!((f.0[1] - albedo.0[1] * expected_inv_pi).abs() < 1e-5);
        assert!((f.0[2] - albedo.0[2] * expected_inv_pi).abs() < 1e-5);
    }

    #[test]
    fn back_side_double_sided_flips_normal() {
        // 双面材质：wo 与法线反向时仍然能构造 BSDF，光线从背面照射
        let material =
            LambertBSDFBuilder::<SpectrumRgb>::new(SpectrumRgb::new(0.8, 0.2, 0.2));
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        // wo 与法线反向
        let bsdf = material.build(&hit, -Vec3::Y, &mut sampler, &NoWavelengths);
        assert!(bsdf.has_reflection());
    }

    #[test]
    #[should_panic(expected = "WoOnWrongSide")]
    fn back_side_single_sided_returns_zero() {
        let material = LambertBSDFBuilder::<SpectrumRgb>::new(SpectrumRgb::new(0.8, 0.2, 0.2))
            .single_sided(true);
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        // wo 和 法线反向 panic
        let bsdf = material.build(&hit, -Vec3::Y, &mut sampler, &NoWavelengths);
        // eval 在正向方向应返回零
        let wi = Vec3::Y;
        let f = bsdf.eval(wi, TransportMode::Radiance);
        
        assert_eq!(f, SpectrumRgb::zero());
    }

    #[test]
    fn oblique_wo_still_works() {
        // 斜向 wo 也能构造 BSDF（双面）
        let material =
            LambertBSDFBuilder::<SpectrumRgb>::new(SpectrumRgb::new(0.8, 0.2, 0.2));
        let hit = dummy_hit();
        let mut sampler = TestSampler;
        let wo = Vec3::new(0.5, 0.5, 0.7).normalize(); // 与 Y 夹角 ~45°
        let bsdf = material.build(&hit, wo, &mut sampler, &NoWavelengths);
        assert!(bsdf.has_reflection());
    }

    #[test]
    fn albedo_getter() {
        let albedo = SpectrumRgb::new(0.5, 0.5, 0.5);
        let material = LambertBSDFBuilder::<SpectrumRgb>::new(albedo);
        assert_eq!(material.albedo(), albedo);
    }

    #[test]
    fn builder_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LambertBSDFBuilder<SpectrumRgb>>();
    }

    #[test]
    fn builder_is_object_safe() {
        let _: Arc<dyn BSDFBuilder<SpectrumRgb>> =
            Arc::new(LambertBSDFBuilder::<SpectrumRgb>::new(SpectrumRgb::ONE));
    }
}
