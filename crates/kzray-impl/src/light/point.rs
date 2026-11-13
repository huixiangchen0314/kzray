//! 点光源。
//!
//! 位置上的 delta 分布。方向连续，但采样是确定的（从着色点指向光源），
//! 不需要 2D 样本。`sample_li` 忽略 `sample` 参数。
//!
//! # `pdf` 的语义
//!
//! 点光源的 `LiSample::pdf` 无有限值——方向由几何唯一确定，是 delta。
//! 契约要求 `pdf: f32` 字段存在，实现填 0.0。调用方依据
//! `LightType::is_delta() == true` 跳过 MIS 读取。
//!
//! # `phi`
//!
//! 总辐射功率 = `intensity × 4π`（假设全向发射）。用于 `PowerLightSampler`
//! 构造时的权重计算。返回光谱值 `S`，标量转换由采样器显式完成。

use glam::{Vec2, Vec3};

use kzray_abs::light::{
    LiGeometry, LiSample, Light, LightSampleContext, LightType,
};
use kzray_abs::spectrum::SpectralAlgebra;

// ═══════════════════════════════════════════════════════════════
// PointLight
// ═══════════════════════════════════════════════════════════════

/// 全向点光源。
///
/// `intensity` 是辐射强度（每立体角功率，W/sr），通常按光谱存储。
pub struct PointLight<S: SpectralAlgebra> {
    position: Vec3,
    intensity: S,
}

impl<S: SpectralAlgebra> PointLight<S> {
    /// 构造点光源。
    ///
    /// `intensity` 是辐射强度（W/sr）。全向发射时总功率 =
    /// `intensity × 4π`。
    pub fn new(position: Vec3, intensity: S) -> Self {
        PointLight { position, intensity }
    }

    /// 光源位置。
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// 辐射强度。
    pub fn intensity(&self) -> S {
        self.intensity
    }
}

impl<S: SpectralAlgebra> Light<S> for PointLight<S> {
    fn light_type(&self) -> LightType {
        LightType::DeltaPosition
    }

    fn sample_li(
        &self,
        ctx: &LightSampleContext,
        _sample: Vec2,
        _wavelengths: &S::WavelengthContext,
    ) -> Option<LiSample<S>> {
        let to_light = self.position - ctx.position;
        let distance_sq = to_light.length_squared();
        // 着色点与光源重合，方向退化
        if distance_sq <= f32::EPSILON {
            return None;
        }
        let distance = distance_sq.sqrt();
        let wi = to_light / distance;

        // L_i = intensity / d²（辐射亮度按距离衰减）
        let incident = self.intensity.scale(S::Scalar::from(1.0 / distance_sq));

        Some(LiSample {
            geometry: LiGeometry::Surface {
                position: self.position,
                distance,
            },
            wi,
            incident,
            // delta：无有限值。调用方依据 is_delta() 跳过
            pdf: 0.0,
        })
    }

    fn pdf_li(&self, _ctx: &LightSampleContext, _wi: Vec3) -> f32 {
        // delta：连续方向采样采不到
        0.0
    }

    fn phi(&self, _wavelengths: &S::WavelengthContext) -> S {
        // 全向发射：phi = intensity × 4π
        self.intensity
            .scale(S::Scalar::from(4.0 * std::f32::consts::PI))
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb};

    fn make_light() -> PointLight<SpectrumRgb> {
        PointLight::new(
            Vec3::new(1.0, 2.0, 3.0),
            SpectrumRgb::new(10.0, 10.0, 10.0),
        )
    }

    fn make_ctx(pos: Vec3) -> LightSampleContext {
        LightSampleContext {
            position: pos,
            normal: Vec3::Y,
        }
    }

    #[test]
    fn light_type_is_delta_position() {
        let light = make_light();
        assert_eq!(light.light_type(), LightType::DeltaPosition);
        assert!(light.light_type().is_delta());
    }

    #[test]
    fn sample_li_points_toward_light() {
        let light = make_light();
        let ctx = make_ctx(Vec3::ZERO);
        let s = light
            .sample_li(&ctx, Vec2::ZERO, &NoWavelengths)
            .expect("sample should succeed");
        // wi 应该指向 (1,2,3)
        let expected = Vec3::new(1.0, 2.0, 3.0).normalize();
        assert!((s.wi - expected).length() < 1e-5);
    }

    #[test]
    fn sample_li_distance_correct() {
        let light = PointLight::<SpectrumRgb>::new(
            Vec3::new(0.0, 0.0, 5.0),
            SpectrumRgb::ONE,
        );
        let ctx = make_ctx(Vec3::ZERO);
        let s = light
            .sample_li(&ctx, Vec2::ZERO, &NoWavelengths)
            .unwrap();
        match s.geometry {
            LiGeometry::Surface { position, distance } => {
                assert!((position - Vec3::new(0.0, 0.0, 5.0)).length() < 1e-5);
                assert!((distance - 5.0).abs() < 1e-5);
            }
            LiGeometry::Direction => panic!("point light should have Surface geometry"),
        }
    }

    #[test]
    fn sample_li_inverse_square_falloff() {
        let light = PointLight::<SpectrumRgb>::new(
            Vec3::ZERO,
            SpectrumRgb::new(1.0, 1.0, 1.0),
        );
        let ctx = make_ctx(Vec3::new(2.0, 0.0, 0.0));
        let s = light
            .sample_li(&ctx, Vec2::ZERO, &NoWavelengths)
            .unwrap();
        // L_i = 1 / 4 = 0.25
        assert!((s.incident.0[0] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn sample_li_degenerate_returns_none() {
        let light = PointLight::<SpectrumRgb>::new(Vec3::ZERO, SpectrumRgb::ONE);
        let ctx = make_ctx(Vec3::ZERO); // 与光源重合
        assert!(light.sample_li(&ctx, Vec2::ZERO, &NoWavelengths).is_none());
    }

    #[test]
    fn sample_li_pdf_is_zero() {
        let light = make_light();
        let ctx = make_ctx(Vec3::ZERO);
        let s = light
            .sample_li(&ctx, Vec2::ZERO, &NoWavelengths)
            .unwrap();
        assert_eq!(s.pdf, 0.0);
    }

    #[test]
    fn pdf_li_is_zero() {
        let light = make_light();
        let ctx = make_ctx(Vec3::ZERO);
        assert_eq!(light.pdf_li(&ctx, Vec3::Y), 0.0);
    }

    #[test]
    fn phi_is_intensity_times_4_pi() {
        let light = PointLight::<SpectrumRgb>::new(
            Vec3::ZERO,
            SpectrumRgb::new(2.0, 2.0, 2.0),
        );
        let phi = light.phi(&NoWavelengths);
        let expected = 2.0 * 4.0 * std::f32::consts::PI;
        assert!((phi.0[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn light_is_object_safe() {
        let _: Box<dyn Light<SpectrumRgb>> = Box::new(make_light());
    }

    #[test]
    fn light_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PointLight<SpectrumRgb>>();
    }
}
