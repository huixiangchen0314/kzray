use std::sync::Arc;

use glam::Vec3;
use kzray_abs::bsdf::Diffuse;
use kzray_abs::{
    BSDFBuilder, BxDF, BxDFEntry, BxDFFlags, BxDFSample, Library, Sampler, ScalarAlgebra,
    SpectralAlgebra, SurfaceHit, TransportMode, BSDF,
};
use kzray_data::TextureId;
use kzray_math::microfacet;
use kzray_math::microfacet::ggx;

use crate::material::fresnel;

/// GGX 微表面的导体（金属）BRDF。
///
/// # BRDF
///
/// ```text
/// f(wo, wi) = D(h) · G(wo, wi) · F(wo, h) / (4 · cos_o · cos_i)
/// ```
///
/// - `D`：GGX / Trowbridge-Reitz NDF；
/// - `G`：Smith-GGX 高度相关遮蔽；
/// - `F`：Schlick 近似的导体 Fresnel，`F0 = self.f0`。
///
/// # 采样
///
/// VNDF（Heitz 2018）采样微表面法线 `m`，然后 `wi = reflect(wo, m)`。
/// 立体角 PDF 为 `pdf_vndf(m) / (4 · dot(wo, m))`。
///
/// # 粗糙度
///
/// `alpha` 是 GGX 形状参数，不是感知粗糙度。调用方（material builder）
/// 负责 `alpha = roughness²` 的映射。`alpha → 0` 时退化为镜面，但
/// 本 BxDF **不**把 `alpha` 太小的情况标为 delta——为避免数值不稳定，
/// builder 应将 `alpha` clamp 到 `[1e-3, 1.0]`。用户要完美镜面用
/// [`Mirror`]。
///
/// # 能力标签
///
/// `REFLECTION | GLOSSY`——非 delta，参与 NEE。`is_pure_delta() = false`。
///
/// # Albedo
///
/// `hemispherical_reflectance()` 返回 `F0`（近似）。精确的半球反射率
/// 需要积分 Fresnel 全角度分布，比 `F0` 更高（随角度递增）。`F0` 作为
/// 保守下界对能量估计安全。
#[derive(Clone, Copy, Debug)]
pub struct Conductor<S: ScalarAlgebra> {
    /// 垂直入射的 Fresnel 反射率（金属颜色）。
    pub f0: S,
    /// GGX 形状参数，`roughness²`。builder 应 clamp 到 `[1e-3, 1.0]`。
    pub alpha: f32,
}

impl<S: ScalarAlgebra> Conductor<S> {
    pub fn new(f0: S, alpha: f32) -> Self {
        Conductor {
            f0,
            alpha: alpha.clamp(1e-3, 1.0),
        }
    }

    /// 半球-半球反射率近似。导体无透射，全部反射——`F0` 是保守下界。
    pub fn hemispherical_reflectance(&self) -> S {
        self.f0
    }
}

impl<S: ScalarAlgebra> BxDF<S> for Conductor<S> {
    fn flags(&self) -> BxDFFlags {
        // GLOSSY（非 delta），参与 NEE / MIS
        BxDFFlags::REFLECTION | BxDFFlags::GLOSSY
    }

    fn eval(&self, wo: Vec3, wi: Vec3, _mode: TransportMode) -> S {
        // 导体只反射，无透射——任一方向在背面即贡献 0
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return S::zero();
        }
        let h = microfacet::half_vector(wo, wi);
        if h.z <= 0.0 {
            return S::zero();
        }
        let d = ggx::d(h.z, self.alpha);
        let g = ggx::g2(wo.z, wi.z, self.alpha);
        let f = fresnel::fresnel_schlick(&self.f0, wo.dot(h));
        let denom = 4.0 * wo.z * wi.z;
        if denom <= f32::EPSILON {
            return S::zero();
        }
        f.scale(S::Scalar::from(d * g / denom))
    }

    fn sample_f(
        &self,
        wo: Vec3,
        sampler: &mut dyn Sampler,
        dimension: u32,
        mode: TransportMode,
    ) -> Option<BxDFSample<S>> {
        if wo.z <= 0.0 {
            return None;
        }
        let (u1, u2) = sampler.next_2d(dimension);
        let m = ggx::sample_vndf(wo, self.alpha, u1, u2);
        let wi = microfacet::reflect(wo, m);
        if wi.z <= 0.0 {
            return None;
        }
        let cos_om = wo.dot(m);
        if cos_om <= f32::EPSILON {
            return None;
        }

        // 立体角 PDF
        let pdf_m = ggx::pdf_vndf(wo, m, self.alpha);
        let pdf = pdf_m / (4.0 * cos_om);
        if pdf <= f32::EPSILON {
            return None;
        }

        // BRDF 值——直接用 m 算，不再走 half_vector（m 就是 h）
        let d = ggx::d(m.z, self.alpha);
        let g = ggx::g2(wo.z, wi.z, self.alpha);
        let f = fresnel::fresnel_schlick(&self.f0, cos_om);
        let denom = 4.0 * wo.z * wi.z;
        if denom <= f32::EPSILON {
            return None;
        }
        let value = f.scale(S::Scalar::from(d * g / denom));

        let cos_i = wi.z;
        let weight = value.scale(S::Scalar::from(cos_i / pdf));

        let _ = mode;
        Some(BxDFSample {
            wi,
            value,
            weight,
            pdf,
            pdf_is_exact: true,
            flags: self.flags(),
            eta: 1.0,
            consumed: 2,
        })
    }

    fn pdf(&self, wo: Vec3, wi: Vec3, _mode: TransportMode) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        let h = microfacet::half_vector(wo, wi);
        if h.z <= 0.0 {
            return 0.0;
        }
        let cos_om = wo.dot(h);
        if cos_om <= f32::EPSILON {
            return 0.0;
        }
        let pdf_m = ggx::pdf_vndf(wo, h, self.alpha);
        pdf_m / (4.0 * cos_om)
    }

    fn rho(
        &self,
        _wo: Vec3,
        _sampler: &mut dyn Sampler,
        _dimension: u32,
        _mode: TransportMode,
    ) -> (S, u32) {
        (self.f0, 0)
    }

    fn max_sample_dimensions(&self) -> u32 {
        2
    }
}

/// GGX 导体材质构建器。
#[derive(Clone, Copy, Debug)]
pub struct ConductorBSDFBuilder {
    /// F0 纹理（`AlbedoChannel<S>`，金属颜色）。
    f0: TextureId,
    /// 粗糙度纹理（`RoughnessChannel`，感知粗糙度 `[0, 1]`）。
    roughness: TextureId,
}

impl ConductorBSDFBuilder {
    pub fn new(f0: TextureId, roughness: TextureId) -> Self {
        ConductorBSDFBuilder { f0, roughness }
    }

    pub fn f0(&self) -> TextureId {
        self.f0
    }

    pub fn roughness(&self) -> TextureId {
        self.roughness
    }
}

impl<S: SpectralAlgebra> BSDFBuilder<S> for ConductorBSDFBuilder {
    fn build(
        &self,
        hit: &SurfaceHit,
        wo: Vec3,
        _sampler: &mut dyn Sampler,
        _wavelengths: &S::WavelengthContext,
        lib: &Library<S>,
    ) -> BSDF<S> {
        let f0 = lib.albedo(self.f0).eval(hit.uv, None);
        let roughness = lib.roughness(self.roughness).eval(hit.uv, None);
        let alpha = roughness * roughness;


        let original_normal = hit.shading_normal;
        let cos_wo = wo.dot(original_normal);
        
        
        const COS_EDGE_EPS: f32 = 1e-6;
        if cos_wo.abs() < COS_EDGE_EPS {
            let zero = S::zero();
            let bxdf: Arc<dyn BxDF<S>> = Arc::new(Diffuse::new(zero));
            return BSDF::new(vec![BxDFEntry::new(bxdf, 1.0, zero)], wo, wo)
                .expect("zero BSDF with wo = normal should succeed");
        }


        let normal = hit.shading_normal;
        let normal = if wo.dot(normal) > 0.0 {
            normal
        } else {
            -normal
        };

        let conductor = Conductor::new(f0, alpha);
        let entry = BxDFEntry::new(
            Arc::new(conductor),
            1.0,
            conductor.hemispherical_reflectance(),
        );

        BSDF::new(vec![entry], wo, normal).expect("Conductor BSDF construction should succeed")
    }
}
