//! kzray-impl/src/material/rough_dielectric.rs

// ═══════════════════════════════════════════════════════════════
// 折射辅助
// ═══════════════════════════════════════════════════════════════

use std::sync::Arc;

use glam::Vec3;
use kzray_abs::{BSDF, BSDFBuilder, BxDF, BxDFEntry, BxDFFlags, BxDFSample, Library, Sampler, ScalarAlgebra, SpectralAlgebra, SurfaceHit, TransportMode, bsdf::Diffuse};
use kzray_data::TextureId;
use kzray_math::microfacet::{self, ggx};

use crate::material::fresnel;

/// 局部空间折射。`wo` 指向表面外（`z > 0`），`wm` 是微表面法线
/// （`z > 0`），`ior_i_over_t` = 入射侧 IOR / 透射侧 IOR。
///
/// 返回 `Some(wi)`，`wi.z < 0`；TIR 或退化时 `None`。
#[inline]
pub fn refract_local(wo: Vec3, wm: Vec3, ior_i_over_t: f32) -> Option<Vec3> {
    let cos_wo = wo.dot(wm);
    if cos_wo <= 0.0 {
        return None;
    }
    let eta = ior_i_over_t; // eta_i / eta_t
    let sin2_wo = (1.0 - cos_wo * cos_wo).max(0.0);
    let sin2_wt = eta * eta * sin2_wo;
    if sin2_wt >= 1.0 {
        return None; // TIR
    }
    let cos_wt = (1.0 - sin2_wt).sqrt();
    // TODO: 校验公式是否正确, 对于 walter 2007 和 pbrt-v4
    // pbrt-v3 形式：wi = -eta*wo + (eta*cos_wo - cos_wt) * wm
    let wi = -eta * wo + (eta * cos_wo - cos_wt) * wm;
    let n = wi.length();
    if n < 1e-8 {
        return None;
    }
    Some(wi / n)
}

// ═══════════════════════════════════════════════════════════════
// 粗糙电介质 BxDF（Walter 2007）
// ═══════════════════════════════════════════════════════════════

/// GGX 微表面的粗糙电介质（磨砂玻璃、粗糙水）。
///
/// # 物理
///
/// 每个微表面按 Fresnel 反射 / 折射。反射和透射都是**非 delta**——
/// 微表面法线在 VNDF 上分布，采样出连续方向。
///
/// 组合 PDF 的关键：反射 `wi` 和折射 `wi` 在**不同半球**，所以对任意
/// `wi` 只有一个分支有非零概率——`pdf(wi)` 按 `wi.z` 符号选分支。
///
/// # 采样
///
/// 1. VNDF 采样微表面法线 `wm`；
/// 2. 按 `F` 概率选反射 / 透射分支（`u < F` 反射，否则透射）；
/// 3. 反射：`wi = reflect(wo, wm)`；
/// 4. 透射：`wi = refract(wo, wm, eta_i/eta_t)`，TIR 时强制反射。
///
/// # PDF
///
/// 边缘 PDF（含分支概率）：
///
/// - 反射：`F · pdf_vndf(wm) / (4 |wo·wm|)`；
/// - 透射：`(1-F) · pdf_vndf(wm) · |wo·wm| · (eta_i/eta_t)² / |wi·wm|²`。
///
/// `BSDF::pdf` 与 `sample_f` 返回的 `pdf` 用同一公式，MIS 一致。
///
/// # 能力标签
///
/// `REFLECTION | TRANSMISSION | GLOSSY`——非 delta，参与 NEE / MIS。
#[derive(Clone, Copy, Debug)]
pub struct RoughDielectric<S: ScalarAlgebra> {
    /// 入射侧 IOR。
    pub ior_i: f32,
    /// 透射侧 IOR。
    pub ior_t: f32,
    /// GGX 形状参数，`roughness²`。
    pub alpha: f32,
    /// 透射染色谱。
    pub tint: S,
}

impl<S: ScalarAlgebra> RoughDielectric<S> {
    pub fn new(ior_i: f32, ior_t: f32, alpha: f32, tint: S) -> Self {
        RoughDielectric {
            ior_i,
            ior_t,
            alpha: alpha.clamp(1e-3, 1.0),
            tint,
        }
    }

    /// 半球-半球反射率近似。粗糙电介质无吸收（tint 除外）——
    /// 返回 `tint` 作为保守近似。
    pub fn hemispherical_reflectance(&self) -> S {
        self.tint
    }
}

impl<S: ScalarAlgebra> BxDF<S> for RoughDielectric<S> {
    fn flags(&self) -> BxDFFlags {
        BxDFFlags::REFLECTION | BxDFFlags::TRANSMISSION | BxDFFlags::GLOSSY
    }

    fn eval(&self, wo: Vec3, wi: Vec3, _mode: TransportMode) -> S {
        if wo.z <= 0.0 || wi.z.abs() < 1e-8 {
            return S::zero();
        }
        let reflect = wi.z > 0.0;

        let wm = if reflect {
            microfacet::half_vector(wo, wi)
        } else {
            let v = wo * self.ior_i + wi * self.ior_t;
            let v = v.normalize_or_zero();
            if v.z < 0.0 { -v } else { v }
        };
        if wm.z <= 0.0 {
            return S::zero();
        }
        let cos_wo_m = wo.dot(wm);
        if cos_wo_m <= 0.0 {
            return S::zero();
        }

        let d = ggx::d(wm.z, self.alpha);
        let f = fresnel::fresnel_dielectric::<S>(cos_wo_m, self.ior_i, self.ior_t);

        if reflect {
            let g = ggx::g2(wo.z, wi.z, self.alpha);
            let denom = 4.0 * wo.z * wi.z;
            if denom < 1e-8 {
                return S::zero();
            }
            f.scale(S::Scalar::from(d * g / denom))
        } else {
            // ── 透射 BTDF（Walter 2007 公式 21）──
            //
            // f_t = (1-F) · D · G · |wo·wm| · |wi·wm| /
            //       (|cos_o| · |cos_i| · |m·(wo + η·wi)|²) · η²
            //
            // 由于 m·(wo+η·wi) = η · denom（denom = wi·wm + wo·wm/η），
            // 分母的 |m·(wo+η·wi)|² = η² · denom²，与分子 η² 抵消：
            //
            // f_t = (1-F) · D · G · |wo·wm| · |wi·wm| /
            //       (|cos_o| · |cos_i| · denom²)
            let cos_i = -wi.z;
            let cos_wi_m = -wi.dot(wm); // > 0（wi·wm < 0）
            if cos_wi_m <= 1e-8 {
                return S::zero();
            }
            let g = ggx::g2(wo.z, cos_i, self.alpha);
            let eta = self.ior_i / self.ior_t;
            let denom = wi.dot(wm) + wo.dot(wm) / eta;
            let denom_sq = denom * denom;
            if denom_sq < 1e-10 {
                return S::zero();
            }
            let one_minus_f = S::one().sub(&f);
            one_minus_f.mul(&self.tint).scale(S::Scalar::from(
                d * g * cos_wo_m * cos_wi_m / (cos_i * wo.z * denom_sq),
            ))
        }
    }

    fn sample_f(
        &self,
        wo: Vec3,
        sampler: &mut dyn Sampler,
        dimension: u32,
        _mode: TransportMode,
    ) -> Option<BxDFSample<S>> {
        if wo.z <= 0.0 {
            return None;
        }
        let (u1, u2) = sampler.next_2d(dimension);
        let wm = ggx::sample_vndf(wo, self.alpha, u1, u2);
        if wm.z <= 0.0 {
            return None;
        }
        let cos_wo_m = wo.dot(wm);
        if cos_wo_m <= 0.0 {
            return None;
        }

        let f = fresnel::fresnel_dielectric::<S>(cos_wo_m, self.ior_i, self.ior_t);
        let f_scalar =
            kzray_math::fresnel::fresnel_dielectric(cos_wo_m, self.ior_i, self.ior_t);

        let pdf_m = ggx::pdf_vndf(wo, wm, self.alpha);
        if pdf_m <= 1e-8 {
            return None;
        }
        let d = ggx::d(wm.z, self.alpha);

        let u3 = sampler.next_f32(dimension + 2);

        if u3 < f_scalar {
            // ── 反射分支 ──
            let wi = microfacet::reflect(wo, wm);
            if wi.z <= 0.0 {
                return None;
            }
            let g = ggx::g2(wo.z, wi.z, self.alpha);
            let denom = 4.0 * wo.z * wi.z;
            if denom < 1e-8 {
                return None;
            }
            let value = f.scale(S::Scalar::from(d * g / denom));

            let pdf_edge = f_scalar * pdf_m / (4.0 * cos_wo_m);
            if pdf_edge <= 1e-8 {
                return None;
            }
            let weight = value.scale(S::Scalar::from(wi.z / pdf_edge));

            Some(BxDFSample {
                wi,
                value,
                weight,
                pdf: pdf_edge,
                pdf_is_exact: true,
                flags: self.flags(),
                eta: 1.0,
                consumed: 3,
            })
        } else {
            // ── 透射分支 ──
            let eta_i_over_t = self.ior_i / self.ior_t;
            let wi = refract_local(wo, wm, eta_i_over_t)?;
            if wi.z >= 0.0 {
                return None;
            }
            let cos_i = -wi.z;
            let cos_wi_m = -wi.dot(wm); // > 0
            if cos_wi_m <= 1e-8 {
                return None;
            }

            let g = ggx::g2(wo.z, cos_i, self.alpha);
            let eta = eta_i_over_t;
            let denom = wi.dot(wm) + wo.dot(wm) / eta;
            let denom_sq = denom * denom;
            if denom_sq < 1e-10 {
                return None;
            }
            let one_minus_f = S::one().sub(&f);
            let value = one_minus_f.mul(&self.tint).scale(S::Scalar::from(
                d * g * cos_wo_m * cos_wi_m / (cos_i * wo.z * denom_sq),
            ));

            // 折射 PDF：pdf(wi) = (1-F) · pdf_m · |dm/dwi|
            //                  = (1-F) · pdf_m · |wi·wm| / denom²
            let pdf_edge = (1.0 - f_scalar) * pdf_m * cos_wi_m / denom_sq;
            if pdf_edge <= 1e-8 {
                return None;
            }
            let weight = value.scale(S::Scalar::from(cos_i / pdf_edge));

            Some(BxDFSample {
                wi,
                value,
                weight,
                pdf: pdf_edge,
                pdf_is_exact: true,
                flags: self.flags(),
                eta: eta_i_over_t,
                consumed: 3,
            })
        }
    }

    fn pdf(&self, wo: Vec3, wi: Vec3, _mode: TransportMode) -> f32 {
        if wo.z <= 0.0 || wi.z.abs() < 1e-8 {
            return 0.0;
        }
        let reflect = wi.z > 0.0;
        let wm = if reflect {
            microfacet::half_vector(wo, wi)
        } else {
            let v = wo * self.ior_i + wi * self.ior_t;
            let v = v.normalize_or_zero();
            if v.z < 0.0 { -v } else { v }
        };
        if wm.z <= 0.0 {
            return 0.0;
        }
        let cos_wo_m = wo.dot(wm);
        if cos_wo_m <= 0.0 {
            return 0.0;
        }

        let f_scalar =
            kzray_math::fresnel::fresnel_dielectric(cos_wo_m, self.ior_i, self.ior_t);
        let pdf_m = ggx::pdf_vndf(wo, wm, self.alpha);
        if pdf_m <= 1e-8 {
            return 0.0;
        }

        if reflect {
            f_scalar * pdf_m / (4.0 * cos_wo_m)
        } else {
            let cos_wi_m = -wi.dot(wm);
            if cos_wi_m <= 1e-8 {
                return 0.0;
            }
            let eta = self.ior_i / self.ior_t;
            let denom = wi.dot(wm) + wo.dot(wm) / eta;
            let denom_sq = denom * denom;
            if denom_sq < 1e-10 {
                return 0.0;
            }
            (1.0 - f_scalar) * pdf_m * cos_wi_m / denom_sq
        }
    }

    fn rho(
        &self,
        _wo: Vec3,
        _sampler: &mut dyn Sampler,
        _dimension: u32,
        _mode: TransportMode,
    ) -> (S, u32) {
        (self.tint, 0)
    }

    fn max_sample_dimensions(&self) -> u32 {
        3
    }
}

// ═══════════════════════════════════════════════════════════════
// RoughDielectricBSDFBuilder
// ═══════════════════════════════════════════════════════════════

/// 粗糙玻璃材质构建器。
#[derive(Clone, Copy, Debug)]
pub struct RoughDielectricBSDFBuilder {
    ior: f32,
    roughness: TextureId,
    tint: TextureId,
}

impl RoughDielectricBSDFBuilder {
    pub fn new(ior: f32, roughness: TextureId, tint: TextureId) -> Self {
        RoughDielectricBSDFBuilder {
            ior,
            roughness,
            tint,
        }
    }
}

impl<S: SpectralAlgebra> BSDFBuilder<S> for RoughDielectricBSDFBuilder {
    fn build(
        &self,
        hit: &SurfaceHit,
        wo: Vec3,
        _sampler: &mut dyn Sampler,
        _wavelengths: &S::WavelengthContext,
        lib: &Library<S>,
    ) -> BSDF<S> {
        let tint = lib.albedo(self.tint).eval(hit.uv, None);
        let roughness = lib.roughness(self.roughness).eval(hit.uv, None);
        let alpha = roughness * roughness;

        let original_normal = hit.shading_normal;
        let cos_wo = wo.dot(original_normal);

        // 擦边 guard（同 Dielectric）
        const COS_EDGE_EPS: f32 = 1e-6;
        if cos_wo.abs() < COS_EDGE_EPS {
            let zero = S::zero();
            let bxdf: Arc<dyn BxDF<S>> = Arc::new(Diffuse::new(zero));
            return BSDF::new(vec![BxDFEntry::new(bxdf, 1.0, zero)], wo, wo)
                .expect("zero BSDF with wo = normal should succeed");
        }

        let entering = cos_wo > 0.0;
        let (ior_i, ior_t) = if entering {
            (1.0, self.ior)
        } else {
            (self.ior, 1.0)
        };
        let normal = if entering { original_normal } else { -original_normal };

        let bxdf = RoughDielectric::new(ior_i, ior_t, alpha, tint);
        let entry = BxDFEntry::new(
            Arc::new(bxdf),
            1.0,
            bxdf.hemispherical_reflectance(),
        );

        BSDF::new(vec![entry], wo, normal)
            .expect("RoughDielectric BSDF construction should succeed")
    }
}
