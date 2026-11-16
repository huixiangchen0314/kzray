// kzray-impl/src/material/dielectric.rs
//!
//! 完美光滑玻璃（电介质）材质。

use std::sync::Arc;

use glam::Vec3;

use kzray_abs::{
    BSDF, BSDFBuilder, BxDF, BxDFEntry, BxDFFlags, BxDFSample, Library, Sampler, ScalarAlgebra, SpectralAlgebra, SurfaceHit, TransportMode, bsdf::Diffuse,
};
use kzray_data::TextureId;

use crate::material::fresnel;

// ═══════════════════════════════════════════════════════════════
// Dielectric BxDF
// ═══════════════════════════════════════════════════════════════

/// 完美光滑玻璃（电介质）的 BSDF。
///
/// # 物理
///
/// 每个表面点处，光按 Fresnel 反射率 `F` 反射，`1 - F` 折射。两个分支
/// 都是 delta 分布——光线只沿单一方向传播，没有连续散射。
///
/// 透射方向按 Snell 定律：`sin θ_t = (ior_i / ior_t) · sin θ_i`。全内反射
/// （TIR）在 `sin θ_t ≥ 1` 时发生（仅当光线从高 IOR 介质射向低 IOR 时）。
///
/// # 进入 / 离开的区分
///
/// `ior_i` / `ior_t` 由 builder 根据 `wo · shading_normal` 符号决定：
///
/// - **进入**（`wo` 在法线同侧，`wo·n > 0`）：光从外部进入玻璃。
///   `ior_i = external`（通常 1.0），`ior_t = glass`（例如 1.5）；
/// - **离开**（`wo` 在法线异侧，`wo·n < 0`）：光从玻璃内部离开。
///   `ior_i = glass`，`ior_t = external`。
///
/// `BSDF::new` 会把法线翻转到 `wo` 同侧，所以 BxDF 内部 `wo.z > 0` 始终
/// 成立。`ior_i` / `ior_t` 的先后由 builder 保证——BxDF 内部不再判断
/// 进入/离开。
///
/// # 能力标签
///
/// `REFLECTION | TRANSMISSION | SPECULAR`——纯 delta，`is_pure_delta()`。
/// 两种散射方向都由几何唯一确定，NEE 采不到，`specular_bounce = true`。
///
/// # `weight` 与 `value`
///
/// 反射和折射分支共用一套 `value / pdf` 记账。由于是 delta，`pdf` 无
/// 连续测度——`sample_f` 把**选择该分支的概率**（`F` 或 `1-F`）折进
/// `value`，`BSDF::sample_f` 的 delta 分支再乘 `|cos|`，最终 `weight`
/// 恰为该分支的 Fresnel 权重。例如 `tint = 1`：
///
/// - 反射：`weight = F`；
/// - 透射：`weight = 1 - F`。
///
/// `tint` 是着色谱（玻璃的透射染色，例如彩色玻璃），乘在 `weight` 上。
#[derive(Clone, Copy, Debug)]
pub struct Dielectric<S: ScalarAlgebra> {
    /// 入射侧 IOR（进入 = 外部介质，离开 = 玻璃）。
    pub ior_i: f32,
    /// 透射侧 IOR（进入 = 玻璃，离开 = 外部介质）。
    pub ior_t: f32,
    /// 透射染色谱。`S::one()` 表示无色透明玻璃。
    pub tint: S,
}

impl<S: ScalarAlgebra> Dielectric<S> {
    pub fn new(ior_i: f32, ior_t: f32, tint: S) -> Self {
        Dielectric { ior_i, ior_t, tint }
    }

    /// 半球-半球反射率近似。电介质无吸收，透过 + 反射 = 1——返回
    /// `S::one() - albedo_of_transmission`。用 `tint` 作为保守近似。
    pub fn hemispherical_reflectance(&self) -> S {
        self.tint
    }
}

impl<S: ScalarAlgebra> BxDF<S> for Dielectric<S> {
    fn flags(&self) -> BxDFFlags {
        BxDFFlags::REFLECTION | BxDFFlags::TRANSMISSION | BxDFFlags::SPECULAR
    }

    fn eval(&self, _wo: Vec3, _wi: Vec3, _mode: TransportMode) -> S {
        S::zero()
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
        let cos_theta_o = wo.z;
        let sin2_theta_o = (1.0 - cos_theta_o * cos_theta_o).max(0.0);

        // Snell（pbrt 约定：eta = eta_t / eta_i）
        let eta_t_over_i = self.ior_t / self.ior_i;
        let sin2_theta_t = sin2_theta_o / (eta_t_over_i * eta_t_over_i);
        let tir = sin2_theta_t >= 1.0;

        // Fresnel 反射率（标量，用于概率）与（谱，用于权重）
        let f_scalar = if tir {
            1.0
        } else {
            kzray_math::fresnel::fresnel_dielectric(cos_theta_o, self.ior_i, self.ior_t)
        };
        let f = fresnel::fresnel_dielectric::<S>(cos_theta_o, self.ior_i, self.ior_t);
        let one_minus_f = S::one().sub(&f);

        // 以概率 f_scalar 选反射
        let u = sampler.next_f32(dimension);

        if tir || u < f_scalar {
            // ── 反射分支 ──
            let wi = Vec3::new(-wo.x, -wo.y, wo.z);
            // value = tint * F / cos_o，weight = value * cos = tint * F
            let value = f.mul(&self.tint).scale(S::Scalar::from(1.0 / cos_theta_o));
            Some(BxDFSample {
                wi,
                value,
                weight: value, // BSDF::sample_f delta 分支会覆盖
                pdf: 0.0,
                pdf_is_exact: true,
                flags: self.flags(),
                eta: 1.0, // 反射无介质变化
                consumed: 1,
            })
        } else {
            // ── 透射分支 ──
            let cos_theta_t = (1.0 - sin2_theta_t).max(0.0).sqrt();
            // pbrt 方向公式：wi = -wo/eta + (cos_o/eta - cos_t) * n
            let wi = Vec3::new(-wo.x / eta_t_over_i, -wo.y / eta_t_over_i, -cos_theta_t);
            // |cos_i| = cos_theta_t（wi.z = -cos_theta_t）
            let abs_cos_i = cos_theta_t;
            if abs_cos_i <= f32::EPSILON {
                return None;
            }
            // value = tint * (1 - F) / |cos_t|
            // weight = value * |cos_t| = tint * (1 - F)
            let value = one_minus_f
                .mul(&self.tint)
                .scale(S::Scalar::from(1.0 / abs_cos_i));
            Some(BxDFSample {
                wi,
                value,
                weight: value,
                pdf: 0.0,
                pdf_is_exact: true,
                flags: self.flags(),
                eta: eta_t_over_i,
                consumed: 1,
            })
        }
    }

    fn pdf(&self, _wo: Vec3, _wi: Vec3, _mode: TransportMode) -> f32 {
        0.0
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
        1
    }
}

// ═══════════════════════════════════════════════════════════════
// DielectricBSDFBuilder
// ═══════════════════════════════════════════════════════════════

/// 完美光滑玻璃材质构建器。
///
/// 从 `tint` 纹理取透射染色谱，`ior` 为常量（玻璃 IOR 在可见光范围内
/// 几乎无色散，常量足够；色散留后续）。
///
/// # 进入/离开自动判断
///
/// builder 检查 `wo · shading_normal` 的符号来决定 `ior_i` / `ior_t`：
///
/// - `wo · n > 0`：光从外部进入玻璃，`ior_i = 1.0, ior_t = ior`；
/// - `wo · n < 0`：光从玻璃内部离开，`ior_i = ior, ior_t = 1.0`。
///
/// 外部介质固定为空气（IOR = 1.0）。需要玻璃在水中等嵌套介质的场景
/// 留待后续实现介质栈管理。
#[derive(Clone, Copy, Debug)]
pub struct DielectricBSDFBuilder {
    /// 玻璃的 IOR（例如 1.5 为典型光学玻璃，1.33 为水，2.4 为钻石）。
    ior: f32,
    /// 透射染色纹理（`AlbedoChannel<S>`，`S::one()` 为无色）。
    tint: TextureId,
}

impl DielectricBSDFBuilder {
    pub fn new(ior: f32, tint: TextureId) -> Self {
        DielectricBSDFBuilder { ior, tint }
    }

    pub fn ior(&self) -> f32 {
        self.ior
    }

    pub fn tint(&self) -> TextureId {
        self.tint
    }
}

impl<S: SpectralAlgebra> BSDFBuilder<S> for DielectricBSDFBuilder {
    fn build(
        &self,
        hit: &SurfaceHit,
        wo: Vec3,
        _sampler: &mut dyn Sampler,
        _wavelengths: &S::WavelengthContext,
        lib: &Library<S>,
    ) -> BSDF<S> {
        let tint = lib.albedo(self.tint).eval(hit.uv, None);

        let original_normal = hit.shading_normal;
        let cos_wo = wo.dot(original_normal);

        // ── 擦边处理 ──
        //
        // 玻璃球内部 TIR 后光线从内部极掠射表面时，|wo·n| 趋近 0——
        // 翻转法线无法帮助（0 的相反数仍是 0），`BSDF::new` 的
        // `wo_local.z > 0` 检查必然失败。
        //
        // 擦边贡献在连续测度下为零（积分测度 → 0），返回零反射 BSDF
        // 不引入偏差。用 `wo` 自身作为 fallback 法线——`BSDF::new`
        // 只要求 `wo·normal > 0`，`wo·wo = 1` 恒成立。零反射 BSDF 的
        // `eval` / `sample_f` 始终返回零，法线方向不影响结果。
        const COS_EDGE_EPS: f32 = 1e-6;
        if cos_wo.abs() < COS_EDGE_EPS {
            let zero = S::zero();
            let bxdf: Arc<dyn BxDF<S>> = Arc::new(Diffuse::new(zero));
            return BSDF::new(vec![BxDFEntry::new(bxdf, 1.0, zero)], wo, wo)
                .expect("zero BSDF with wo = normal should succeed");
        }

        // ── 正常路径 ──
        let entering = cos_wo > 0.0;
        let (ior_i, ior_t) = if entering {
            (1.0, self.ior)
        } else {
            (self.ior, 1.0)
        };
        let normal = if entering {
            original_normal
        } else {
            -original_normal
        };

        let dielectric = Dielectric::new(ior_i, ior_t, tint);
        let entry = BxDFEntry::new(
            Arc::new(dielectric),
            1.0,
            dielectric.hemispherical_reflectance(),
        );

        BSDF::new(vec![entry], wo, normal).expect("Dielectric BSDF construction should succeed")
    }
}



