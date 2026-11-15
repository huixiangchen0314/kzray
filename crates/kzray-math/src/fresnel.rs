//! kzray-math/src/fresnel.rs
//! Fresnel 反射率。介电（玻璃、水）和导体（金属）的两种精确公式，
//! 外加 Schlick 近似。
//!
//! 所有函数返回**反射率** `F ∈ [0, 1]`，透射率 `= 1 - F`（无吸收时）。
//! `cos_theta_i` 是入射角余弦（对法线的绝对值），负值会被内部处理
//! （交换 ior 并取绝对值），所以调用方可以传 `wo.dot(n)`（可能为负）。

// ═══════════════════════════════════════════════════════════════
// 介电 Fresnel（精确，含全内反射）
// ═══════════════════════════════════════════════════════════════

/// 介电 Fresnel 反射率。
///
/// `ior_i` / `ior_t` 是入射/透射侧的折射率。`cos_theta_i` 可以是负值，
/// 内部会交换 `ior_i` / `ior_t` 并取绝对值。
///
/// 全内反射（TIR）时返回 1.0。垂直入射时返回 `((ηi - ηt) / (ηi + ηt))²`。
pub fn fresnel_dielectric(mut cos_theta_i: f32, mut ior_i: f32, mut ior_t: f32) -> f32 {
    cos_theta_i = cos_theta_i.clamp(-1.0, 1.0);
    if cos_theta_i < 0.0 {
        std::mem::swap(&mut ior_i, &mut ior_t);
        cos_theta_i = -cos_theta_i;
    }

    let sin2_theta_t = (ior_i / ior_t).powi(2) * (1.0 - cos_theta_i * cos_theta_i);
    if sin2_theta_t >= 1.0 {
        return 1.0;   // 全内反射
    }

    let cos_theta_t = (1.0 - sin2_theta_t).max(0.0).sqrt();
    let r_parallel = (ior_t * cos_theta_i - ior_i * cos_theta_t)
        / (ior_t * cos_theta_i + ior_i * cos_theta_t);
    let r_perp = (ior_i * cos_theta_i - ior_t * cos_theta_t)
        / (ior_i * cos_theta_i + ior_t * cos_theta_t);

    0.5 * (r_parallel * r_parallel + r_perp * r_perp)
}

// ═══════════════════════════════════════════════════════════════
// Schlick 近似
// ═══════════════════════════════════════════════════════════════

/// Schlick 近似：`F = F0 + (1 - F0) * (1 - cosθ)^5`。
///
/// 在介电和导体上都可近似使用。`f0` 是垂直入射时的反射率。
pub fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let c = (1.0 - cos_theta.abs().clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let c5 = c * c * c * c * c;
    f0 + (1.0 - f0) * c5
}

/// 介电 Schlick 近似。`f0 = ((ηi - ηt) / (ηi + ηt))²`。
///
/// 比精确版快，但丢失 TIR 行为——掠射时趋近 1，但不会在临界角突然跳变。
/// 仅在不需要 TIR 的场合（金属反射近似、微表面单次散射）使用。
pub fn fresnel_schlick_dielectric(cos_theta_i: f32, ior_i: f32, ior_t: f32) -> f32 {
    let eta = (ior_t - ior_i) / (ior_t + ior_i);
    fresnel_schlick(cos_theta_i, eta * eta)
}

// ═══════════════════════════════════════════════════════════════
// 导体 Fresnel（精确）
// ═══════════════════════════════════════════════════════════════

/// 导体 Fresnel 反射率。复折射率 `η + ik`。
///
/// `eta` / `k` 是相对折射率（相对周围介质）。金属的典型值：
/// - 金：`eta ≈ 0.15`, `k ≈ 3.5`（红光）
/// - 银：`eta ≈ 0.05`, `k ≈ 4.0`
/// - 铝：`eta ≈ 1.3`, `k ≈ 7.0`
///
/// 垂直入射：`F0 = ((η-1)² + k²) / ((η+1)² + k²)`。掠射时趋于 1。
pub fn fresnel_conductor(cos_theta_i: f32, eta: f32, k: f32) -> f32 {
    // 内部用 f64 中间计算：单调性/垂直入射的误差从 ~1e-5 降到 ~1e-7
    let c = cos_theta_i.abs().clamp(0.0, 1.0) as f64;
    let c2 = c * c;
    let s2 = 1.0 - c2;
    let eta = eta as f64;
    let k = k as f64;
    let eta2 = eta * eta;
    let k2 = k * k;

    let t = eta2 - k2 - s2;
    let a2plusb2 = (t * t + 4.0 * eta2 * k2).sqrt();
    let a = ((a2plusb2 + t) * 0.5).max(0.0).sqrt();
    let b = ((a2plusb2 - t) * 0.5).max(0.0).sqrt();

    let rs_num = (a - c) * (a - c) + b * b;
    let rs_den = (a + c) * (a + c) + b * b;
    let rs = rs_num / rs_den;

    let ac = a * c;
    let bc = b * c;
    let rp_num = (ac - 1.0) * (ac - 1.0) + bc * bc;
    let rp_den = (ac + 1.0) * (ac + 1.0) + bc * bc;
    let rp = rp_num / rp_den;

    (0.5 * (rs + rp)) as f32
}

// ═══════════════════════════════════════════════════════════════
// 便捷常量
// ═══════════════════════════════════════════════════════════════

/// 常用 IOR。
pub mod ior {
    /// 空气（近似 1.0）。
    pub const AIR: f32 = 1.0;
    /// 水。
    pub const WATER: f32 = 1.333;
    /// 玻璃（普通冕牌）。
    pub const GLASS: f32 = 1.5;
    /// 钻石。
    pub const DIAMOND: f32 = 2.417;
    /// 冰。
    pub const ICE: f32 = 1.31;
}
