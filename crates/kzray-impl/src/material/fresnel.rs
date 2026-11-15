use kzray_abs::ScalarAlgebra;

/// Schlick 近似的 Fresnel 反射率（导体）。
///
/// `F(cos θ) = F0 + (1 - F0) · (1 - cos θ)^5`
///
/// 对 `cos θ ∈ [-1, 1]` 输入，内部 clamp 到 `[0, 1]`——`cos θ` 为负
/// （背面）时按 0 处理，Fresnel 趋于 1。
#[inline]
pub fn fresnel_schlick<S: ScalarAlgebra>(f0: &S, cos_theta: f32) -> S {
    let m = (1.0 - cos_theta.clamp(0.0, 1.0)).max(0.0);
    let m5 = m * m * m * m * m;
    let one_minus_f0 = S::one().sub(f0);
    f0.add(&one_minus_f0.scale(S::Scalar::from(m5)))
}
