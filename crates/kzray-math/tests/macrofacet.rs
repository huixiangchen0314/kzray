//! `kzray-math::microfacet` 的集成测试。

use glam::Vec3;
use kzray_math::microfacet::{beckmann, ggx, half_vector, reflect};
use std::f32::consts::{FRAC_PI_2, PI, TAU};

const N: usize = 100_000;

fn u_seq(i: usize, k: f32) -> f32 {
    ((i as f32 + 0.5) * k).fract()
}

fn local_dir(cos_theta: f32, phi: f32) -> Vec3 {
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

/// 2D 网格确定性地积分半球上的函数 `f(m) dω_m`。
/// 比均匀 MC 稳定，适合验证 NDF 归一化这类极尖的被积函数。
/// 2D 网格确定性积分半球。内部用 f32 采样、f64 累加。
fn integrate_hemisphere<F: Fn(Vec3) -> f32>(f: F) -> f64 {
    let n_theta = 512;
    let n_phi = 512;
    let mut sum = 0.0f64;
    let d_theta = std::f64::consts::FRAC_PI_2 / n_theta as f64;
    let d_phi = std::f64::consts::TAU / n_phi as f64;
    for i in 0..n_theta {
        let theta = (i as f32 + 0.5) / n_theta as f32 * FRAC_PI_2;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        for j in 0..n_phi {
            let phi = (j as f32 + 0.5) / n_phi as f32 * TAU;
            let m = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
            sum += (f(m) * sin_theta) as f64 * d_theta * d_phi;
        }
    }
    sum
}

// ── 公共辅助 ──

#[test]
fn reflect_about_normal() {
    let wo = Vec3::new(0.3, 0.4, 0.866).normalize();
    let n = Vec3::Z;
    let wi = reflect(wo, n);
    // 关于 n 反射：xy 翻转，z 保留
    let expected = Vec3::new(-0.3, -0.4, 0.866).normalize();
    assert!((wi - expected).length() < 1e-5, "wi={wi:?} expected={expected:?}");
    // 半向量关于 n 对称
    let h = half_vector(wo, wi);
    assert!((h - n).length() < 1e-5);
}

#[test]
fn half_vector_of_same_direction() {
    let v = Vec3::new(0.3, 0.4, 0.866).normalize();
    let h = half_vector(v, v);
    assert!((h - v).length() < 1e-5);
}

// ── GGX NDF ──

#[test]
fn ggx_d_integrates_to_one() {
    for alpha in [0.1f32, 0.3, 0.5, 0.8, 1.0] {
        let estimate = integrate_hemisphere(|m| ggx::d(m.z, alpha) * m.z);
        assert!((estimate - 1.0).abs() < 0.01, "alpha={alpha} ∫D·cos = {estimate}");
    }
}

#[test]
fn ggx_d_zero_below_horizon() {
    assert_eq!(ggx::d(0.0, 0.5), 0.0);
    assert_eq!(ggx::d(-0.5, 0.5), 0.0);
}

#[test]
fn ggx_d_at_normal_is_large() {
    for alpha in [0.1f32, 0.3, 0.5] {
        let d0 = ggx::d(1.0, alpha);
        let expected = 1.0 / (PI * alpha * alpha);
        assert!((d0 - expected).abs() / expected < 0.01, "alpha={alpha} d0={d0} expected={expected}");
    }
}

// ── GGX G1 / G2 ──

#[test]
fn ggx_g1_bounds() {
    for alpha in [0.1f32, 0.5, 1.0] {
        for i in 0..=100 {
            let cos_v = i as f32 / 100.0;
            let g = ggx::g1(cos_v, alpha);
            assert!((0.0..=1.0).contains(&g), "alpha={alpha} cos={cos_v} g={g}");
        }
    }
}

#[test]
fn ggx_g1_grazing_near_zero() {
    // 掠射时遮蔽严重，G1 → 0（物理正确）
    for alpha in [0.1f32, 0.5, 1.0] {
        let g = ggx::g1(0.01, alpha);
        assert!(g < 0.2, "alpha={alpha} g={g}，掠射应接近 0");
    }
}

#[test]
fn ggx_g1_vertical_near_one() {
    for alpha in [0.1f32, 0.5, 1.0] {
        let g = ggx::g1(1.0, alpha);
        assert!((g - 1.0).abs() < 1e-4, "alpha={alpha} g={g}");
    }
}

#[test]
fn ggx_g2_le_g1() {
    for alpha in [0.1f32, 0.5, 1.0] {
        for i in 0..=50 {
            for j in 0..=50 {
                let co = (i as f32 / 50.0).max(0.01);
                let ci = (j as f32 / 50.0).max(0.01);
                let g2 = ggx::g2(co, ci, alpha);
                let g1o = ggx::g1(co, alpha);
                let g1i = ggx::g1(ci, alpha);
                assert!(g2 <= g1o + 1e-6, "alpha={alpha} co={co} ci={ci} g2={g2} g1o={g1o}");
                assert!(g2 <= g1i + 1e-6);
            }
        }
    }
}

// ── GGX VNDF 采样 ──

#[test]
fn ggx_sample_vndf_returns_upper_hemisphere() {
    for alpha in [0.1f32, 0.5, 1.0] {
        let wo = local_dir(0.8, 0.5).normalize();
        for i in 0..1000 {
            let u1 = u_seq(i, 0.618_033_988_7);
            let u2 = u_seq(i, 0.754_877_666_2);
            let m = ggx::sample_vndf(wo, alpha, u1, u2);
            assert!(m.z > 0.0, "alpha={alpha} i={i} m.z={}", m.z);
            assert!((m.length() - 1.0).abs() < 1e-4, "i={i} |m|={}", m.length());
            // VNDF 保证 m 可视（wo·m > 0）
            assert!(wo.dot(m) > 0.0, "alpha={alpha} i={i} wo·m={}", wo.dot(m));
        }
    }
}

#[test]
fn ggx_sample_vndf_pdf_integrates_to_one() {
    // ∫ pdf_vndf(wo, m) dω_m = 1（VNDF 归一化）
    let alpha = 0.3;
    let wo = local_dir(0.8, 0.5).normalize();
    let estimate = integrate_hemisphere(|m| ggx::pdf_vndf(wo, m, alpha));
    assert!((estimate - 1.0).abs() < 0.02, "∫pdf_vndf = {estimate}");
}

#[test]
fn ggx_sample_vndf_mean_aligned_with_wo() {
    let alpha = 0.3;
    let wo = local_dir(0.9, 0.3).normalize();
    let mut sum = 0.0f64;
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let m = ggx::sample_vndf(wo, alpha, u1, u2);
        sum += m.dot(wo) as f64;
    }
    let mean = sum / N as f64;
    assert!(mean > 0.7, "E[m·wo] = {mean}，VNDF 应偏向 wo");
}



// ── Beckmann NDF ──

#[test]
fn beckmann_d_integrates_to_one() {
    for alpha in [0.1f32, 0.3, 0.5, 0.8] {
        let estimate = integrate_hemisphere(|m| beckmann::d(m.z, alpha) * m.z);
        assert!((estimate - 1.0).abs() < 0.02, "alpha={alpha} ∫D·cos = {estimate}");
    }
}

#[test]
fn beckmann_d_zero_below_horizon() {
    assert_eq!(beckmann::d(0.0, 0.5), 0.0);
    assert_eq!(beckmann::d(-0.5, 0.5), 0.0);
}

#[test]
fn beckmann_d_at_normal() {
    for alpha in [0.1f32, 0.5] {
        let d0 = beckmann::d(1.0, alpha);
        let expected = 1.0 / (PI * alpha * alpha);
        assert!((d0 - expected).abs() / expected < 0.01, "alpha={alpha} d0={d0}");
    }
}

#[test]
fn beckmann_g1_bounds() {
    for alpha in [0.1f32, 0.5, 1.0] {
        for i in 0..=100 {
            let cos_v = i as f32 / 100.0;
            let g = beckmann::g1(cos_v, alpha);
            assert!((0.0..=1.0).contains(&g), "alpha={alpha} cos={cos_v} g={g}");
        }
    }
}

#[test]
fn beckmann_g1_grazing_near_zero() {
    for alpha in [0.1f32, 0.5, 1.0] {
        let g = beckmann::g1(0.01, alpha);
        assert!(g < 0.5, "alpha={alpha} g={g}，掠射应接近 0");
    }
}

#[test]
fn beckmann_g1_vertical_near_one() {
    for alpha in [0.1f32, 0.5, 1.0] {
        let g = beckmann::g1(1.0, alpha);
        assert!((g - 1.0).abs() < 1e-4, "alpha={alpha} g={g}");
    }
}

#[test]
fn beckmann_g2_le_g1() {
    for alpha in [0.1f32, 0.5, 1.0] {
        for i in 0..=20 {
            for j in 0..=20 {
                let co = (i as f32 / 20.0).max(0.05);
                let ci = (j as f32 / 20.0).max(0.05);
                let g2 = beckmann::g2(co, ci, alpha);
                let g1o = beckmann::g1(co, alpha);
                let g1i = beckmann::g1(ci, alpha);
                assert!(g2 <= g1o + 1e-4, "alpha={alpha} co={co} ci={ci} g2={g2} g1o={g1o}");
                assert!(g2 <= g1i + 1e-4);
            }
        }
    }
}

#[test]
fn beckmann_sample_vndf_upper_hemisphere() {
    for alpha in [0.1f32, 0.5, 1.0] {
        let wo = local_dir(0.8, 0.5).normalize();
        for i in 0..1000 {
            let u1 = u_seq(i, 0.618_033_988_7);
            let u2 = u_seq(i, 0.754_877_666_2);
            let m = beckmann::sample_vndf(wo, alpha, u1, u2);
            assert!(m.z > 0.0, "alpha={alpha} i={i} m.z={}", m.z);
            assert!((m.length() - 1.0).abs() < 1e-4);
        }
    }
}



#[test]
fn ggx_and_beckmann_agree_on_low_roughness() {
    let alpha = 0.1;
    let d_ggx = ggx::d(0.99, alpha);
    let d_beck = beckmann::d(0.99, alpha);
    let ratio = d_ggx / d_beck;
    assert!(ratio > 0.1 && ratio < 10.0, "ggx={d_ggx} beckmann={d_beck} ratio={ratio}");
}

#[test]
fn ggx_sample_vndf_typical_roughness_yields_majority_valid_wi() {
    // VNDF 采样只保证 m 可见（wo·m > 0），不保证 wi 向上。
    // 高粗糙度（alpha 大）时 wi 向下是物理必然，由积分器过滤。
    //
    // 这个测试只验证**典型材质 roughness（alpha ≤ 0.5）下多数样本有效**，
    // 这是渲染器"不被无效样本淹没"的经验判据。
    // 具体阈值参考 PBRT-v4 的采样实验。
    for (alpha, threshold) in [(0.1f32, 0.95), (0.3, 0.85), (0.5, 0.70)] {
        let wo = local_dir(0.9, 0.5).normalize();
        let mut above = 0;
        for i in 0..1000 {
            let u1 = u_seq(i, 0.618_033_988_7);
            let u2 = u_seq(i, 0.754_877_666_2);
            let m = ggx::sample_vndf(wo, alpha, u1, u2);
            let wi = reflect(wo, m);
            if wi.z > 0.0 { above += 1; }
        }
        let frac = above as f32 / 1000.0;
        assert!(
            frac >= threshold,
            "alpha={alpha} 上半球比例 {frac}，典型粗糙度应 ≥ {threshold}"
        );
    }
}

#[test]
fn beckmann_sample_vndf_typical_roughness_yields_majority_valid_wi() {
    // Beckmann 用 NDF 采样（不是严格的 VNDF），wi 向下的比例比 GGX VNDF 高。
    // 只测典型 roughness，阈值比 GGX 更宽松。
    for (alpha, threshold) in [(0.1f32, 0.85), (0.3, 0.75), (0.5, 0.65)] {
        let wo = local_dir(0.9, 0.5).normalize();
        let mut above = 0;
        for i in 0..1000 {
            let u1 = u_seq(i, 0.618_033_988_7);
            let u2 = u_seq(i, 0.754_877_666_2);
            let m = beckmann::sample_vndf(wo, alpha, u1, u2);
            let wi = reflect(wo, m);
            if wi.z > 0.0 { above += 1; }
        }
        let frac = above as f32 / 1000.0;
        assert!(
            frac >= threshold,
            "alpha={alpha} 上半球比例 {frac}，典型粗糙度应 ≥ {threshold}"
        );
    }
}
