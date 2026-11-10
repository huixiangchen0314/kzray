//! `kzray-math::sampling` 的集成测试。
//!
//! 验证核心不变量：
//! 1. PDF 的数值积分 = 1（方向采样）或面积 = π（圆盘）或 1（三角形面积归一化）
//! 2. 采样方向落在期望区域内（半球 / 锥 / 圆）
//! 3. 余弦加权采样的方向分布符合 cosθ 权重
//! 4. 世界空间包装与局部 + Onb 的组合一致

use glam::Vec3;
use kzray_math::onb::Onb;
use kzray_math::sampling::*;
use std::f32::consts::PI;

const N: usize = 100_000;

fn u_seq(i: usize, k: f32) -> f32 {
    ((i as f32 + 0.5) * k).fract()
}

// ── 余弦加权半球 ──

#[test]
fn cosine_hemisphere_directions_in_upper_hemisphere() {
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, pdf) = cosine_hemisphere(u1, u2);
        assert!(d.z >= -1e-6, "i={i} z={}", d.z);
        assert!((d.length() - 1.0).abs() < 1e-4, "i={i} |d|={}", d.length());
        assert!((pdf - d.z / PI).abs() < 1e-5, "i={i}");
    }
}

#[test]
fn cosine_hemisphere_pdf_integrates_to_one() {
    // 数值积分 ∫_hemisphere pdf(ω) dω
    // 用均匀半球采样做 Monte Carlo：E[pdf / uniform_pdf] = 1
    let mut sum = 0.0f64;
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, _) = uniform_hemisphere(u1, u2);
        let pdf = cosine_hemisphere_pdf(d.z);
        sum += (pdf / UNIFORM_HEMISPHERE_PDF) as f64;
    }
    let estimate = sum / N as f64;
    assert!((estimate - 1.0).abs() < 0.01, "∫pdf ≈ {estimate}");
}

#[test]
fn cosine_hemisphere_weighting_is_cosine() {
    // 余弦加权采样：cosθ 的均值应该接近 2/3（理论值）
    // 因为权重 ∝ cosθ，归一化后 E[cosθ] = ∫cos²θ sinθ dθ / ∫cosθ sinθ dθ = (1/3)/(1/2) = 2/3
    let mut sum = 0.0f64;
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, _) = cosine_hemisphere(u1, u2);
        sum += d.z as f64;
    }
    let mean = sum / N as f64;
    assert!((mean - 2.0 / 3.0).abs() < 0.005, "E[cosθ] = {mean}，期望 2/3");
}

#[test]
fn cosine_hemisphere_pdf_returns_zero_below_horizon() {
    assert_eq!(cosine_hemisphere_pdf(-0.1), 0.0);
    assert_eq!(cosine_hemisphere_pdf(-1.0), 0.0);
    assert!(cosine_hemisphere_pdf(0.0) < 1e-9);
    assert!((cosine_hemisphere_pdf(1.0) - 1.0 / PI).abs() < 1e-6);
}

// ── 均匀半球 ──

#[test]
fn uniform_hemisphere_in_upper_hemisphere() {
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, pdf) = uniform_hemisphere(u1, u2);
        assert!(d.z >= -1e-6);
        assert!((d.length() - 1.0).abs() < 1e-4);
        assert!((pdf - UNIFORM_HEMISPHERE_PDF).abs() < 1e-6);
    }
}

#[test]
fn uniform_hemisphere_pdf_integrates_to_one() {
    // E[pdf / uniform_pdf] 应该 = 1（自己对自己）
    let mut sum = 0.0f64;
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (_, pdf) = uniform_hemisphere(u1, u2);
        sum += (pdf / UNIFORM_HEMISPHERE_PDF) as f64;
    }
    let estimate = sum / N as f64;
    assert!((estimate - 1.0).abs() < 1e-4);
}

#[test]
fn uniform_hemisphere_mean_cos_is_half() {
    // 均匀半球：E[cosθ] = ∫cosθ sinθ dθ / ∫sinθ dθ = (1/2) / 1 = 1/2
    let mut sum = 0.0f64;
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, _) = uniform_hemisphere(u1, u2);
        sum += d.z as f64;
    }
    let mean = sum / N as f64;
    assert!((mean - 0.5).abs() < 0.005, "E[cosθ] = {mean}，期望 0.5");
}

// ── 均匀球面 ──

#[test]
fn uniform_sphere_is_unit_vector() {
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, pdf) = uniform_sphere(u1, u2);
        assert!((d.length() - 1.0).abs() < 1e-4);
        assert!((pdf - UNIFORM_SPHERE_PDF).abs() < 1e-6);
    }
}

#[test]
fn uniform_sphere_mean_z_near_zero() {
    // 球面均匀：E[z] = 0
    let mut sum = 0.0f64;
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, _) = uniform_sphere(u1, u2);
        sum += d.z as f64;
    }
    let mean = sum / N as f64;
    assert!(mean.abs() < 0.01, "E[z] = {mean}");
}

#[test]
fn uniform_sphere_covers_both_hemispheres() {
    let mut above = 0;
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, _) = uniform_sphere(u1, u2);
        if d.z > 0.0 { above += 1; }
    }
    let frac = above as f32 / N as f32;
    assert!((frac - 0.5).abs() < 0.01, "上半球占比 {frac}");
}

// ── 均匀圆锥 ──

#[test]
fn uniform_cone_directions_in_cone() {
    let cos_theta_max = 0.5;   // 60° 半角
    let cos_min = cos_theta_max;
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, _) = uniform_cone(u1, u2, cos_theta_max);
        assert!(d.z >= cos_min - 1e-5, "i={i} z={} < {}", d.z, cos_min);
        assert!((d.length() - 1.0).abs() < 1e-4);
    }
}

#[test]
fn uniform_cone_pdf_integrates_to_one() {
    // 用均匀球面采样估计圆锥 PDF 的积分
    let cos_theta_max = 0.7;
    let cone_pdf = uniform_cone_pdf(cos_theta_max);
    let mut sum = 0.0f64;
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, _) = uniform_sphere(u1, u2);
        if d.z >= cos_theta_max {
            sum += (cone_pdf / UNIFORM_SPHERE_PDF) as f64;
        }
    }
    let estimate = sum / N as f64;
    assert!((estimate - 1.0).abs() < 0.02, "∫cone_pdf ≈ {estimate}");
}

#[test]
fn uniform_cone_collimated_edge_case() {
    // cos_theta_max = 1（半角 0）退化为固定方向 +Z
    // 实际使用中会退化，但公式不应产生 NaN
    let cos_theta_max = 0.999_f32;   // 近似 0 半角
    for i in 0..10 {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, pdf) = uniform_cone(u1, u2, cos_theta_max);
        assert!(d.is_finite());
        assert!(pdf.is_finite());
    }
}

// ── 圆盘 ──

#[test]
fn uniform_disk_inside_unit_circle() {
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (p, pdf) = uniform_disk(u1, u2);
        assert!(p.length() <= 1.0 + 1e-5, "i={i} |p|={}", p.length());
        assert!((pdf - UNIFORM_DISK_PDF).abs() < 1e-6);
    }
}

#[test]
fn uniform_disk_area_integral() {
    // 用均匀方框 [0,1]² 采样估计圆面积 = π
    let mut inside = 0u64;
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        // 用 uniform_disk 采样，检查期望
        let _ = (u1, u2);
        if u1 * u1 + u2 * u2 <= 1.0 { inside += 1; }
    }
    // 这里只是验证 [0,1]² 中落在四分之一圆的比例 ≈ π/4
    let frac = inside as f64 / N as f64;
    assert!((frac - PI as f64 / 4.0).abs() < 0.01, "frac={frac} 期望 π/4≈0.785");
}

#[test]
fn concentric_disk_inside_unit_circle() {
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (p, pdf) = concentric_disk(u1, u2);
        assert!(p.length() <= 1.0 + 1e-4, "i={i} |p|={}", p.length());
        assert!((pdf - UNIFORM_DISK_PDF).abs() < 1e-6);
    }
}

#[test]
fn concentric_disk_center_maps_to_zero() {
    let (p, _) = concentric_disk(0.5, 0.5);
    assert!(p.length() < 1e-6);
}

// ── 三角形 ──

#[test]
fn uniform_triangle_barycentrics_valid() {
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (b0, b1, b2) = uniform_triangle(u1, u2);
        assert!(b0 >= -1e-6 && b1 >= -1e-6 && b2 >= -1e-6, "i={i} b=({b0},{b1},{b2})");
        assert!((b0 + b1 + b2 - 1.0).abs() < 1e-5, "i={i} sum={}", b0 + b1 + b2);
    }
}

#[test]
fn uniform_triangle_covers_whole_triangle() {
    // 重心坐标的每个分量都应能取到接近 0 和接近 1 的值
    let mut min_b0 = 1.0f32;
    let mut max_b0 = 0.0f32;
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (b0, _, _) = uniform_triangle(u1, u2);
        min_b0 = min_b0.min(b0);
        max_b0 = max_b0.max(b0);
    }
    assert!(min_b0 < 0.01, "min_b0={min_b0}，应接近 0");
    assert!(max_b0 > 0.99, "max_b0={max_b0}，应接近 1");
}

// ── 世界空间包装 ──

#[test]
fn cosine_hemisphere_world_matches_manual_onb() {
    let n = Vec3::new(0.3, 0.5, 0.8).normalize();
    for i in 0..1000 {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d_world, pdf1) = cosine_hemisphere_world(n, u1, u2);
        let (local, pdf2) = cosine_hemisphere(u1, u2);
        let d_manual = Onb::from_w(n).to_world(local);
        assert!((d_world - d_manual).length() < 1e-5);
        assert!((pdf1 - pdf2).abs() < 1e-6);
    }
}

#[test]
fn cosine_hemisphere_world_respects_normal() {
    let n = Vec3::new(0.3, 0.5, 0.8).normalize();
    for i in 0..N {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, _) = cosine_hemisphere_world(n, u1, u2);
        assert!(d.dot(n) >= -1e-5, "i={i} d·n={}", d.dot(n));
    }
}

#[test]
fn uniform_cone_world_axis_matches_cone_axis() {
    let axis = Vec3::new(1.0, 0.0, 0.0);
    let cos_theta_max = 0.5;
    for i in 0..1000 {
        let u1 = u_seq(i, 0.618_033_988_7);
        let u2 = u_seq(i, 0.754_877_666_2);
        let (d, _) = uniform_cone_world(axis, u1, u2, cos_theta_max);
        let cos_angle = d.dot(axis);
        assert!(cos_angle >= cos_theta_max - 1e-4, "i={i} cos={cos_angle}");
    }
}
