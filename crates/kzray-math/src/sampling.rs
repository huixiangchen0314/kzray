//! 采样。所有方向采样返回**局部空间**方向（`z` 轴为法线），
//! 调用方用 `Onb::to_world` 转到世界空间。
//!
//! 所有 PDF 都是**立体角密度**（方向采样）或**面积密度**（圆盘、三角形）。
//! 所有 `u` 参数期望在 `[0, 1)`；边界 `u = 1` 由调用方保证不会出现。

use glam::{Vec2, Vec3};
use std::f32::consts::PI;

use crate::onb::Onb;

// ═══════════════════════════════════════════════════════════════
// 半球
// ═══════════════════════════════════════════════════════════════

/// 余弦加权半球采样（Malley 方法）。
///
/// 局部空间 `z ≥ 0`，PDF `= cosθ / π`。
/// 返回 `(local_dir, pdf)`。
#[inline]
pub fn cosine_hemisphere(u1: f32, u2: f32) -> (Vec3, f32) {
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - u1).max(0.0).sqrt();
    let dir = Vec3::new(x, y, z);
    (dir, z / PI)
}

/// 余弦加权半球的 PDF。`cos_theta = wi · n`。
#[inline]
pub fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    if cos_theta <= 0.0 { 0.0 } else { cos_theta / PI }
}

/// 均匀半球采样。
///
/// 局部空间 `z ≥ 0`，PDF `= 1 / (2π)`。
#[inline]
pub fn uniform_hemisphere(u1: f32, u2: f32) -> (Vec3, f32) {
    let z = u1;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    let dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
    (dir, UNIFORM_HEMISPHERE_PDF)
}

pub const UNIFORM_HEMISPHERE_PDF: f32 = 1.0 / (2.0 * PI);

// ═══════════════════════════════════════════════════════════════
// 球面
// ═══════════════════════════════════════════════════════════════

/// 均匀球面采样。
///
/// 全向，PDF `= 1 / (4π)`。
#[inline]
pub fn uniform_sphere(u1: f32, u2: f32) -> (Vec3, f32) {
    let z = 1.0 - 2.0 * u1;   // [-1, 1]
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    let dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
    (dir, UNIFORM_SPHERE_PDF)
}

pub const UNIFORM_SPHERE_PDF: f32 = 1.0 / (4.0 * PI);

// ═══════════════════════════════════════════════════════════════
// 圆锥（聚光灯）
// ═══════════════════════════════════════════════════════════════

/// 均匀圆锥采样。围绕局部 `+Z` 轴，半角 `theta_max`。
///
/// `cos_theta_max = cos(theta_max)`。返回值在圆锥内，PDF 均匀。
#[inline]
pub fn uniform_cone(u1: f32, u2: f32, cos_theta_max: f32) -> (Vec3, f32) {
    let cos_theta = 1.0 - u1 * (1.0 - cos_theta_max);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    let dir = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
    let pdf = 1.0 / (2.0 * PI * (1.0 - cos_theta_max));
    (dir, pdf)
}

/// 圆锥采样的 PDF。
#[inline]
pub fn uniform_cone_pdf(cos_theta_max: f32) -> f32 {
    1.0 / (2.0 * PI * (1.0 - cos_theta_max))
}

// ═══════════════════════════════════════════════════════════════
// 圆盘（景深、面光源）
// ═══════════════════════════════════════════════════════════════

/// 均匀圆盘采样。返回单位圆内的点，PDF 关于面积 `= 1/π`。
#[inline]
pub fn uniform_disk(u1: f32, u2: f32) -> (Vec2, f32) {
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let p = Vec2::new(r * phi.cos(), r * phi.sin());
    (p, UNIFORM_DISK_PDF)
}

pub const UNIFORM_DISK_PDF: f32 = 1.0 / PI;

/// 同心圆盘采样（Shirley-Chiu 1997）。
///
/// 相比 `uniform_disk` 的更均匀映射，避免极坐标变换在圆心的密度挤压。
/// 返回单位圆内点，PDF 关于面积 `= 1/π`。
#[inline]
pub fn concentric_disk(u1: f32, u2: f32) -> (Vec2, f32) {
    let a = 2.0 * u1 - 1.0;
    let b = 2.0 * u2 - 1.0;
    if a == 0.0 && b == 0.0 {
        return (Vec2::ZERO, UNIFORM_DISK_PDF);
    }
    let (r, phi) = if a.abs() > b.abs() {
        (a, (PI / 4.0) * (b / a))
    } else {
        (b, PI / 2.0 - (PI / 4.0) * (a / b))
    };
    let p = Vec2::new(r * phi.cos(), r * phi.sin());
    (p, UNIFORM_DISK_PDF)
}

// ═══════════════════════════════════════════════════════════════
// 三角形（面光源）
// ═══════════════════════════════════════════════════════════════

/// 均匀三角形采样（Turk 1990）。返回重心坐标 `(b0, b1, b2)`，和 = 1。
///
/// 使用：`p = b0 * v0 + b1 * v1 + b2 * v2`。
/// 关于**面积**的 PDF `= 1 / area(triangle)`。
#[inline]
pub fn uniform_triangle(u1: f32, u2: f32) -> (f32, f32, f32) {
    let (u1, u2) = if u1 + u2 > 1.0 {
        (1.0 - u1, 1.0 - u2)
    } else {
        (u1, u2)
    };
    let b0 = 1.0 - u1 - u2;
    (b0, u1, u2)
}

// ═══════════════════════════════════════════════════════════════
// 世界空间便捷包装
// ═══════════════════════════════════════════════════════════════

/// 余弦加权半球采样，直接返回世界空间方向。内部构造 `Onb` 并变换。
#[inline]
pub fn cosine_hemisphere_world(n: Vec3, u1: f32, u2: f32) -> (Vec3, f32) {
    let (local, pdf) = cosine_hemisphere(u1, u2);
    (Onb::from_w(n).to_world(local), pdf)
}

/// 均匀半球采样，世界空间方向。
#[inline]
pub fn uniform_hemisphere_world(n: Vec3, u1: f32, u2: f32) -> (Vec3, f32) {
    let (local, pdf) = uniform_hemisphere(u1, u2);
    (Onb::from_w(n).to_world(local), pdf)
}

/// 均匀圆锥采样，世界空间方向。`axis` 是圆锥的对称轴（单位向量）。
#[inline]
pub fn uniform_cone_world(
    axis: Vec3,
    u1: f32,
    u2: f32,
    cos_theta_max: f32,
) -> (Vec3, f32) {
    let (local, pdf) = uniform_cone(u1, u2, cos_theta_max);
    (Onb::from_w(axis).to_world(local), pdf)
}
