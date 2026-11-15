//! 球体。

use glam::{Affine3A, Vec3A};
use glam::{Vec2, Vec3};
use kzray_data::Aabb;
use std::ops::Range;

use kzray_abs::ray::Ray;

use crate::geometry::RayA;
use crate::geometry::ShapeA;
use crate::geometry::ShapeHit;
use crate::geometry::shape::AreaSample;

/// 球体（**运行时光几何表示**）。
///
/// 仅持有几何数据，不含求交逻辑。求交见 [`SphereA`]。
/// 需要求交时用 `SphereA::from(sphere)` 或 `sphere.into()`。
///
/// **不序列化**——序列化一律走 `kzray_data::geometry::Sphere`，
/// 通过 `From` 与 `kzray_data::geometry::Sphere` 互转。
///
/// # UV 参数化
///
/// 标准球面坐标：
///
/// - `u = atan2(z, x) / (2π) + 0.5`；
/// - `v = acos(y / radius) / π`。
///
/// `y` 轴为球体北极。接缝在 `x = -radius, z = 0`（`u = 0/1` 处）。
#[derive(Clone, Copy, Debug)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f32) -> Self {
        assert!(radius > 0.0, "sphere radius must be positive");
        Sphere { center, radius }
    }
}

// ═══════════════════════════════════════════════════════════════
// SphereA —— 求交层（SIMD 内部表示）
// ═══════════════════════════════════════════════════════════════

/// 球体的 SIMD 内部表示，**仅用于求交热点路径**。
///
/// 语义与 [`Sphere`] 完全一致，唯一区别是 `center` 用 `Vec3A` 存储，
/// 避免每次求交都做 `Vec3 -> Vec3A` 转换。不序列化——序列化一律用
/// [`Sphere`]。
///
/// # UV 参数化
///
/// 与 [`Sphere`] 相同（见其文档）。
#[derive(Clone, Copy, Debug)]
pub struct SphereA {
    pub center: Vec3A,
    pub radius: f32,
}

impl SphereA {
    pub fn new(center: Vec3A, radius: f32) -> Self {
        assert!(radius > 0.0, "sphere radius must be positive");
        SphereA { center, radius }
    }

    /// 由单位局部坐标（= 球面单位法线）算 UV。
    ///
    /// `local` 必须是单位向量。内部全程 `Vec3A`，避免在 UV 计算里转回
    /// `Vec3`；标量提取 `.x/.y/.z` 成本可忽略。
    #[inline]
    fn uv_from_local(local: Vec3A) -> Vec2 {
        // v：北极到南极，[0, 1]
        let v = local.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
        // u：绕 y 轴一圈，atan2(z, x) ∈ [-π, π]，映射到 [0, 1)
        let phi = local.z.atan2(local.x);
        let u = phi / (2.0 * std::f32::consts::PI) + 0.5;
        Vec2::new(u, v)
    }
}

impl ShapeA for SphereA {
    /// 短路：判别式 + 根区间判断，不构造 `ShapeHit`。
    ///
    /// 比 `intersect` 少一次 `ray.at`（位置）、一次除法（法线）、
    /// 一次 `sqrt`（distance）、两次 `Vec3A -> Vec3` 转换。
    fn occluded(&self, ray: &RayA, t_range: Range<f32>) -> bool {
        let oc = ray.origin - self.center;
        let a = ray.dir.length_squared();
        let half_b = oc.dot(ray.dir);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return false;
        }

        let sqrt_disc = discriminant.sqrt();
        let inv_a = 1.0 / a;
        let t0 = (-half_b - sqrt_disc) * inv_a;
        let t1 = (-half_b + sqrt_disc) * inv_a;

        // 任意根落在区间内即命中。不选最近、不算 distance。
        (t0 >= t_range.start && t0 < t_range.end) || (t1 >= t_range.start && t1 < t_range.end)
    }

    /// 光线与球求交。
    ///
    /// 收 `&RayA`——调用方应在 trace 入口把 `Ray` 转成 `RayA` 一次，
    /// **不要在求交函数内部转换**。
    ///
    /// 返回 `[t_range.start, t_range.end]` 内最近的命中。
    /// 光线起点在球内时返回出射点（`t > 0`），因为进入点的 `t < 0` 不在
    /// `t_range` 内（假设 `t_range.start >= 0`）。
    fn intersect(&self, ray: &RayA, t_range: Range<f32>) -> Option<ShapeHit> {
        let oc = ray.origin - self.center;
        let a = ray.dir.length_squared();
        let half_b = oc.dot(ray.dir);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrt_disc = discriminant.sqrt();
        let inv_a = 1.0 / a;

        // 两个根：先小后大
        let t0 = (-half_b - sqrt_disc) * inv_a;
        let t1 = (-half_b + sqrt_disc) * inv_a;

        // 选 [t_range] 内最小的
        let t = if t0 >= t_range.start && t0 < t_range.end {
            t0
        } else if t1 >= t_range.start && t1 < t_range.end {
            t1
        } else {
            return None;
        };

        // ── 热点内部全程 Vec3A，只算一次 local ──
        // 球面点的归一化局部坐标即为单位几何法线，二者合并计算。
        let position_a = ray.at(t); // Vec3A
        let local = (position_a - self.center) / self.radius; // 单位法线
        let uv = Self::uv_from_local(local);
        // 复用 a = |dir|²，避免二次 length()（length() 内部会再 sqrt）
        let distance = t * a.sqrt();

        // ── 只在输出边界转回标量 ──
        Some(ShapeHit {
            t,
            distance,
            position: position_a.into(),
            geometric_normal: local.into(),
            uv,
        })
    }

    fn sample_area(&self, u: Vec2) -> Option<AreaSample> {
        // 球面均匀采样：z = 1 - 2u.x（cos θ），φ = 2π u.y。
        //
        // 注意：sphere 的 center/radius 已是世界空间。
        let cos_theta = 1.0 - 2.0 * u.x;
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = 2.0 * std::f32::consts::PI * u.y;

        let normal = Vec3A::new(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin());
        let position = self.center + normal * self.radius;

        let area = 4.0 * std::f32::consts::PI * self.radius * self.radius;

        Some(AreaSample {
            position,
            normal,
            pdf_area: 1.0 / area,
        })
    }
}

// ═══════════════════════════════════════════════════════════════
// 表示 <-> 求交 转换
// ═══════════════════════════════════════════════════════════════

impl From<Sphere> for SphereA {
    #[inline]
    fn from(s: Sphere) -> Self {
        SphereA {
            center: s.center.into(),
            radius: s.radius,
        }
    }
}

impl From<SphereA> for Sphere {
    #[inline]
    fn from(s: SphereA) -> Self {
        Sphere {
            center: s.center.into(),
            radius: s.radius,
        }
    }
}

impl Sphere {
    /// 便捷转发：内部构造临时 `SphereA` + `RayA` 后调用。
    ///
    /// **仅适合单次调用**。热点循环里请预先 `let sa: SphereA = sphere.into();`
    /// 然后复用 `sa.intersect(&ray_a, ...)`，避免逐次转换。
    #[inline]
    pub fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<ShapeHit> {
        SphereA::from(*self).intersect(&RayA::from(*ray), t_range)
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const INF: Range<f32> = 0.0..f32::INFINITY;

    // ── SphereA 原生测试（热点路径） ──────────────────────────

    fn ray_a(origin: Vec3, dir: Vec3) -> RayA {
        RayA::from(Ray::new(origin, dir, 0.0))
    }

    #[test]
    fn ray_hits_sphere() {
        let sphere: SphereA = Sphere::new(Vec3::ZERO, 1.0).into();
        let ray = ray_a(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);
        let hit = sphere.intersect(&ray, INF).unwrap();
        assert!((hit.t - 4.0).abs() < 1e-5);
        assert!((hit.position - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-5);
        assert!((hit.geometric_normal - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn ray_misses_sphere() {
        let sphere: SphereA = Sphere::new(Vec3::ZERO, 1.0).into();
        let ray = ray_a(Vec3::new(-5.0, 2.0, 0.0), Vec3::X);
        assert!(sphere.intersect(&ray, INF).is_none());
    }

    #[test]
    fn ray_behind_sphere_misses() {
        let sphere: SphereA = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0).into();
        let ray = ray_a(Vec3::ZERO, Vec3::NEG_X);
        assert!(sphere.intersect(&ray, INF).is_none());
    }

    #[test]
    fn t_range_clips() {
        let sphere: SphereA = Sphere::new(Vec3::ZERO, 1.0).into();
        let ray = ray_a(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);
        assert!(sphere.intersect(&ray, 0.0..3.0).is_none());
        assert!(sphere.intersect(&ray, 3.0..5.0).is_some());
    }

    #[test]
    fn closest_hit_wins() {
        let sphere: SphereA = Sphere::new(Vec3::ZERO, 1.0).into();
        let ray = ray_a(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);
        let hit = sphere.intersect(&ray, INF).unwrap();
        assert!((hit.t - 4.0).abs() < 1e-5);
    }

    #[test]
    fn ray_inside_sphere_returns_exit() {
        let sphere: SphereA = Sphere::new(Vec3::ZERO, 1.0).into();
        let ray = ray_a(Vec3::ZERO, Vec3::X);
        let hit = sphere.intersect(&ray, INF).unwrap();
        assert!((hit.t - 1.0).abs() < 1e-5);
        assert!((hit.position - Vec3::X).length() < 1e-5);
        assert!((hit.geometric_normal - Vec3::X).length() < 1e-5);
    }

    #[test]
    fn non_normalized_dir_distance() {
        let sphere: SphereA = Sphere::new(Vec3::ZERO, 1.0).into();
        let ray = ray_a(Vec3::new(-5.0, 0.0, 0.0), Vec3::X * 2.0);
        let hit = sphere.intersect(&ray, INF).unwrap();
        assert!((hit.t - 2.0).abs() < 1e-5);
        assert!((hit.distance - 4.0).abs() < 1e-5);
    }

    #[test]
    fn uv_at_equator() {
        let sphere: SphereA = Sphere::new(Vec3::ZERO, 1.0).into();
        let ray = ray_a(Vec3::new(0.0, 0.0, 5.0), Vec3::NEG_Z);
        let hit = sphere.intersect(&ray, INF).unwrap();
        assert!((hit.position - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-5);
        assert!((hit.uv.x - 0.75).abs() < 1e-5);
        assert!((hit.uv.y - 0.5).abs() < 1e-5);
    }

    #[test]
    fn uv_seam_at_negative_x() {
        let sphere: SphereA = Sphere::new(Vec3::ZERO, 1.0).into();
        let ray = ray_a(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);
        let hit = sphere.intersect(&ray, INF).unwrap();
        assert!(
            hit.uv.x < 1e-5 || (hit.uv.x - 1.0).abs() < 1e-5,
            "u at seam should be 0 or 1, got {}",
            hit.uv.x
        );
        assert!((hit.uv.y - 0.5).abs() < 1e-5);
    }

    #[test]
    fn uv_at_north_pole() {
        let sphere: SphereA = Sphere::new(Vec3::ZERO, 1.0).into();
        let ray = ray_a(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y);
        let hit = sphere.intersect(&ray, INF).unwrap();
        assert!((hit.uv.y - 0.0).abs() < 1e-5);
    }

    #[test]
    fn tangent_ray_touches_at_one_point() {
        let sphere: SphereA = Sphere::new(Vec3::ZERO, 1.0).into();
        let ray = ray_a(Vec3::new(-5.0, 1.0, 0.0), Vec3::X);
        if let Some(hit) = sphere.intersect(&ray, INF) {
            assert!((hit.position.y - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    #[should_panic(expected = "radius must be positive")]
    fn reject_zero_radius() {
        Sphere::new(Vec3::ZERO, 0.0);
    }

    // ── 表示层 <-> 求交层 一致性 ─────────────────────────────

    #[test]
    fn sphere_intersect_forwards_to_sphere_a() {
        let s = Sphere::new(Vec3::new(0.3, -0.2, 0.5), 1.7);
        let r = Ray::new(Vec3::new(-5.0, 0.1, 0.0), Vec3::new(1.0, 0.0, 0.2), 0.0);

        let a = s.intersect(&r, INF);
        let b = SphereA::from(s).intersect(&RayA::from(r), INF);

        match (a, b) {
            (Some(a), Some(b)) => {
                assert!((a.t - b.t).abs() < 1e-6);
                assert!((a.position - b.position).length() < 1e-5);
                assert!((a.geometric_normal - b.geometric_normal).length() < 1e-5);
                assert!((a.uv - b.uv).length() < 1e-5);
            }
            (None, None) => {}
            _ => panic!("forward result diverged: {a:?} vs {b:?}"),
        }
    }

    #[test]
    fn sphere_roundtrip() {
        let s = Sphere::new(Vec3::new(1.0, 2.0, 3.0), 0.5);
        let back: Sphere = SphereA::from(s).into();
        assert_eq!(s.center, back.center);
        assert_eq!(s.radius, back.radius);
    }
}

/// 球体经仿射变换后的**世界空间** AABB。
///
/// 不走"局部 AABB → 8 角点变换"路径。球在任意仿射变换下变椭球，
/// 其世界 AABB 每轴半宽 = `radius × ||M_linear 第 i 行||`。只需 3 个
/// 平方和 + 3 次 `sqrt`，比变换 8 个角点更快且**精确**（8 角点变换
/// 对椭球是紧的，但构造路径多 5 次 `transform_point3`）。
pub fn sphere_aabb_transformed(sphere: &SphereA, transform: &Affine3A) -> Aabb {
    let center = transform.transform_point3(Vec3::from(sphere.center));
    let m = &transform.matrix3;
    let r = sphere.radius;

    // glam Mat3 列主序：第 i 行元素 = (x_axis[i], y_axis[i], z_axis[i])
    let rx =
        (m.x_axis.x * m.x_axis.x + m.y_axis.x * m.y_axis.x + m.z_axis.x * m.z_axis.x).sqrt() * r;
    let ry =
        (m.x_axis.y * m.x_axis.y + m.y_axis.y * m.y_axis.y + m.z_axis.y * m.z_axis.y).sqrt() * r;
    let rz =
        (m.x_axis.z * m.x_axis.z + m.y_axis.z * m.y_axis.z + m.z_axis.z * m.z_axis.z).sqrt() * r;

    let extent = Vec3::new(rx, ry, rz);
    Aabb::new(center - extent, center + extent)
}
