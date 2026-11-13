//! 球体。

use glam::{Vec2, Vec3};
use std::ops::Range;

use kzray_abs::ray::Ray;

use super::GeometryHit;

// ═══════════════════════════════════════════════════════════════
// Sphere
// ═══════════════════════════════════════════════════════════════

/// 球体。
///
/// 解析求交，二次方程。`ray.dir` 可以不归一化——`a = |dir|²` 处理，
/// `distance = t × |dir|`。
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

    /// 光线与球求交。
    ///
    /// 返回 `[t_range.start, t_range.end]` 内最近的命中。
    ///
    /// 光线起点在球内时返回出射点（`t > 0`），因为进入点的 `t < 0` 不在
    /// `t_range` 内（假设 `t_range.start >= 0`）。如果要看内壁，传
    /// `t_range.start = 0` 且光线起点在球内，返回的是出射点（正向）。
    pub fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<GeometryHit> {
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

        let position = ray.at(t);
        let normal = (position - self.center) / self.radius;
        let uv = self.uv(position);
        let distance = t * ray.dir.length();

        Some(GeometryHit {
            t,
            distance,
            position,
            shading_normal: normal,
            geometric_normal: normal,
            uv,
        })
    }

    /// 球面 UV 参数化。
    fn uv(&self, p: Vec3) -> Vec2 {
        let local = (p - self.center) / self.radius;
        // v：北极到南极，[0, 1]
        let v = local.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
        // u：绕 y 轴一圈，atan2(z, x) ∈ [-π, π]，映射到 [0, 1)
        let phi = local.z.atan2(local.x);
        let u = phi / (2.0 * std::f32::consts::PI) + 0.5;
        Vec2::new(u, v)
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const INF: Range<f32> = 0.0..f32::INFINITY;

    #[test]
    fn ray_hits_sphere() {
        let sphere = Sphere::new(Vec3::ZERO, 1.0);
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 0.0);
        let hit = sphere.intersect(&ray, INF).unwrap();
        // 进入点在 (-1, 0, 0)，t = 4
        assert!((hit.t - 4.0).abs() < 1e-5);
        assert!((hit.position - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-5);
        assert!((hit.geometric_normal - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn ray_misses_sphere() {
        let sphere = Sphere::new(Vec3::ZERO, 1.0);
        // 光线在 y=2 处平行于 x 轴，球半径 1，不命中
        let ray = Ray::new(Vec3::new(-5.0, 2.0, 0.0), Vec3::X, 0.0);
        assert!(sphere.intersect(&ray, INF).is_none());
    }

    #[test]
    fn ray_behind_sphere_misses() {
        let sphere = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        // 光线朝 -x，球在 +x 方向
        let ray = Ray::new(Vec3::ZERO, Vec3::NEG_X, 0.0);
        assert!(sphere.intersect(&ray, INF).is_none());
    }

    #[test]
    fn t_range_clips() {
        let sphere = Sphere::new(Vec3::ZERO, 1.0);
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 0.0);
        // [0, 3] 不包含 t=4
        assert!(sphere.intersect(&ray, 0.0..3.0).is_none());
        // [3, 5] 包含 t=4
        assert!(sphere.intersect(&ray, 3.0..5.0).is_some());
    }

    #[test]
    fn closest_hit_wins() {
        let sphere = Sphere::new(Vec3::ZERO, 1.0);
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 0.0);
        // 两个交点 t=4, t=6，应返回 t=4
        let hit = sphere.intersect(&ray, INF).unwrap();
        assert!((hit.t - 4.0).abs() < 1e-5);
    }

    #[test]
    fn ray_inside_sphere_returns_exit() {
        let sphere = Sphere::new(Vec3::ZERO, 1.0);
        // 起点在球心，朝 +x
        let ray = Ray::new(Vec3::ZERO, Vec3::X, 0.0);
        let hit = sphere.intersect(&ray, INF).unwrap();
        // 出射点 (1, 0, 0)，t = 1
        assert!((hit.t - 1.0).abs() < 1e-5);
        assert!((hit.position - Vec3::X).length() < 1e-5);
        assert!((hit.geometric_normal - Vec3::X).length() < 1e-5);
    }

    #[test]
    fn non_normalized_dir_distance() {
        let sphere = Sphere::new(Vec3::ZERO, 1.0);
        // dir 长度 2，t 是仿射参数，distance = t × 2
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X * 2.0, 0.0);
        let hit = sphere.intersect(&ray, INF).unwrap();
        // t = 2（仿射），distance = 4
        assert!((hit.t - 2.0).abs() < 1e-5);
        assert!((hit.distance - 4.0).abs() < 1e-5);
    }

    #[test]
    fn uv_at_equator() {
        let sphere = Sphere::new(Vec3::ZERO, 1.0);
        // 从 +z 方向射向原点，命中 (0, 0, 1)
        // u = atan2(1, 0) / (2π) + 0.5 = (π/2)/(2π) + 0.5 = 0.75
        // v = acos(0) / π = 0.5
        let ray = Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::NEG_Z, 0.0);
        let hit = sphere.intersect(&ray, INF).unwrap();
        assert!((hit.position - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-5);
        assert!((hit.uv.x - 0.75).abs() < 1e-5);
        assert!((hit.uv.y - 0.5).abs() < 1e-5);
    }

    #[test]
    fn uv_seam_at_negative_x() {
        // 接缝在 x = -radius, z = 0。命中 (-1, 0, 0) 时 u 在接缝上，
        // 取 0.0 或 1.0 取决于 atan2 的符号约定和浮点细节。
        let sphere = Sphere::new(Vec3::ZERO, 1.0);
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 0.0);
        let hit = sphere.intersect(&ray, INF).unwrap();
        // atan2(0, -1) = π → u = 1.0（接缝的另一端）
        assert!(
            hit.uv.x < 1e-5 || (hit.uv.x - 1.0).abs() < 1e-5,
            "u at seam should be 0 or 1, got {}",
            hit.uv.x
        );
        // v 仍在赤道
        assert!((hit.uv.y - 0.5).abs() < 1e-5);
    }

    #[test]
    fn uv_at_north_pole() {
        let sphere = Sphere::new(Vec3::ZERO, 1.0);
        // 从上方垂直打下，命中北极 (0, 1, 0) → v = acos(1)/π = 0
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, 0.0);
        let hit = sphere.intersect(&ray, INF).unwrap();
        assert!((hit.uv.y - 0.0).abs() < 1e-5);
    }

    #[test]
    fn tangent_ray_touches_at_one_point() {
        // 光线恰好擦过球面（判别式 = 0），返回切点
        let sphere = Sphere::new(Vec3::ZERO, 1.0);
        // 光线 y=1 沿 x 轴，与球在 (0, 1, 0) 相切
        let ray = Ray::new(Vec3::new(-5.0, 1.0, 0.0), Vec3::X, 0.0);
        let result = sphere.intersect(&ray, INF);
        // 可能命中也可能因浮点精度视为 miss，接受两种
        if let Some(hit) = result {
            assert!((hit.position.y - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    #[should_panic(expected = "radius must be positive")]
    fn reject_zero_radius() {
        Sphere::new(Vec3::ZERO, 0.0);
    }
}
