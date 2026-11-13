//! 无限平面。

use glam::{Vec2, Vec3};
use std::ops::Range;

use kzray_abs::ray::Ray;

use super::GeometryHit;

// ═══════════════════════════════════════════════════════════════
// Plane
// ═══════════════════════════════════════════════════════════════

/// 无限平面。
///
/// 由平面上一点和法线定义。法线不要求归一化，构造时归一化。
///
/// # UV 参数化
///
/// 用世界坐标的 `(x, z)` 投影（假设平面大致水平，法线接近 ±Y）。
/// 这是一个 hack——正确的平面 UV 需要切线 + 副切线，但无限平面没有
/// 自然的 UV 原点。对 PT 调试场景（地面 / 天花板 / 墙面）够用。
///
/// 未来如果平面需要纹理，加 `tangent` / `bitangent` 字段，按面内坐标
/// 计算 UV。
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    /// 平面上一点。
    pub point: Vec3,
    /// 单位法线。
    pub normal: Vec3,
}

impl Plane {
    /// 构造平面。
    ///
    /// # Panics
    ///
    /// `normal` 长度为 0 时 panic。
    pub fn new(point: Vec3, normal: Vec3) -> Self {
        assert!(
            normal.length_squared() > 0.0,
            "plane normal must be non-zero"
        );
        Plane {
            point,
            normal: normal.normalize(),
        }
    }

    /// 构造水平面 `y = height`，法线 `+Y`。
    pub fn horizontal(height: f32) -> Self {
        Plane {
            point: Vec3::new(0.0, height, 0.0),
            normal: Vec3::Y,
        }
    }

    /// 光线与平面求交。
    ///
    /// 平行光线（`|dot(normal, dir)| < eps`）返回 `None`。
    pub fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<GeometryHit> {
        let denom = self.normal.dot(ray.dir);
        if denom.abs() < 1e-8 {
            return None;
        }
        let t = (self.point - ray.origin).dot(self.normal) / denom;
        if t < t_range.start || t >= t_range.end {
            return None;
        }

        let position = ray.at(t);
        let uv = Self::uv_from_position(position);
        let distance = t * ray.dir.length();

        Some(GeometryHit {
            t,
            distance,
            position,
            shading_normal: self.normal,
            geometric_normal: self.normal,
            uv,
        })
    }

    /// 世界坐标 → UV。用 `(x, z)` 投影。
    fn uv_from_position(p: Vec3) -> Vec2 {
        Vec2::new(p.x, p.z)
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
    fn ray_hits_horizontal_plane() {
        let plane = Plane::horizontal(0.0);
        // 从上方垂直打下
        let ray = Ray::new(Vec3::new(1.0, 5.0, 2.0), Vec3::NEG_Y, 0.0);
        let hit = plane.intersect(&ray, INF).unwrap();
        assert!((hit.t - 5.0).abs() < 1e-5);
        assert!((hit.position - Vec3::new(1.0, 0.0, 2.0)).length() < 1e-5);
        assert!((hit.geometric_normal - Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn ray_misses_parallel_to_plane() {
        let plane = Plane::horizontal(0.0);
        // 光线与平面平行（y=1）
        let ray = Ray::new(Vec3::new(0.0, 1.0, 0.0), Vec3::X, 0.0);
        assert!(plane.intersect(&ray, INF).is_none());
    }

    #[test]
    fn ray_behind_plane_misses() {
        let plane = Plane::horizontal(0.0);
        // 光线朝 +y，平面在 y=0，但起点 y=1 → t < 0
        let ray = Ray::new(Vec3::new(0.0, 1.0, 0.0), Vec3::Y, 0.0);
        assert!(plane.intersect(&ray, INF).is_none());
    }

    #[test]
    fn t_range_clips() {
        let plane = Plane::horizontal(0.0);
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, 0.0);
        // t=5，[0, 3] 排除
        assert!(plane.intersect(&ray, 0.0..3.0).is_none());
        // [3, 10] 包含
        assert!(plane.intersect(&ray, 3.0..10.0).is_some());
    }

    #[test]
    fn tilted_plane() {
        // 法线 (1, 1, 0)/√2 的平面，过原点
        let plane = Plane::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0));
        // 从 (1, 1, 0) 沿平面法线反向（-n）出发，会经过原点
        let ray = Ray::new(
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0).normalize(),
            0.0,
        );
        let hit = plane.intersect(&ray, INF).unwrap();
        // 交点应在原点
        assert!((hit.position - Vec3::ZERO).length() < 1e-5);
        // t = √2（起点到原点的距离，dir 归一化）
        assert!((hit.t - 2.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn non_normalized_dir_distance() {
        let plane = Plane::horizontal(0.0);
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y * 2.0, 0.0);
        let hit = plane.intersect(&ray, INF).unwrap();
        // t = 2.5（仿射），distance = 5
        assert!((hit.t - 2.5).abs() < 1e-5);
        assert!((hit.distance - 5.0).abs() < 1e-5);
    }

    #[test]
    fn uv_from_xz_projection() {
        let plane = Plane::horizontal(0.0);
        let ray = Ray::new(Vec3::new(3.0, 5.0, 7.0), Vec3::NEG_Y, 0.0);
        let hit = plane.intersect(&ray, INF).unwrap();
        assert!((hit.uv.x - 3.0).abs() < 1e-5);
        assert!((hit.uv.y - 7.0).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "normal must be non-zero")]
    fn reject_zero_normal() {
        Plane::new(Vec3::ZERO, Vec3::ZERO);
    }
}
