//! 无限平面。

use glam::Affine3A;
use glam::{Vec2, Vec3, Vec3A};
use kzray_data::Aabb;
use std::ops::Range;

use kzray_abs::ray::Ray;

use crate::geometry::RayA;
use crate::geometry::ShapeA;
use crate::geometry::ShapeHit;
use crate::geometry::shape::AreaSample;

/// 无限平面（**运行时光几何表示**）。
///
/// 由平面上一点和法线定义。法线不要求归一化，构造时归一化。
/// 仅持有几何数据，不含求交逻辑。求交见 [`PlaneA`]。
/// 需要求交时用 `PlaneA::from(plane)` 或 `plane.into()`。
///
/// **不序列化**——序列化一律走 `kzray_data::geometry::Plane`，
/// 通过 `From` 与 `kzray_data::geometry::Plane` 互转。
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
}

// ═══════════════════════════════════════════════════════════════
// PlaneA —— 求交层（SIMD 内部表示）
// ═══════════════════════════════════════════════════════════════

/// 无限平面的 SIMD 内部表示，**仅用于求交热点路径**。
///
/// 语义与 [`Plane`] 完全一致，唯一区别是 `point`/`normal` 用 `Vec3A`
/// 存储，避免每次求交都做 `Vec3 -> Vec3A` 转换。不序列化——序列化
/// 一律用 [`Plane`]。
///
/// # UV 参数化
///
/// 与 [`Plane`] 相同（见其文档）。
#[derive(Clone, Copy, Debug)]
pub struct PlaneA {
    /// 平面上一点。
    pub point: Vec3A,
    /// 单位法线。
    pub normal: Vec3A,
}

impl PlaneA {
    /// 构造平面，法线归一化。
    ///
    /// # Panics
    ///
    /// `normal` 长度为 0 时 panic。
    pub fn new(point: Vec3A, normal: Vec3A) -> Self {
        assert!(
            normal.length_squared() > 0.0,
            "plane normal must be non-zero"
        );
        PlaneA {
            point,
            normal: normal.normalize(),
        }
    }

    /// 构造水平面 `y = height`，法线 `+Y`。
    pub fn horizontal(height: f32) -> Self {
        PlaneA {
            point: Vec3A::new(0.0, height, 0.0),
            normal: Vec3A::Y,
        }
    }
}

impl ShapeA for PlaneA {
    /// 短路：只判 `t` 是否在区间，不算位置/UV/distance。
    fn occluded(&self, ray: &RayA, t_range: Range<f32>) -> bool {
        let denom = self.normal.dot(ray.dir);
        if denom.abs() < 1e-8 {
            return false;
        }
        let t = (self.point - ray.origin).dot(self.normal) / denom;
        t >= t_range.start && t < t_range.end
    }

    /// 光线与平面求交。
    ///
    /// 收 `&RayA`——调用方应在 trace 入口把 `Ray` 转成 `RayA` 一次，
    /// **不要在求交函数内部转换**。
    ///
    /// 平行光线（`|dot(normal, dir)| < 1e-8`）返回 `None`。
    /// 注意该阈值作用于**未归一化的 `dir`**，与 `|dir|` 相关；这是与
    /// 标量实现一致的语义，保持不改。
    fn intersect(&self, ray: &RayA, t_range: Range<f32>) -> Option<ShapeHit> {
        let denom = self.normal.dot(ray.dir);
        if denom.abs() < 1e-8 {
            return None;
        }

        let t = (self.point - ray.origin).dot(self.normal) / denom;
        if t < t_range.start || t >= t_range.end {
            return None;
        }

        // ── 热点内部全程 Vec3A ──
        let position_a = ray.at(t); // Vec3A
                                    // UV 直接取 Vec3A 的标量分量，无需转回 Vec3
        let uv = Vec2::new(position_a.x, position_a.z);
        let distance = t * ray.dir.length();

        // ── 只在输出边界转回标量 ──
        Some(ShapeHit {
            t,
            distance,
            position: position_a.into(),
            geometric_normal: self.normal.into(),
            uv,
        })
    }

    fn sample_area(&self, _u: Vec2) -> Option<AreaSample> {
        None // 无限平面无面积采样
    }
}

// ═══════════════════════════════════════════════════════════════
// 表示 <-> 求交 转换
// ═══════════════════════════════════════════════════════════════

impl From<Plane> for PlaneA {
    #[inline]
    fn from(p: Plane) -> Self {
        PlaneA {
            point: p.point.into(),
            normal: p.normal.into(),
        }
    }
}

impl From<PlaneA> for Plane {
    #[inline]
    fn from(p: PlaneA) -> Self {
        Plane {
            point: p.point.into(),
            normal: p.normal.into(),
        }
    }
}

impl Plane {
    /// 便捷转发：内部构造临时 `PlaneA` + `RayA` 后调用。
    ///
    /// **仅适合单次调用**。热点循环里请预先 `let pa: PlaneA = plane.into();`
    /// 然后复用 `pa.intersect(&ray_a, ...)`，避免逐次转换。
    #[inline]
    pub fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<ShapeHit> {
        PlaneA::from(*self).intersect(&RayA::from(*ray), t_range)
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const INF: Range<f32> = 0.0..f32::INFINITY;

    fn ray_a(origin: Vec3, dir: Vec3) -> RayA {
        RayA::from(Ray::new(origin, dir, 0.0))
    }

    // ── PlaneA 原生测试（热点路径） ──────────────────────────

    #[test]
    fn ray_hits_horizontal_plane() {
        let plane: PlaneA = Plane::horizontal(0.0).into();
        // 从上方垂直打下
        let ray = ray_a(Vec3::new(1.0, 5.0, 2.0), Vec3::NEG_Y);
        let hit = plane.intersect(&ray, INF).unwrap();
        assert!((hit.t - 5.0).abs() < 1e-5);
        assert!((hit.position - Vec3::new(1.0, 0.0, 2.0)).length() < 1e-5);
        assert!((hit.geometric_normal - Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn ray_misses_parallel_to_plane() {
        let plane: PlaneA = Plane::horizontal(0.0).into();
        // 光线与平面平行（y=1）
        let ray = ray_a(Vec3::new(0.0, 1.0, 0.0), Vec3::X);
        assert!(plane.intersect(&ray, INF).is_none());
    }

    #[test]
    fn ray_behind_plane_misses() {
        let plane: PlaneA = Plane::horizontal(0.0).into();
        // 光线朝 +y，平面在 y=0，但起点 y=1 → t < 0
        let ray = ray_a(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        assert!(plane.intersect(&ray, INF).is_none());
    }

    #[test]
    fn t_range_clips() {
        let plane: PlaneA = Plane::horizontal(0.0).into();
        let ray = ray_a(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y);
        // t=5，[0, 3] 排除
        assert!(plane.intersect(&ray, 0.0..3.0).is_none());
        // [3, 10] 包含
        assert!(plane.intersect(&ray, 3.0..10.0).is_some());
    }

    #[test]
    fn tilted_plane() {
        // 法线 (1, 1, 0)/√2 的平面，过原点
        let plane: PlaneA = Plane::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0)).into();
        // 从 (1, 1, 0) 沿平面法线反向（-n）出发，会经过原点
        let ray = ray_a(
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0).normalize(),
        );
        let hit = plane.intersect(&ray, INF).unwrap();
        // 交点应在原点
        assert!((hit.position - Vec3::ZERO).length() < 1e-5);
        // t = √2（起点到原点的距离，dir 归一化）
        assert!((hit.t - 2.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn non_normalized_dir_distance() {
        let plane: PlaneA = Plane::horizontal(0.0).into();
        let ray = ray_a(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y * 2.0);
        let hit = plane.intersect(&ray, INF).unwrap();
        // t = 2.5（仿射），distance = 5
        assert!((hit.t - 2.5).abs() < 1e-5);
        assert!((hit.distance - 5.0).abs() < 1e-5);
    }

    #[test]
    fn uv_from_xz_projection() {
        let plane: PlaneA = Plane::horizontal(0.0).into();
        let ray = ray_a(Vec3::new(3.0, 5.0, 7.0), Vec3::NEG_Y);
        let hit = plane.intersect(&ray, INF).unwrap();
        assert!((hit.uv.x - 3.0).abs() < 1e-5);
        assert!((hit.uv.y - 7.0).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "normal must be non-zero")]
    fn reject_zero_normal() {
        Plane::new(Vec3::ZERO, Vec3::ZERO);
    }

    #[test]
    #[should_panic(expected = "normal must be non-zero")]
    fn reject_zero_normal_a() {
        PlaneA::new(Vec3A::ZERO, Vec3A::ZERO);
    }

    // ── 表示层 <-> 求交层 一致性 ─────────────────────────────

    #[test]
    fn plane_intersect_forwards_to_plane_a() {
        let p = Plane::new(Vec3::new(0.3, -0.2, 0.5), Vec3::new(1.0, 2.0, -0.5));
        let r = Ray::new(Vec3::new(-5.0, 1.0, 0.5), Vec3::new(1.0, -0.3, 0.2), 0.0);

        let a = p.intersect(&r, INF);
        let b = PlaneA::from(p).intersect(&RayA::from(r), INF);

        match (a, b) {
            (Some(a), Some(b)) => {
                assert!((a.t - b.t).abs() < 1e-6);
                assert!((a.distance - b.distance).abs() < 1e-6);
                assert!((a.position - b.position).length() < 1e-5);
                assert!((a.geometric_normal - b.geometric_normal).length() < 1e-5);
                assert!((a.uv - b.uv).length() < 1e-5);
            }
            (None, None) => {}
            _ => panic!("forward result diverged: {a:?} vs {b:?}"),
        }
    }

    #[test]
    fn plane_roundtrip() {
        let p = Plane::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 2.0));
        let back: Plane = PlaneA::from(p).into();
        assert!((p.point - back.point).length() < 1e-6);
        // 构造时已归一化，往返后法线应不变
        assert!((p.normal - back.normal).length() < 1e-6);
    }
}

/// 无限平面经任何仿射变换后仍无界——返回无限 AABB。
///
/// 构建层需单独处理（顶层遍历、或 DCC 预处理成有限面片）。求交层
/// 如实报告"我是无限的"，不假装有界。
pub fn plane_aabb_transformed(_plane: &PlaneA, _transform: &Affine3A) -> Aabb {
    Aabb::new(Vec3::splat(f32::NEG_INFINITY), Vec3::splat(f32::INFINITY))
}
