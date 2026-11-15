//! kzray-impl/src/geometry/triangle.rs
//! 三角形解析图元。
//!
//! 求交核心是**自由函数** [`intersect_triangle`]——不依赖任何类型，
//! mesh 遍历三角形时直接从顶点缓冲取三个顶点调用，零对象构造。
//! [`TriangleA`] 只是给 `ShapeA` / `AnyShapeA` 用的薄封装。

use glam::{Affine3A, Vec2, Vec3, Vec3A};
use kzray_data::Aabb;
use std::ops::Range;

use kzray_abs::ray::Ray;

use crate::geometry::{RayA, ShapeA, ShapeHit, shape::AreaSample};

// ═══════════════════════════════════════════════════════════════
// Triangle —— 表示层
// ═══════════════════════════════════════════════════════════════

/// 三角形（**运行时光几何表示**）。
///
/// 三个顶点，值语义。仅持有几何数据，不含求交逻辑。
/// **不序列化**——序列化走 `kzray_data::geometry::Triangle`（若存在）。
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
}

impl Triangle {
    /// 构造三角形，校验非退化。
    ///
    /// # Panics
    ///
    /// 三个顶点共线（面积为零）时 panic。mesh 加载路径应自行跳过退化
    /// 三角形，不要用 `new` 逐面构造。
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        assert!(
            (v1 - v0).cross(v2 - v0).length_squared() > 0.0,
            "triangle vertices are degenerate (collinear)"
        );
        Triangle { v0, v1, v2 }
    }

    /// 便捷转发，见 [`TriangleA::intersect`](ShapeA::intersect)。
    ///
    /// **仅适合单次调用**。热点循环里请预先转 `TriangleA`。
    #[inline]
    pub fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<ShapeHit> {
        TriangleA::from(*self).intersect(&RayA::from(*ray), t_range)
    }
}

// ═══════════════════════════════════════════════════════════════
// TriangleA —— SIMD 求交层
// ═══════════════════════════════════════════════════════════════

/// 三角形的 SIMD 内部表示，**仅用于求交热点路径**。
///
/// 顶点用 `Vec3A` 存储，避免每次求交转换。不序列化。
#[derive(Clone, Copy, Debug)]
pub struct TriangleA {
    pub v0: Vec3A,
    pub v1: Vec3A,
    pub v2: Vec3A,
}

impl TriangleA {
    /// 构造三角形，校验非退化。见 [`Triangle::new`]。
    pub fn new(v0: Vec3A, v1: Vec3A, v2: Vec3A) -> Self {
        assert!(
            (v1 - v0).cross(v2 - v0).length_squared() > 0.0,
            "triangle vertices are degenerate (collinear)"
        );
        TriangleA { v0, v1, v2 }
    }
}

// ═══════════════════════════════════════════════════════════════
// 自由函数：求交核心（mesh 复用入口）
// ═══════════════════════════════════════════════════════════════

/// 三角形命中的详细结果，含重心坐标。
///
/// 与 [`ShapeHit`] 的区别：`ShapeHit::uv` 是泛化的"参数化 UV"，对三角形
/// 它取重心坐标 `(u, v)` 作为默认值；但 **mesh 需要明确的重心坐标来插值
/// 顶点属性**（顶点 UV、法线、颜色），所以 mesh 求交应直接调用
/// [`intersect_triangle`] 并消费本类型，不要经由 `TriangleA::intersect`
/// 再从 `ShapeHit::uv` 反解。
#[derive(Clone, Copy, Debug)]
pub struct TriangleHit {
    pub t: f32,
    pub distance: f32,
    pub position: Vec3A,
    /// 单位几何法线，方向由顶点顺序（右手法则）决定。
    pub geometric_normal: Vec3A,
    /// 重心坐标，顶点 `v1` 的权重。
    pub u: f32,
    /// 重心坐标，顶点 `v2` 的权重。
    pub v: f32,
    // v0 的权重 = 1 - u - v
}

impl TriangleHit {
    /// 顶点 `v0` 的重心权重。
    #[inline]
    pub fn w(&self) -> f32 {
        1.0 - self.u - self.v
    }

    /// 按重心坐标插值三个顶点属性（mesh 层用）。
    ///
    /// 例：插值顶点 UV `uv0/uv1/uv2` 或顶点法线。
    #[inline]
    pub fn interpolate<T>(&self, a0: T, a1: T, a2: T) -> T
    where
        T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>,
    {
        a0 * self.w() + a1 * self.u + a2 * self.v
    }
}

/// Möller–Trumbore 三角形求交。
///
/// **自由函数**：不依赖 `Triangle` / `TriangleA`，mesh 遍历三角形时直接
/// 从顶点缓冲取三个顶点调用，零对象构造。
///
/// # 参数
///
/// - `v0/v1/v2`：三角形顶点（与 `ray` 同一坐标系，局部空间）；
/// - `cull_backface`：`true` 时背面命中返回 `None`；光追通常传 `false`
///   （薄片、玻璃需要双侧）。
///
/// # 返回
///
/// 命中返回 [`TriangleHit`]（含重心坐标）；`t` 落在 `t_range` 内。
/// 平行光线、退化三角形、背面（当 cull）均返回 `None`。
#[inline]
pub fn intersect_triangle(
    v0: Vec3A,
    v1: Vec3A,
    v2: Vec3A,
    ray: &RayA,
    t_range: Range<f32>,
    cull_backface: bool,
) -> Option<TriangleHit> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;

    let h = ray.dir.cross(edge2);
    let a = edge1.dot(h);

    // 平行（a≈0）或退化三角形。阈值作用于未归一化 dir，与 |dir| 相关，
    // 继承 Plane 的约定。
    if a.abs() < 1e-8 {
        return None;
    }
    if cull_backface && a < 0.0 {
        return None;
    }

    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * s.dot(h);
    if u < 0.0 || u > 1.0 {
        return None;
    }

    let q = s.cross(edge1);
    let v = f * ray.dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * edge2.dot(q);
    if t < t_range.start || t >= t_range.end {
        return None;
    }

    let position = ray.at(t);
    // 法线：edge1 × edge2，右手法则，正面朝外。
    // `a` 已排除退化，二次保护防 normalize NaN。
    let normal_raw = edge1.cross(edge2);
    let len_sq = normal_raw.length_squared();
    if len_sq <= 0.0 {
        return None;
    }
    let geometric_normal = normal_raw / len_sq.sqrt();

    let distance = t * ray.dir.length();

    Some(TriangleHit {
        t,
        distance,
        position,
        geometric_normal,
        u,
        v,
    })
}

// ═══════════════════════════════════════════════════════════════
// ShapeA impl —— 给 AnyShapeA / accel 层用的薄封装
// ═══════════════════════════════════════════════════════════════

impl ShapeA for TriangleA {
    fn intersect(&self, ray: &RayA, t_range: Range<f32>) -> Option<ShapeHit> {
        // 解析图元默认双侧；背面剔除由 mesh 层通过自由函数控制。
        let hit = intersect_triangle(self.v0, self.v1, self.v2, ray, t_range, false)?;
        Some(ShapeHit {
            t: hit.t,
            distance: hit.distance,
            position: hit.position.into(),
            geometric_normal: hit.geometric_normal.into(),
            // 解析三角形没有顶点 UV，用重心坐标作为默认参数化。
            // mesh 需要真实 UV 时，直接调 intersect_triangle 并 interpolate。
            uv: Vec2::new(hit.u, hit.v),
        })
    }

    /// 短路：Möller–Trumbore 判别部分通过即 `true`。
    ///
    /// 背面剔除语义与 `intersect` 一致：`ShapeA::intersect` 对三角形
    /// 固定双侧（`cull_backface = false`），`occluded` 保持双侧。
    fn occluded(&self, ray: &RayA, t_range: Range<f32>) -> bool {
        intersect_triangle_occluded(self.v0, self.v1, self.v2, ray, t_range, false)
    }

    fn sample_area(&self, u: Vec2) -> Option<AreaSample> {
        // 重心坐标均匀采样：u ∈ [0,1)² 映射到三角形内。
        //
        // 标准方法：sqrt 重参数化使面积均匀。
        let su = u.x.sqrt();
        let b0 = 1.0 - su;
        let b1 = u.y * su;
        let b2 = su - b1; // = su * (1 - u.y)

        let p = self.v0 * b0 + self.v1 * b1 + self.v2 * b2;

        // 法线与面积
        let edge1 = self.v1 - self.v0;
        let edge2 = self.v2 - self.v0;
        let cross = edge1.cross(edge2);
        let area = cross.length() * 0.5;
        if area <= 0.0 {
            return None; // 退化三角形
        }
        let normal = cross / (area * 2.0);

        Some(AreaSample {
            position: p,
            normal,
            pdf_area: 1.0 / area,
        })
    }
}

// ═══════════════════════════════════════════════════════════════
// 转换
// ═══════════════════════════════════════════════════════════════

impl From<Triangle> for TriangleA {
    #[inline]
    fn from(t: Triangle) -> Self {
        TriangleA {
            v0: t.v0.into(),
            v1: t.v1.into(),
            v2: t.v2.into(),
        }
    }
}

impl From<TriangleA> for Triangle {
    #[inline]
    fn from(t: TriangleA) -> Self {
        Triangle {
            v0: t.v0.into(),
            v1: t.v1.into(),
            v2: t.v2.into(),
        }
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

    /// 标准测试三角形：xy 平面，直角边 2，法线 +z。
    ///   v0=(0,0,0)  v1=(2,0,0)  v2=(0,2,0)
    /// 命中点 (0.5, 0.5, 0) 对应 u=0.25, v=0.25。
    fn tri_a() -> TriangleA {
        TriangleA::new(
            Vec3A::new(0.0, 0.0, 0.0),
            Vec3A::new(2.0, 0.0, 0.0),
            Vec3A::new(0.0, 2.0, 0.0),
        )
    }

    // ── 自由函数（mesh 复用路径） ────────────────────────────

    #[test]
    fn front_face_hit() {
        let tri = tri_a();
        let ray = ray_a(Vec3::new(0.5, 0.5, 1.0), Vec3::NEG_Z);
        let hit = intersect_triangle(tri.v0, tri.v1, tri.v2, &ray, INF, false).unwrap();

        assert!((hit.t - 1.0).abs() < 1e-5, "t = {}", hit.t);
        assert!((hit.position - Vec3A::new(0.5, 0.5, 0.0)).length() < 1e-5);
        assert!((hit.geometric_normal - Vec3A::Z).length() < 1e-5);
        assert!((hit.u - 0.25).abs() < 1e-5, "u = {}", hit.u);
        assert!((hit.v - 0.25).abs() < 1e-5, "v = {}", hit.v);
        assert!((hit.w() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn back_face_hit_when_not_culled() {
        let tri = tri_a();
        // 从 -z 朝 +z 射入，背面
        let ray = ray_a(Vec3::new(0.5, 0.5, -1.0), Vec3::Z);
        let hit = intersect_triangle(tri.v0, tri.v1, tri.v2, &ray, INF, false);
        assert!(hit.is_some());
        // 双侧命中，法线仍朝 +z（几何法线方向不变）
        assert!((hit.unwrap().geometric_normal - Vec3A::Z).length() < 1e-5);
    }

    #[test]
    fn back_face_culled() {
        let tri = tri_a();
        let ray = ray_a(Vec3::new(0.5, 0.5, -1.0), Vec3::Z);
        assert!(intersect_triangle(tri.v0, tri.v1, tri.v2, &ray, INF, true).is_none());
    }

    #[test]
    fn miss_outside_triangle() {
        let tri = tri_a();
        // 射线在 (1.5, 1.5)：u+v = 0.75+0.75 > 1，在三角形外
        let ray = ray_a(Vec3::new(1.5, 1.5, 1.0), Vec3::NEG_Z);
        assert!(intersect_triangle(tri.v0, tri.v1, tri.v2, &ray, INF, false).is_none());
    }

    #[test]
    fn parallel_ray_misses() {
        let tri = tri_a();
        // dir 在 xy 平面内，与三角形法线垂直
        let ray = ray_a(Vec3::new(0.5, 0.5, 1.0), Vec3::X);
        assert!(intersect_triangle(tri.v0, tri.v1, tri.v2, &ray, INF, false).is_none());
    }

    #[test]
    fn t_range_clips() {
        let tri = tri_a();
        let ray = ray_a(Vec3::new(0.5, 0.5, 1.0), Vec3::NEG_Z); // t = 1
        assert!(intersect_triangle(tri.v0, tri.v1, tri.v2, &ray, 0.0..0.5, false).is_none());
        assert!(intersect_triangle(tri.v0, tri.v1, tri.v2, &ray, 0.5..2.0, false).is_some());
    }

    #[test]
    fn non_normalized_dir_distance() {
        let tri = tri_a();
        // dir 长度 2，t = 0.5，distance = 1
        let ray = ray_a(Vec3::new(0.5, 0.5, 1.0), Vec3::NEG_Z * 2.0);
        let hit = intersect_triangle(tri.v0, tri.v1, tri.v2, &ray, INF, false).unwrap();
        assert!((hit.t - 0.5).abs() < 1e-5);
        assert!((hit.distance - 1.0).abs() < 1e-5);
    }

    #[test]
    fn degenerate_triangle_returns_none() {
        // 三点共线
        let v0 = Vec3A::ZERO;
        let v1 = Vec3A::X;
        let v2 = Vec3A::X * 2.0;
        let ray = ray_a(Vec3::new(0.5, 0.5, 1.0), Vec3::NEG_Z);
        assert!(intersect_triangle(v0, v1, v2, &ray, INF, false).is_none());
    }

    #[test]
    fn interpolate_weights() {
        let tri = tri_a();
        let ray = ray_a(Vec3::new(0.5, 0.5, 1.0), Vec3::NEG_Z);
        let hit = intersect_triangle(tri.v0, tri.v1, tri.v2, &ray, INF, false).unwrap();

        // 插值三个标量 10/20/40：0.5*10 + 0.25*20 + 0.25*40 = 20
        let value = hit.interpolate(10.0_f32, 20.0, 40.0);
        assert!((value - 20.0).abs() < 1e-5, "value = {value}");
    }

    // ── ShapeA impl ─────────────────────────────────────────

    #[test]
    fn shape_a_impl_uses_barycentric_uv() {
        let tri = tri_a();
        let ray = ray_a(Vec3::new(0.5, 0.5, 1.0), Vec3::NEG_Z);
        let hit = tri.intersect(&ray, INF).unwrap();
        assert!((hit.uv.x - 0.25).abs() < 1e-5, "uv.x = {}", hit.uv.x);
        assert!((hit.uv.y - 0.25).abs() < 1e-5, "uv.y = {}", hit.uv.y);
    }

    #[test]
    fn shape_a_impl_matches_free_fn() {
        let tri = tri_a();
        let ray = ray_a(Vec3::new(0.3, 0.4, 1.0), Vec3::NEG_Z);

        let a = tri.intersect(&ray, INF);
        let b = intersect_triangle(tri.v0, tri.v1, tri.v2, &ray, INF, false);

        match (a, b) {
            (Some(a), Some(b)) => {
                assert!((a.t - b.t).abs() < 1e-6);
            }
            (None, None) => {}
            _ => panic!("ShapeA impl and free fn diverged"),
        }
    }

    // ── 表示层 <-> 求交层 ────────────────────────────────────

    #[test]
    fn triangle_roundtrip() {
        let t = Triangle::new(
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let back: Triangle = TriangleA::from(t).into();
        assert!((t.v0 - back.v0).length() < 1e-6);
        assert!((t.v1 - back.v1).length() < 1e-6);
        assert!((t.v2 - back.v2).length() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "degenerate")]
    fn reject_degenerate_new() {
        Triangle::new(Vec3::ZERO, Vec3::X, Vec3::X * 2.0);
    }

    #[test]
    fn triangle_convenience_forward() {
        let t = Triangle::new(
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        );
        let ray = Ray::new(Vec3::new(0.5, 0.5, 1.0), Vec3::NEG_Z, 0.0);
        let hit = t.intersect(&ray, INF).unwrap();
        assert!((hit.t - 1.0).abs() < 1e-5);
    }
}

/// Möller–Trumbore 遮挡查询（自由函数，mesh 复用）。
///
/// 判别部分通过即 `true`，不算法线/位置/distance。`cull_backface` 语义
/// 同 [`intersect_triangle`]。
#[inline]
pub fn intersect_triangle_occluded(
    v0: Vec3A,
    v1: Vec3A,
    v2: Vec3A,
    ray: &RayA,
    t_range: Range<f32>,
    cull_backface: bool,
) -> bool {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;

    let h = ray.dir.cross(edge2);
    let a = edge1.dot(h);
    if a.abs() < 1e-8 {
        return false;
    }
    if cull_backface && a < 0.0 {
        return false;
    }

    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * s.dot(h);
    if u < 0.0 || u > 1.0 {
        return false;
    }

    let q = s.cross(edge1);
    let v = f * ray.dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = f * edge2.dot(q);
    t >= t_range.start && t < t_range.end
}

/// 三角形经仿射变换后的**世界空间** AABB。
///
/// 直接变换 3 个顶点取 min/max——三角形是平面图元，变换 3 点即得
/// 精确紧 AABB，无需 8 角点。
pub fn triangle_aabb_transformed(tri: &TriangleA, transform: &Affine3A) -> Aabb {
    let v0 = transform.transform_point3(Vec3::from(tri.v0));
    let v1 = transform.transform_point3(Vec3::from(tri.v1));
    let v2 = transform.transform_point3(Vec3::from(tri.v2));

    Aabb::from_point(v0).union_point(v1).union_point(v2)
}
