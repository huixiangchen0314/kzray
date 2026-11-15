//! kzray-data/src/geometry/aabb.rs
//! 轴对齐包围盒（AABB）。
//!
//! `Aabb` 是纯几何数据：轴对齐的最小/最大角点，用于世界范围、
//! 加速结构（BVH）、视锥剔除、体积 bounds。它不依赖 `Ray`——求交
//! 方法接受 `origin` / `dir` 参数，避免 `kzray-data` 依赖 `kzray-abs`。
//!
//! # 字段拆分为 6 个 `f32`
//!
//! 字段用 `min_x` / `min_y` / `min_z` / `max_x` / `max_y` / `max_z`
//! 而非 `Vec3`，原因是**序列化稳定性**：`glam::Vec3` 的 `Serialize`
//! 实现依赖 `glam` 的 `serde` feature，且其序列化格式（字段名、是否
//! 用数组）可能随 `glam` 版本变化。拆成 6 个 `f32` 后，`Aabb` 的
//! `Serialize` / `Deserialize` 是**数据层自己的契约**，不受 `glam`
//! 版本影响。
//!
//! 便利访问器 [`Aabb::min`] / [`Aabb::max`] 返回 `Vec3`，内部计算不受
//! 影响；构造用 [`Aabb::new`] 接受 `Vec3`，字段拆分的代价对调用方
//! 不可见。
//!
//! # 空盒语义
//!
//! [`Aabb::empty`] 返回 `min = +∞`、`max = -∞` 的盒，`is_empty()` 为
//! `true`。空盒是 `union` 的单位元：`empty().union(&a) == a`，
//! `a.union(&empty()) == a`。`union_point` 把点并入空盒得到单点盒。
//!
//! # 退化盒
//!
//! `min == max` 的盒（单点）是合法的，`surface_area` 为 0。求交时
//! 单点盒只对恰好穿过该点的光线命中。数据层的几何（薄面、线段）可能
//! 产生退化盒，不视为错误。
//!
//! # 非有限值
//!
//! 字段含 `NaN` 或 `±∞` 时，`is_empty` / `union` / `intersect_range`
//! 的行为未定义。调用方应保证角点有限；构建层的校验负责这一点。

use std::ops::Range;

use glam::Vec3;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════
// Aabb
// ═══════════════════════════════════════════════════════════════

/// 轴对齐包围盒。
///
/// 字段 `pub`，允许直接读写。构造用 [`Aabb::new`] 或 [`Aabb::empty`]；
/// 聚合用 [`Aabb::union`] / [`Aabb::union_point`]。
///
/// 6 个 `f32` 字段是序列化契约的一部分——字段名和顺序固定，不随
/// `glam` 版本变化。便利方法 [`Aabb::min`] / [`Aabb::max`] 返回
/// `Vec3` 供计算使用。
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// 最小角点的 x 分量。空盒时为 `+∞`。
    pub min_x: f32,
    /// 最小角点的 y 分量。空盒时为 `+∞`。
    pub min_y: f32,
    /// 最小角点的 z 分量。空盒时为 `+∞`。
    pub min_z: f32,
    /// 最大角点的 x 分量。空盒时为 `-∞`。
    pub max_x: f32,
    /// 最大角点的 y 分量。空盒时为 `-∞`。
    pub max_y: f32,
    /// 最大角点的 z 分量。空盒时为 `-∞`。
    pub max_z: f32,
}

impl Aabb {
    /// 空盒：`min = +∞`，`max = -∞`。
    ///
    /// `union` 的单位元。`union_point` 后得到单点盒。
    pub const fn empty() -> Self {
        Aabb {
            min_x: f32::INFINITY,
            min_y: f32::INFINITY,
            min_z: f32::INFINITY,
            max_x: f32::NEG_INFINITY,
            max_y: f32::NEG_INFINITY,
            max_z: f32::NEG_INFINITY,
        }
    }

    /// 构造指定角点的盒。
    ///
    /// 调用方应保证 `min <= max`（逐分量）；`is_empty` 检测反序
    /// （某分量 `min > max`），但不自动交换。
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Aabb {
            min_x: min.x,
            min_y: min.y,
            min_z: min.z,
            max_x: max.x,
            max_y: max.y,
            max_z: max.z,
        }
    }

    /// 单点盒。
    pub const fn from_point(p: Vec3) -> Self {
        Aabb {
            min_x: p.x,
            min_y: p.y,
            min_z: p.z,
            max_x: p.x,
            max_y: p.y,
            max_z: p.z,
        }
    }

    /// 最小角点。
    #[inline]
    pub const fn min(&self) -> Vec3 {
        Vec3::new(self.min_x, self.min_y, self.min_z)
    }

    /// 最大角点。
    #[inline]
    pub const fn max(&self) -> Vec3 {
        Vec3::new(self.max_x, self.max_y, self.max_z)
    }

    /// 是否是空盒（某分量 `min > max`）。
    ///
    /// `empty()` 返回 `true`；`union_point` 后返回 `false`。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min_x > self.max_x || self.min_y > self.max_y || self.min_z > self.max_z
    }

    /// 两个盒的并集。
    ///
    /// 交换、结合；`empty()` 是单位元。
    #[inline]
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            min_z: self.min_z.min(other.min_z),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
            max_z: self.max_z.max(other.max_z),
        }
    }

    /// 把点并入盒。
    ///
    /// 对空盒调用得到单点盒；对非空盒扩大范围以包含 `p`。
    #[inline]
    pub fn union_point(&self, p: Vec3) -> Aabb {
        Aabb {
            min_x: self.min_x.min(p.x),
            min_y: self.min_y.min(p.y),
            min_z: self.min_z.min(p.z),
            max_x: self.max_x.max(p.x),
            max_y: self.max_y.max(p.y),
            max_z: self.max_z.max(p.z),
        }
    }

    /// 盒中心。
    ///
    /// 空盒返回 `NaN`（`+∞ + -∞ = NaN`），调用方应先用 `is_empty` 检查。
    #[inline]
    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
            (self.min_z + self.max_z) * 0.5,
        )
    }

    /// 对角线向量 `max - min`。空盒为负值。
    #[inline]
    pub fn diagonal(&self) -> Vec3 {
        Vec3::new(
            self.max_x - self.min_x,
            self.max_y - self.min_y,
            self.max_z - self.min_z,
        )
    }

    /// 表面积。退化盒（某轴为 0）返回 0；空盒返回 0（负数 clamp 到 0）。
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let d = self.diagonal();
        if d.x < 0.0 || d.y < 0.0 || d.z < 0.0 {
            return 0.0;
        }
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// 点是否在盒内（含边界）。
    ///
    /// 空盒恒返回 `false`。
    #[inline]
    pub fn contains(&self, p: Vec3) -> bool {
        self.min_x <= p.x
            && p.x <= self.max_x
            && self.min_y <= p.y
            && p.y <= self.max_y
            && self.min_z <= p.z
            && p.z <= self.max_z
    }

    /// 光线与盒求交。
    ///
    /// 返回 `[t_enter, t_exit]`，与 `t_range` 求交后的区间。返回 `None`
    /// 表示光线不命中盒（或交集为空）。
    ///
    /// 参数用 `origin` / `dir` 而非 `&Ray`：`Aabb` 在 `kzray-data`，
    /// `Ray` 在 `kzray-abs`，避免依赖方向反转。
    ///
    /// `dir` 不要求归一化；`t` 是仿射参数。`dir` 某分量为 0 时，该轴
    /// 视为平行——`origin` 在该轴范围外则返回 `None`。
    ///
    /// `t_range` 应满足 `start < end`；`start >= end` 时返回 `None`。
    ///
    /// **返回值与 `t_range` 求交**：原始 slab 结果中 `t_enter` 可能为负
    /// （光线起点在盒内），但返回值被 `t_range.start` 裁剪。例如
    /// `origin` 在盒内、`t_range = 0.0..f32::INFINITY` 时返回
    /// `(0.0, t_exit)`，而非负的 `t_enter`。调用方需要原始进入点时，
    /// 用 `t_range.start = f32::NEG_INFINITY`。
    pub fn intersect_range(
        &self,
        origin: Vec3,
        dir: Vec3,
        t_range: Range<f32>,
    ) -> Option<(f32, f32)> {
        // 空盒
        if self.is_empty() {
            return None;
        }
        // 无效 t_range
        if t_range.start >= t_range.end {
            return None;
        }
        // 非有限输入
        if !origin.is_finite() || !dir.is_finite() {
            return None;
        }

        let mut t_min = t_range.start;
        let mut t_max = t_range.end;

        // slab method：逐轴计算 [t0, t1] 与 [t_min, t_max] 的交集
        let mins = [self.min_x, self.min_y, self.min_z];
        let maxs = [self.max_x, self.max_y, self.max_z];
        for i in 0..3 {
            let o = origin[i];
            let d = dir[i];
            let lo = mins[i];
            let hi = maxs[i];

            if d.abs() <= f32::EPSILON {
                // 平行于该轴：origin 必须在 [lo, hi] 内
                if o < lo || o > hi {
                    return None;
                }
            } else {
                let inv_d = 1.0 / d;
                let mut t0 = (lo - o) * inv_d;
                let mut t1 = (hi - o) * inv_d;
                if t0 > t1 {
                    std::mem::swap(&mut t0, &mut t1);
                }
                t_min = t_min.max(t0);
                t_max = t_max.min(t1);
                if t_min > t_max {
                    return None;
                }
            }
        }

        Some((t_min, t_max))
    }

    /// 光线与盒求交，返回进入的 `t`。
    ///
    /// 便利方法：等价于 `intersect_range(...).map(|(t0, _)| t0)`。
    /// 返回 `None` 表示不命中。
    ///
    /// 光线起点在盒内时返回 `t_range.start`（不是负的 `t_enter`），
    /// 因为进入点在 `t_range` 之外。
    pub fn hit(&self, origin: Vec3, dir: Vec3, t_range: Range<f32>) -> Option<f32> {
        self.intersect_range(origin, dir, t_range).map(|(t0, _)| t0)
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    // ─────────── 构造与空盒 ───────────

    #[test]
    fn empty_is_empty() {
        assert!(Aabb::empty().is_empty());
    }

    #[test]
    fn new_unit_cube_not_empty() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(!b.is_empty());
    }

    #[test]
    fn from_point_is_single_point() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let b = Aabb::from_point(p);
        assert_eq!(b.min(), p);
        assert_eq!(b.max(), p);
        assert!(!b.is_empty());
    }

    #[test]
    fn min_max_roundtrip() {
        let lo = Vec3::new(-1.0, -2.0, -3.0);
        let hi = Vec3::new(4.0, 5.0, 6.0);
        let b = Aabb::new(lo, hi);
        assert_eq!(b.min(), lo);
        assert_eq!(b.max(), hi);
    }

    #[test]
    fn reversed_min_max_is_empty() {
        let b = Aabb::new(Vec3::splat(1.0), Vec3::splat(-1.0));
        assert!(b.is_empty());
    }

    // ─────────── union ───────────

    #[test]
    fn union_with_empty_is_identity() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert_eq!(b.union(&Aabb::empty()), b);
        assert_eq!(Aabb::empty().union(&b), b);
    }

    #[test]
    fn union_point_into_empty_gives_point_box() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let b = Aabb::empty().union_point(p);
        assert_eq!(b, Aabb::from_point(p));
        assert!(!b.is_empty());
    }

    #[test]
    fn union_point_extends_box() {
        let b = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b2 = b.union_point(Vec3::splat(2.0));
        assert_eq!(b2.min(), Vec3::ZERO);
        assert_eq!(b2.max(), Vec3::splat(2.0));
    }

    #[test]
    fn union_is_commutative() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(0.0));
        let b = Aabb::new(Vec3::splat(0.0), Vec3::splat(1.0));
        assert_eq!(a.union(&b), b.union(&a));
    }

    #[test]
    fn union_is_associative() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(0.0));
        let b = Aabb::new(Vec3::splat(0.0), Vec3::splat(1.0));
        let c = Aabb::new(Vec3::splat(1.0), Vec3::splat(2.0));
        assert_eq!(a.union(&b).union(&c), a.union(&b.union(&c)));
    }

    // ─────────── 派生量 ───────────

    #[test]
    fn center_of_unit_cube() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert_eq!(b.center(), Vec3::ZERO);
    }

    #[test]
    fn diagonal_of_unit_cube() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert_eq!(b.diagonal(), Vec3::splat(2.0));
    }

    #[test]
    fn surface_area_of_unit_cube() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        // 边长 2，表面积 6 × 4 = 24
        assert!(approx_eq(b.surface_area(), 24.0));
    }

    #[test]
    fn surface_area_of_degenerate_box() {
        // 薄面：z 轴为 0
        let b = Aabb::new(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
        // 面积 = 2 × (2×2 + 2×0 + 0×2) = 8
        assert!(approx_eq(b.surface_area(), 8.0));
    }

    #[test]
    fn surface_area_of_empty_is_zero() {
        assert_eq!(Aabb::empty().surface_area(), 0.0);
    }

    // ─────────── contains ───────────

    #[test]
    fn contains_interior_point() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(b.contains(Vec3::ZERO));
        assert!(b.contains(Vec3::splat(0.5)));
    }

    #[test]
    fn contains_boundary() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(b.contains(Vec3::splat(1.0)));
        assert!(b.contains(Vec3::splat(-1.0)));
    }

    #[test]
    fn does_not_contain_outside() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(!b.contains(Vec3::splat(1.1)));
        assert!(!b.contains(Vec3::new(0.0, 0.0, 2.0)));
    }

    #[test]
    fn empty_does_not_contain_anything() {
        assert!(!Aabb::empty().contains(Vec3::ZERO));
    }

    // ─────────── intersect_range ───────────

    #[test]
    fn ray_through_unit_cube_hits() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let dir = Vec3::X;
        let r = b.intersect_range(origin, dir, 0.0..f32::INFINITY).unwrap();
        assert!(approx_eq(r.0, 4.0));
        assert!(approx_eq(r.1, 6.0));
    }

    #[test]
    fn ray_misses_unit_cube() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let origin = Vec3::new(-5.0, 5.0, 0.0);
        let dir = Vec3::X;
        assert!(b.intersect_range(origin, dir, 0.0..f32::INFINITY).is_none());
    }

    #[test]
    fn ray_origin_inside_hits() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let r = b.intersect_range(Vec3::ZERO, Vec3::X, 0.0..f32::INFINITY).unwrap();
        // 原始 slab 的 t_enter = -1.0，但被 t_range.start = 0.0 裁剪。
        // 返回值是 [t_range.start, t_exit]。
        assert!(approx_eq(r.0, 0.0));
        assert!(approx_eq(r.1, 1.0));
    }

    #[test]
    fn ray_origin_inside_with_negative_t_range_shows_raw_enter() {
        // 显式用负的 t_range.start，可以看到原始 t_enter
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let r = b
            .intersect_range(Vec3::ZERO, Vec3::X, f32::NEG_INFINITY..f32::INFINITY)
            .unwrap();
        assert!(approx_eq(r.0, -1.0));
        assert!(approx_eq(r.1, 1.0));
    }

    #[test]
    fn t_range_clips_intersection() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let dir = Vec3::X;
        let r = b.intersect_range(origin, dir, 0.0..5.0).unwrap();
        assert!(approx_eq(r.0, 4.0));
        assert!(approx_eq(r.1, 5.0));
    }

    #[test]
    fn t_range_excludes_hit_returns_none() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let dir = Vec3::X;
        assert!(b.intersect_range(origin, dir, 0.0..3.0).is_none());
    }

    #[test]
    fn parallel_ray_inside_slab_hits() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let r = b.intersect_range(origin, dir, 0.0..f32::INFINITY).unwrap();
        assert!(approx_eq(r.0, 4.0));
        assert!(approx_eq(r.1, 6.0));
    }

    #[test]
    fn parallel_ray_outside_slab_misses() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let origin = Vec3::new(-5.0, 5.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        assert!(b.intersect_range(origin, dir, 0.0..f32::INFINITY).is_none());
    }

    #[test]
    fn empty_box_never_hits() {
        assert!(Aabb::empty()
            .intersect_range(Vec3::ZERO, Vec3::X, 0.0..f32::INFINITY)
            .is_none());
    }

    #[test]
    fn invalid_t_range_returns_none() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(b
            .intersect_range(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 5.0..5.0)
            .is_none());
        assert!(b
            .intersect_range(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 5.0..3.0)
            .is_none());
    }

    #[test]
    fn non_finite_input_returns_none() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(b
            .intersect_range(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::X, 0.0..f32::INFINITY)
            .is_none());
        assert!(b
            .intersect_range(
                Vec3::ZERO,
                Vec3::new(f32::INFINITY, 0.0, 0.0),
                0.0..f32::INFINITY
            )
            .is_none());
    }

    // ─────────── hit ───────────

    #[test]
    fn hit_returns_enter_t() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let t = b
            .hit(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 0.0..f32::INFINITY)
            .unwrap();
        assert!(approx_eq(t, 4.0));
    }

    #[test]
    fn hit_returns_none_on_miss() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(b
            .hit(Vec3::new(-5.0, 5.0, 0.0), Vec3::X, 0.0..f32::INFINITY)
            .is_none());
    }

    // ─────────── Send + Sync ───────────

    #[test]
    fn aabb_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Aabb>();
    }
}
