//! kzray-impl/src/accel/brute_force.rs
//! 暴力加速器——遍历全部图元，用于验证求交语义。

use std::ops::Range;

use kzray_abs::{Ray, SpectralAlgebra, Library};
use kzray_data::Aabb;

use crate::{
    accel::{
        Accelerator, Bounded,
        accelerator::{Hit, Primitive},
    },
    geometry::RayA,
};

/// 暴力加速器：遍历全部图元，找最近命中。
///
/// **仅用于验证语义**——性能 O(n)，不用于生产。它的价值在于：
///
/// - 实现简单，逐行对应求交协议的定义，无加速结构可藏 bug；
/// - 作为 BVH 的对照——BVH 结果必须与暴力结果一致；
/// - 调试时排除加速结构 bug——暴力正确而 BVH 错误，问题必在 BVH。
///
/// 与 BVH 的关系是**同构**：两者实现同一个 [`Accelerator`] trait，
/// 上层（`trace` / `occluded` 调用点）通过 trait 对象或泛型互换，
/// 切换实现不改上层代码。
pub struct BruteForce {
    primitives: Vec<Primitive>,
    /// 所有图元世界 AABB 的并集。供上层快速剔除整条光线——若光线
    /// 与并集不交，整条光线直接 miss，无需遍历。
    bounds: Aabb,
}

impl BruteForce {
    /// 由图元列表构造。整体 AABB 自动计算。
    pub fn new(primitives: Vec<Primitive>) -> Self {
        let bounds = primitives
            .iter()
            .map(|p| p.bounds())
            .fold(Aabb::empty(), |acc, b| acc.union(&b));
        Self { primitives, bounds }
    }

    /// 图元列表（调试用）。
    pub fn primitives(&self) -> &[Primitive] {
        &self.primitives
    }
}

impl Accelerator for BruteForce {
    fn test<S: SpectralAlgebra>(
        &self,
        ray: &Ray,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> Option<Hit> {
        // 整体 AABB 快速剔除：光线与场景无交，直接返回。
        self.bounds
            .intersect_range(ray.origin, ray.dir, t_range.clone())?;

        let ray_a = RayA::from(*ray);
        let mut best: Option<Hit> = None;
        let mut t_max = t_range.end;

        for prim in &self.primitives {
            if let Some(hit) = prim.intersect(&ray_a, t_range.start..t_max, lib) {
                let t = hit.t();
                if t < t_max {
                    t_max = t;
                    best = Some(hit);
                }
            }
        }
        best
    }

    fn occluded<S: SpectralAlgebra>(
        &self,
        ray: &Ray,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> bool {
        if self
            .bounds
            .intersect_range(ray.origin, ray.dir, t_range.clone())
            .is_none()
        {
            return false;
        }

        let ray_a = RayA::from(*ray);
        // 任意图元遮挡即返回，不找最近——这是与 test 的核心区别。
        self.primitives
            .iter()
            .any(|prim| prim.occluded(&ray_a, t_range.clone(), lib))
    }

    fn test_all<S: SpectralAlgebra>(
        &self,
        ray: &Ray,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> Vec<Hit> {
        if self
            .bounds
            .intersect_range(ray.origin, ray.dir, t_range.clone())
            .is_none()
        {
            return Vec::new();
        }

        let ray_a = RayA::from(*ray);
        // 收集所有命中，不维护 t_max——这是"所有"而非"最近"。
        self.primitives
            .iter()
            .filter_map(|prim| prim.intersect(&ray_a, t_range.clone(), lib))
            .collect()
    }
}
