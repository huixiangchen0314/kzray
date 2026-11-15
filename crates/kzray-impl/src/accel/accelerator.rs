//! kzray-impl/src/accel/accelerator.rs

// TODO 并行加速时候，引入 Arc
use std::ops::Range;

use kzray_abs::Ray;
use kzray_data::Aabb;

use crate::{
    accel::{
        animated::AnimatedShapeInstance,
        instance::{InstanceHit, InstancePrimitive, ShapeInstance},
        resource::ResourceTable,
    },
    geometry::RayA,
};

pub trait Bounded {
    fn bounds(&self) -> Aabb;
}

pub enum Primitive {
    Instance { shape: ShapeInstance },
    Animated { shape: AnimatedShapeInstance },
    // MeshTriangle(MeshTriangle),
}

impl Primitive {
    /// 最近命中（世界空间，含 alpha / 动画）。
    pub fn intersect(&self, ray: &RayA, t_range: Range<f32>, table: &ResourceTable) -> Option<Hit> {
        let instance_hit = match self {
            Self::Instance { shape } => shape.intersect(ray, t_range, table)?,
            Self::Animated { shape } => shape.intersect(ray, t_range, table)?,
        };
        Some(Hit::Instance(instance_hit))
    }

    /// 遮挡查询（世界空间，含 alpha / 动画）。
    pub fn occluded(&self, ray: &RayA, t_range: Range<f32>, table: &ResourceTable) -> bool {
        match self {
            Self::Instance { shape } => shape.occluded(ray, t_range, table),
            Self::Animated { shape } => shape.occluded(ray, t_range, table),
        }
    }
}

impl Bounded for Primitive {
    #[inline]
    fn bounds(&self) -> Aabb {
        match self {
            Self::Instance { shape } => shape.bounds(),
            Self::Animated { shape } => shape.bounds(),
        }
    }
}

/// 加速精确相交结果
pub enum Hit {
    /// 实例相交结果
    Instance(InstanceHit),
    // 网格三角形相交结果
    // MeshTriangleHit,
}

impl Hit {
    /// 命中的仿射参数 `t`。
    #[inline]
    pub fn t(&self) -> f32 {
        match self {
            Self::Instance(h) => h.t(),
        }
    }
}

pub trait Accelerator {
    /// 最近命中。
    fn test(&self, ray: &Ray, t_range: Range<f32>, table: &ResourceTable) -> Option<Hit>;

    /// 遮挡查询：`t_range` 内任意可接受命中存在即 `true`。
    fn occluded(&self, ray: &Ray, t_range: Range<f32>, table: &ResourceTable) -> bool;

    /// 所有命中（调试/验证用）。
    fn test_all(&self, ray: &Ray, t_range: Range<f32>, table: &ResourceTable) -> Vec<Hit>;
}
