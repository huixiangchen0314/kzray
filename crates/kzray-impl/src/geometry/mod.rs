
use std::ops::Range;


pub mod plane;
pub mod sphere;
pub mod shape;
pub mod raya;
pub mod triangle;

use glam::{Affine3A, Vec2};
use kzray_data::Aabb;
pub use plane::{Plane, PlaneA};
pub use sphere::{Sphere, SphereA};
pub use shape::{ShapeA, ShapeHit, AreaSample};
pub use triangle::{Triangle, TriangleA, TriangleHit};
pub use raya::RayA;

/// 形状枚举
#[derive(Clone, Copy, Debug)]
pub enum AnyShapeA {
    Sphere(SphereA),
    Plane(PlaneA),
    Triangle(TriangleA),
}

impl AnyShapeA {
    /// 经仿射变换后的世界空间 AABB。
    ///
    /// 各形状**直接构造世界 AABB**，不经过"局部 AABB → 8 角点变换"
    /// 的通用路径：
    ///
    /// - 球：`center' ± radius × ||M 行||`，3 次 `sqrt`；
    /// - 三角形：变换 3 顶点取 min/max；
    /// - 无限平面：直接返回无限盒。
    ///
    /// 构建期调用，不在热点路径。
    #[inline]
    pub fn world_aabb(&self, transform: &Affine3A) -> Aabb {
        match self {
            Self::Sphere(s) => sphere::sphere_aabb_transformed(s, transform),
            Self::Plane(p) => plane::plane_aabb_transformed(p, transform),
            Self::Triangle(t) => triangle::triangle_aabb_transformed(t, transform),
        }
    }

    /// 表面积。`Plane` 无界返回 `None`。
    ///
    /// 构建期调用（`ShapeAreaLight::phi`），不在热点路径。
    pub fn surface_area(&self) -> Option<f32> {
        match self {
            Self::Sphere(s) => Some(4.0 * std::f32::consts::PI * s.radius * s.radius),
            Self::Plane(_) => None,
            Self::Triangle(t) => {
                let e1 = t.v1 - t.v0;
                let e2 = t.v2 - t.v0;
                Some(e1.cross(e2).length() * 0.5)
            }
        }
    }
}

impl ShapeA for AnyShapeA{
    fn intersect(&self, ray: &RayA, t_range: Range<f32>) -> Option<ShapeHit> {
        match self {
            Self::Sphere(s) => s.intersect(ray, t_range),
            Self::Plane(p) => p.intersect(ray, t_range),
            Self::Triangle(p)=>p.intersect(ray, t_range),
        }
    }

    #[inline]
    fn occluded(&self, ray: &RayA, t_range: Range<f32>) -> bool {
        // 显式调各形状的覆写版本，确保短路路径生效（不依赖默认实现）。
        match self {
            Self::Sphere(s) => s.occluded(ray, t_range),
            Self::Plane(p) => p.occluded(ray, t_range),
            Self::Triangle(t) => t.occluded(ray, t_range),
        }
    }

    fn sample_area(&self, u: Vec2) -> Option<AreaSample> {
        match self {
            Self::Sphere(s) => s.sample_area(u),
            Self::Plane(p) => p.sample_area(u),   // 返回 None
            Self::Triangle(t) => t.sample_area(u),
        }
    }
}
