//! 简单场景。
//!
//! 暴力遍历求交（O(n)），无 BVH。适用于：
//!
//! - 阶段 2-4 的 PT 管线验证；
//! - 调试 / 单元测试；
//! - 图元数量很少的场景（< 100）。
//!
//! # 场景模型
//!
//! 当前简化模型：每个图元是一个独立的 instance（`prim_id = 0`），
//! `instance_id` 是图元在场景中的索引。材质按 instance 索引查。
//!
//! 阶段 5 接入 `RenderInput` 时，扩展为：
//!
//! - 多个 mesh 复用同一几何数据（不同 `instance_id`）；
//! - 一个 mesh 的多个图元按材质槽分组（`prim_id` 区分）；
//! - instance 变换。
//!
//! # 形状枚举
//!
//! [`Shape`] 是 `enum`，不是 trait——图元种类由实现层决定，封闭集合
//! （当前 Sphere / Plane）。未来加图元时扩展枚举，与 `Material` 用 enum
//! 是同一哲学。

//! 简单场景。

use std::ops::Range;

use glam::Vec3;

use kzray_abs::light::{InfiniteLight, Light, LightSampler};
use kzray_abs::material::Material;
use kzray_abs::ray::{Ray, SurfaceHit};
use kzray_abs::scene::Scene;
use kzray_abs::spectrum::SpectralAlgebra;
use kzray_data::common::{InstanceId, LightId};
use kzray_data::geometry::Aabb;

use crate::geometry::{GeometryHit, Plane, Sphere};

// ═══════════════════════════════════════════════════════════════
// Shape
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
enum Shape {
    Sphere(Sphere),
    Plane(Plane),
}


impl Shape {
    fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<GeometryHit> {
        match self {
            Shape::Sphere(s) => s.intersect(ray, t_range),
            Shape::Plane(p) => p.intersect(ray, t_range),
        }
    }

    fn bounds(&self) -> Option<Aabb> {
        match self {
            Shape::Sphere(s) => Some(Aabb::new(
                s.center - Vec3::splat(s.radius),
                s.center + Vec3::splat(s.radius),
            )),
            Shape::Plane(_) => None,
        }
    }
}

impl From<Sphere> for Shape {
    fn from(s: Sphere) -> Self {
        Shape::Sphere(s)
    }
}

impl From<Plane> for Shape {
    fn from(p: Plane) -> Self {
        Shape::Plane(p)
    }
}

// ═══════════════════════════════════════════════════════════════
// ShapeEntry
// ═══════════════════════════════════════════════════════════════

struct ShapeEntry<S: SpectralAlgebra> {
    shape: Shape,
    material: Material<S>,
    instance_id: InstanceId,
}

// ═══════════════════════════════════════════════════════════════
// SimpleScene
// ═══════════════════════════════════════════════════════════════

pub struct SimpleScene<S, LS, L, IL>
where
    S: SpectralAlgebra,
    LS: LightSampler,
    L: Light<S>,
    IL: InfiniteLight<S>,
{
    bounds: Aabb,
    shapes: Vec<ShapeEntry<S>>,
    lights: Vec<L>,
    light_sampler: Option<LS>,
    infinite_light: Option<IL>,
}

impl<S, LS, L, IL> Scene<S, LS, L, IL> for SimpleScene<S, LS, L, IL>
where
    S: SpectralAlgebra,
    LS: LightSampler,
    L: Light<S>,
    IL: InfiniteLight<S>,
{
    fn bounds(&self) -> Aabb {
        self.bounds
    }

    fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<SurfaceHit> {
        let mut best_t = t_range.end;
        let mut best_hit = None;
        for entry in &self.shapes {
            let range = t_range.start..best_t;
            if let Some(g) = entry.shape.intersect(ray, range) {
                best_t = g.t;
                best_hit = Some(g.into_surface_hit(entry.instance_id, 0));
            }
        }
        best_hit
    }

    fn intersect_p(&self, ray: &Ray, t_range: Range<f32>) -> bool {
        self.shapes
            .iter()
            .any(|e| e.shape.intersect(ray, t_range.clone()).is_some())
    }

    fn material(&self, hit: &SurfaceHit) -> Option<&Material<S>> {
        let idx = hit.instance_id.as_u64() as usize;
        self.shapes.get(idx).map(|e| &e.material)
    }

    fn light(&self, id: LightId) -> Option<&L> {
        self.lights.get(id.as_u64() as usize)
    }

    fn light_sampler(&self) -> Option<&LS> {
        self.light_sampler.as_ref()
    }

    fn infinite_light(&self) -> Option<&IL> {
        self.infinite_light.as_ref()
    }
}

// ═══════════════════════════════════════════════════════════════
// SimpleSceneBuilder
// ═══════════════════════════════════════════════════════════════

pub struct SimpleSceneBuilder<S, LS, L, IL>
where
    S: SpectralAlgebra,
    LS: LightSampler,
    L: Light<S>,
    IL: InfiniteLight<S>,
{
    shapes: Vec<ShapeEntry<S>>,
    lights: Vec<L>,
    light_sampler: Option<LS>,
    infinite_light: Option<IL>,
    explicit_bounds: Option<Aabb>,
}

impl<S, LS, L, IL> SimpleSceneBuilder<S, LS, L, IL>
where
    S: SpectralAlgebra,
    LS: LightSampler,
    L: Light<S>,
    IL: InfiniteLight<S>,
{
    pub fn new() -> Self {
        SimpleSceneBuilder {
            shapes: Vec::new(),
            lights: Vec::new(),
            light_sampler: None,
            infinite_light: None,
            explicit_bounds: None,
        }
    }

    pub fn add_shape(&mut self, shape: impl Into<Shape>, material: Material<S>) -> InstanceId {
        let instance_id = InstanceId::from(self.shapes.len() as u64);
        self.shapes.push(ShapeEntry {
            shape: shape.into(),
            material,
            instance_id,
        });
        instance_id
    }

    pub fn add_sphere(&mut self, sphere: Sphere, material: Material<S>) -> InstanceId {
        self.add_shape(sphere, material)
    }

    pub fn add_plane(&mut self, plane: Plane, material: Material<S>) -> InstanceId {
        self.add_shape(plane, material)
    }

    pub fn add_light(&mut self, light: L) -> LightId {
        let id = LightId::from(self.lights.len() as u64);
        self.lights.push(light);
        id
    }

    pub fn with_light_sampler(mut self, sampler: LS) -> Self {
        self.light_sampler = Some(sampler);
        self
    }

    pub fn with_infinite_light(mut self, light: IL) -> Self {
        self.infinite_light = Some(light);
        self
    }

    pub fn with_bounds(mut self, bounds: Aabb) -> Self {
        self.explicit_bounds = Some(bounds);
        self
    }

    pub fn build(self) -> SimpleScene<S, LS, L, IL> {
        let bounds = if let Some(b) = self.explicit_bounds {
            b
        } else {
            let mut acc = Aabb::empty();
            let mut has_infinite = false;
            for entry in &self.shapes {
                match entry.shape.bounds() {
                    Some(b) => acc = acc.union(&b),
                    None => has_infinite = true,
                }
            }
            if has_infinite {
                Aabb::empty()
            } else {
                acc
            }
        };
        SimpleScene {
            bounds,
            shapes: self.shapes,
            lights: self.lights,
            light_sampler: self.light_sampler,
            infinite_light: self.infinite_light,
        }
    }
}
