use crate::ray::{Ray, SurfaceHit};

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: glam::Vec3,
    pub max: glam::Vec3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: glam::Vec3::splat(f32::INFINITY),
        max: glam::Vec3::splat(f32::NEG_INFINITY),
    };
    pub const fn new(min: glam::Vec3, max: glam::Vec3) -> Self { Aabb { min, max } }
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb { min: self.min.min(other.min), max: self.max.max(other.max) }
    }
    pub fn union_point(&self, p: glam::Vec3) -> Aabb {
        Aabb { min: self.min.min(p), max: self.max.max(p) }
    }
    pub fn center(&self) -> glam::Vec3 { (self.min + self.max) * 0.5 }
    pub fn surface_area(&self) -> f32 {
        if self.is_empty() { return 0.0; }
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }
}

pub trait Accelerator: Send + Sync {
    fn intersect(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<SurfaceHit>;
    fn occluded(&self, ray: &Ray, t_min: f32, t_max: f32) -> bool;
    fn bounds(&self) -> Aabb;
    fn stats(&self) -> AcceleratorStats { AcceleratorStats::default() }
}

#[derive(Clone, Debug, Default)]
pub struct AcceleratorStats {
    pub node_count: u64,
    pub prim_ref_count: u64,
    pub build_ms: u64,
}
