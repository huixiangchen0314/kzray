//! kzray-impl/src/scene/simple.rs  简单场景。
//!
//! 几何求交委托给 [`BruteForce`]；材质按 `InstanceId` 索引查表。
//! `Library<S>` 统一持有网格、纹理（alpha 等）；`SimpleScene<S, ...>`
//! 泛型只影响材质与光源，加速层通过方法级 `S` 接收 `&Library<S>`。

use std::ops::Range;

use glam::Affine3A;

use kzray_abs::light::{InfiniteLight, Light, LightSampler};
use kzray_abs::material::Material;
use kzray_abs::ray::{Ray, SurfaceHit};
use kzray_abs::scene::Scene;
use kzray_abs::spectrum::SpectralAlgebra;
use kzray_abs::Library;
use kzray_data::common::{InstanceId, LightId};
use kzray_data::geometry::Aabb;
use kzray_data::TextureId;

use crate::accel::instance::ShapeInstance;
use crate::accel::{Accelerator, BruteForce, Hit, InstanceHit, Primitive};
use crate::geometry::{AnyShapeA, Plane, Sphere};

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
    /// 几何求交的加速结构。不泛型——不感知材质/谱。
    accelerator: BruteForce,
    /// 统一数据资产表（网格、纹理）。求交层通过它查 alpha 纹理。
    library: Library<S>,
    /// 材质表，按 `InstanceId` 索引。与 `accelerator` 内的实例顺序一致。
    materials: Vec<Material<S>>,
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

    fn library(&self) -> &Library<S> {
        &self.library
    }

    fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<SurfaceHit> {
        let hit = self.accelerator.test(ray, t_range, &self.library)?;
        match hit {
            Hit::Instance(InstanceHit::Shape(instance_id, shape_hit)) => {
                // prim_id 对解析形状恒为 0；mesh 落地后从 InstanceHit 取真实索引。
                Some(shape_hit.into_surface_hit_prim(instance_id))
            }
        }
    }

    fn intersect_p(&self, ray: &Ray, t_range: Range<f32>) -> bool {
        self.accelerator.occluded(ray, t_range, &self.library)
    }

    fn material(&self, hit: &SurfaceHit) -> Option<&Material<S>> {
        let idx = hit.instance_id.as_u64() as usize;
        self.materials.get(idx)
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
    primitives: Vec<Primitive>,
    library: Library<S>,
    materials: Vec<Material<S>>,
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
            primitives: Vec::new(),
            library: Library::new(),
            materials: Vec::new(),
            lights: Vec::new(),
            light_sampler: None,
            infinite_light: None,
            explicit_bounds: None,
        }
    }

    /// 访问内部 `Library<S>`，用于注册纹理等资源。
    ///
    /// 调用方在此注册 alpha 纹理，得到 `TextureId`，再传给
    /// [`Self::add_shape_with_alpha`]。
    pub fn library_mut(&mut self) -> &mut Library<S> {
        &mut self.library
    }

    /// 用外部已构建的 `Library<S>` 替换内部表。
    ///
    /// 场景构建与资源加载分离时使用——先构建 `Library<S>`，再交给
    /// builder。
    pub fn with_library(mut self, library: Library<S>) -> Self {
        self.library = library;
        self
    }

    /// 添加解析形状实例（静态变换，不透明）。`InstanceId` 与材质表索引一致。
    pub fn add_shape(&mut self, shape: AnyShapeA, material: Material<S>) -> InstanceId {
        self.add_shape_inner(shape, material, None)
    }

    /// 添加带 alpha cutout 的解析形状实例。
    pub fn add_shape_with_alpha(
        &mut self,
        shape: AnyShapeA,
        material: Material<S>,
        alpha: TextureId,
    ) -> InstanceId {
        self.add_shape_inner(shape, material, Some(alpha))
    }

    fn add_shape_inner(
        &mut self,
        shape: AnyShapeA,
        material: Material<S>,
        alpha: Option<TextureId>,
    ) -> InstanceId {
        let id = InstanceId::from(self.primitives.len() as u64);
        let mut instance = ShapeInstance::new(id, shape, Affine3A::IDENTITY);
        if let Some(a) = alpha {
            instance = instance.with_alpha(a);
        }
        self.primitives
            .push(Primitive::Instance { shape: instance });
        self.materials.push(material);
        id
    }

    pub fn add_sphere(&mut self, sphere: Sphere, material: Material<S>) -> InstanceId {
        self.add_shape(AnyShapeA::Sphere(sphere.into()), material)
    }

    pub fn add_plane(&mut self, plane: Plane, material: Material<S>) -> InstanceId {
        self.add_shape(AnyShapeA::Plane(plane.into()), material)
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
            compute_bounds(&self.primitives)
        };
        let accelerator = BruteForce::new(self.primitives);
        SimpleScene {
            bounds,
            accelerator,
            library: self.library,
            materials: self.materials,
            lights: self.lights,
            light_sampler: self.light_sampler,
            infinite_light: self.infinite_light,
        }
    }
}

impl<S, LS, L, IL> Default for SimpleSceneBuilder<S, LS, L, IL>
where
    S: SpectralAlgebra,
    LS: LightSampler,
    L: Light<S>,
    IL: InfiniteLight<S>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// 计算场景整体 AABB。
///
/// 任一图元无界（无限平面）时返回 [`Aabb::empty`]，表示"场景没有有限
/// 包围盒"。上层用 `is_empty()` 判断。
fn compute_bounds(primitives: &[Primitive]) -> Aabb {
    use crate::accel::Bounded;
    let mut acc = Aabb::empty();
    for p in primitives {
        let b = p.bounds();
        if !(b.min().is_finite() && b.max().is_finite()) {
            return Aabb::empty();
        }
        acc = acc.union(&b);
    }
    acc
}
