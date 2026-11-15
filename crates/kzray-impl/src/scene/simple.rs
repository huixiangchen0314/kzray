//! kzray-impl/src/scene/simple.rs

use std::collections::BTreeMap;
use std::ops::Range;

use glam::Affine3A;
use kzray_abs::light::{AreaLight, InfiniteLight, Light, LightSampler};
use kzray_abs::material::Material;
use kzray_abs::ray::{AreaLightHit, Hit, Ray, SurfaceHit};
use kzray_abs::scene::Scene;
use kzray_abs::spectrum::SpectralAlgebra;
use kzray_abs::Library;
use kzray_data::TextureId;
use kzray_data::common::{InstanceId, LightId};
use kzray_data::geometry::Aabb;

use crate::accel::ShapeInstance;
use crate::accel::{Accelerator, BruteForce, InstanceHit, Primitive, PrimitiveHit};
use crate::geometry::{AnyShapeA, Plane, Sphere};

// ═══════════════════════════════════════════════════════════════
// SimpleScene
// ═══════════════════════════════════════════════════════════════

/// 简单场景。`DL` / `IL` / `AL` 三个光源泛型完全独立——用户可分别
/// 替换为自定义光源类型。
///
/// # 存储布局
///
/// - `delta_lights: BTreeMap<LightId, DL>`——delta 光源，按 id 查
///   （NEE 采样器选中后按 id 找）；
/// - `area_lights: BTreeMap<LightId, AL>`——面积光，按 id 查（同 NEE）；
/// - `infinite_lights: Vec<IL>`——无限远光源，**无 id**。环境光 / 天空
///   光通常是全局的，不需要被"选中某一个"——所有逃逸光线求辐射亮度时
///   遍历求和。NEE 通常不采样无限远光源（无有限 pdf 测度），逃逸时
///   直接累加即可。
///
/// `LightId` 由 `SimpleSceneBuilder` 全局分配，只用于 delta / 面积光，
/// 两类 id 不重叠。
pub struct SimpleScene<S, LS, DL, IL, AL>
where
    S: SpectralAlgebra,
    LS: LightSampler,
    DL: Light<S>,
    IL: InfiniteLight<S>,
    AL: AreaLight<S>,
{
    bounds: Aabb,
    accelerator: BruteForce,
    library: Library<S>,
    materials: Vec<Material<S>>,
    delta_lights: BTreeMap<LightId, DL>,
    area_lights: BTreeMap<LightId, AL>,
    infinite_lights: Vec<IL>,
    light_sampler: Option<LS>,
}

impl<S, LS, DL, IL, AL> Scene<S, LS, DL, IL, AL> for SimpleScene<S, LS, DL, IL, AL>
where
    S: SpectralAlgebra,
    LS: LightSampler,
    DL: Light<S>,
    IL: InfiniteLight<S>,
    AL: AreaLight<S>,
{
    fn bounds(&self) -> Aabb {
        self.bounds
    }

    fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<Hit> {
        let prim_hit = self.accelerator.test(ray, t_range, &self.library)?;
        match prim_hit {
            PrimitiveHit::Instance(InstanceHit::Shape(instance_id, shape_hit)) => {
                Some(Hit::Surface(shape_hit.into_surface_hit_prim(instance_id)))
            }
            PrimitiveHit::Instance(InstanceHit::AreaLight(
                instance_id,
                light_id,
                shape_hit,
            )) => Some(Hit::AreaLight(AreaLightHit {
                hit: shape_hit.into_surface_hit_prim(instance_id),
                light_id,
            })),
        }
    }

    fn intersect_p(&self, ray: &Ray, t_range: Range<f32>) -> bool {
        self.accelerator.occluded(ray, t_range, &self.library)
    }

    fn material(&self, hit: &SurfaceHit) -> Option<&Material<S>> {
        let idx = hit.instance_id.as_u64() as usize;
        self.materials.get(idx)
    }

    #[inline]
    fn delta_light(&self, id: LightId) -> Option<&DL> {
        self.delta_lights.get(&id)
    }

    #[inline]
    fn area_light(&self, id: LightId) -> Option<&AL> {
        self.area_lights.get(&id)
    }

    #[inline]
    fn infinite_lights(&self) -> &[IL] {
        &self.infinite_lights
    }

    fn light_sampler(&self) -> Option<&LS> {
        self.light_sampler.as_ref()
    }

    fn library(&self) -> &Library<S> {
        &self.library
    }
}

// ═══════════════════════════════════════════════════════════════
// SimpleSceneBuilder
// ═══════════════════════════════════════════════════════════════

pub struct SimpleSceneBuilder<S, LS, DL, IL, AL>
where
    S: SpectralAlgebra,
    LS: LightSampler,
    DL: Light<S>,
    IL: InfiniteLight<S>,
    AL: AreaLight<S>,
{
    primitives: Vec<Primitive>,
    library: Library<S>,
    materials: Vec<Material<S>>,
    delta_lights: BTreeMap<LightId, DL>,
    area_lights: BTreeMap<LightId, AL>,
    infinite_lights: Vec<IL>,
    /// 全局 `LightId` 计数器。delta 和面积光共用，保证 id 不重叠。
    /// 无限远光源不占用 id——它们按位置索引。
    next_light_id: u64,
    light_sampler: Option<LS>,
    explicit_bounds: Option<Aabb>,
}

impl<S, LS, DL, IL, AL> SimpleSceneBuilder<S, LS, DL, IL, AL>
where
    S: SpectralAlgebra,
    LS: LightSampler,
    DL: Light<S>,
    IL: InfiniteLight<S>,
    AL: AreaLight<S>,
{
    pub fn new() -> Self {
        Self {
            primitives: Vec::new(),
            library: Library::new(),
            materials: Vec::new(),
            delta_lights: BTreeMap::new(),
            area_lights: BTreeMap::new(),
            infinite_lights: Vec::new(),
            next_light_id: 0,
            light_sampler: None,
            explicit_bounds: None,
        }
    }

    pub fn library_mut(&mut self) -> &mut Library<S> {
        &mut self.library
    }

    pub fn with_library(mut self, library: Library<S>) -> Self {
        self.library = library;
        self
    }

    // ── 几何 ────────────────────────────────────────────────

    pub fn add_shape(&mut self, shape: AnyShapeA, material: Material<S>) -> InstanceId {
        self.add_shape_inner(shape, material, None)
    }

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
        self.primitives.push(Primitive::Shape(instance));
        self.materials.push(material);
        id
    }

    pub fn add_sphere(&mut self, sphere: Sphere, material: Material<S>) -> InstanceId {
        self.add_shape(AnyShapeA::Sphere(sphere.into()), material)
    }

    pub fn add_plane(&mut self, plane: Plane, material: Material<S>) -> InstanceId {
        self.add_shape(AnyShapeA::Plane(plane.into()), material)
    }

    // ── 光源 ────────────────────────────────────────────────

    /// 添加 delta 光源（点光源、方向光等）。返回全局唯一的 `LightId`。
    pub fn add_delta_light(&mut self, light: DL) -> LightId {
        let id = LightId::from(self.next_light_id);
        self.next_light_id += 1;
        self.delta_lights.insert(id, light);
        id
    }

    /// 添加面积光。返回全局唯一的 `LightId`，与 delta 光源不重叠。
    ///
    /// `AL` 是泛型——用户构造任意实现 `AreaLight<S>` 的类型传入
    /// （标准实现 `AnyAreaLight::ShapeArea(...)`）。
    pub fn add_area_light(&mut self, light: AL) -> LightId {
        let id = LightId::from(self.next_light_id);
        self.next_light_id += 1;
        self.area_lights.insert(id, light);
        id
    }

    /// 添加无限远光源（环境光、天空光等）。**无 id、无返回值**。
    ///
    /// 无限远光源是全局的——逃逸光线累加时遍历所有无限远光源求和。
    /// 通常场景只有一个环境光，但接口允许添加多个（如分层天空光）。
    pub fn add_infinite_light(&mut self, light: IL) {
        self.infinite_lights.push(light);
    }

    // ── 配置 ────────────────────────────────────────────────

    pub fn with_light_sampler(mut self, sampler: LS) -> Self {
        self.light_sampler = Some(sampler);
        self
    }

    pub fn with_bounds(mut self, bounds: Aabb) -> Self {
        self.explicit_bounds = Some(bounds);
        self
    }

    pub fn build(self) -> SimpleScene<S, LS, DL, IL, AL> {
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
            delta_lights: self.delta_lights,
            area_lights: self.area_lights,
            infinite_lights: self.infinite_lights,
            light_sampler: self.light_sampler,
        }
    }
}

impl<S, LS, DL, IL, AL> Default for SimpleSceneBuilder<S, LS, DL, IL, AL>
where
    S: SpectralAlgebra,
    LS: LightSampler,
    DL: Light<S>,
    IL: InfiniteLight<S>,
    AL: AreaLight<S>,
{
    fn default() -> Self {
        Self::new()
    }
}

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
