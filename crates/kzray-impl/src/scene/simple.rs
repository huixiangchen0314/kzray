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
use kzray_data::common::{InstanceId, LightId};
use kzray_data::geometry::Aabb;
use kzray_data::TextureId;

use crate::accel::ShapeInstance;
use crate::accel::{Accelerator, BruteForce, InstanceHit, Primitive, PrimitiveHit};
use crate::geometry::{AnyShapeA, Plane, Sphere};

// ═══════════════════════════════════════════════════════════════
// SimpleScene
// ═══════════════════════════════════════════════════════════════

/// 简单场景。`DL` / `IL` / `AL` 三个光源泛型独立。
///
/// # id 由外部指定
///
/// `InstanceId` / `LightId` 都从外部传入（分布式场景由全局分配器
/// 保证唯一性）。`SimpleSceneBuilder` 只做冲突检测，不做本地分配。
/// `primitives` / `materials` / `delta_lights` / `area_lights` 都是
/// `BTreeMap<Id, _>`——支持任意 id 值，不要求连续。
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
    materials: BTreeMap<InstanceId, Material<S>>,
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
            PrimitiveHit::Instance(InstanceHit::AreaLight(instance_id, light_id, shape_hit)) => {
                Some(Hit::AreaLight(AreaLightHit {
                    hit: shape_hit.into_surface_hit_prim(instance_id),
                    light_id,
                }))
            }
        }
    }

    fn intersect_p(&self, ray: &Ray, t_range: Range<f32>) -> bool {
        self.accelerator.occluded(ray, t_range, &self.library)
    }

    fn material(&self, hit: &SurfaceHit) -> Option<&Material<S>> {
        self.materials.get(&hit.instance_id)
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
    primitives: BTreeMap<InstanceId, Primitive>,
    library: Library<S>,
    materials: BTreeMap<InstanceId, Material<S>>,
    delta_lights: BTreeMap<LightId, DL>,
    area_lights: BTreeMap<LightId, AL>,
    infinite_lights: Vec<IL>,
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
            primitives: BTreeMap::new(),
            library: Library::new(),
            materials: BTreeMap::new(),
            delta_lights: BTreeMap::new(),
            area_lights: BTreeMap::new(),
            infinite_lights: Vec::new(),
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
    //
    // `InstanceId` 由调用方指定。builder 只做冲突检测。

    /// 添加任意 `Primitive`。`InstanceId` 从 `primitive.instance_id()`
    /// 取——各变体内部自带 id，用户构造 `Primitive` 时已指定。
    ///
    /// 这是几何注册的**底层入口**。`add_shape` / `add_sphere` /
    /// `add_plane` 是便利包装，内部调 `add_primitive`。
    ///
    /// 面积光要可见（参与求交），用户手动构造
    /// `Primitive::ShapeAreaLight(accel::ShapeAreaLight { ... })` 后调
    /// 本方法——几何视图与 light 视图通过同一 `LightId` 关联。
     pub fn add_primitive(&mut self, primitive: Primitive, material: Material<S>) {
        let id = primitive.instance_id();
        let old = self.primitives.insert(id, primitive);
        assert!(
            old.is_none(),
            "duplicate InstanceId {id:?}: already exists in SimpleSceneBuilder"
        );
        let old = self.materials.insert(id, material);
        debug_assert!(old.is_none(), "materials desync from primitives");
    }

    /// 添加解析形状几何（静态变换，无 alpha）。便利方法。
    pub fn add_shape(
        &mut self,
        id: InstanceId,
        shape: AnyShapeA,
        material: Material<S>,
    ) {
        self.add_shape_inner(id, shape, material, None);
    }

    /// 添加带 alpha cutout 的解析形状几何。便利方法。
    pub fn add_shape_with_alpha(
        &mut self,
        id: InstanceId,
        shape: AnyShapeA,
        material: Material<S>,
        alpha: TextureId,
    ) {
        self.add_shape_inner(id, shape, material, Some(alpha));
    }

    fn add_shape_inner(
        &mut self,
        id: InstanceId,
        shape: AnyShapeA,
        material: Material<S>,
        alpha: Option<TextureId>,
    ) {
        let mut instance = ShapeInstance::new(id, shape, Affine3A::IDENTITY);
        if let Some(a) = alpha {
            instance = instance.with_alpha(a);
        }
        self.add_primitive(Primitive::Shape(instance), material);
    }

    /// 添加球体几何。便利方法。
    pub fn add_sphere(&mut self, id: InstanceId, sphere: Sphere, material: Material<S>) {
        self.add_shape(id, AnyShapeA::Sphere(sphere.into()), material);
    }

    /// 添加无限平面几何。便利方法。
    pub fn add_plane(&mut self, id: InstanceId, plane: Plane, material: Material<S>) {
        self.add_shape(id, AnyShapeA::Plane(plane.into()), material);
    }


    // ── 光源 ────────────────────────────────────────────────
    //
    // `LightId` 由调用方指定。builder 只做冲突检测。

    /// 添加 delta 光源。`id` 由调用方指定，已存在则 panic。
    pub fn add_delta_light(&mut self, id: LightId, light: DL) {
        let old = self.delta_lights.insert(id, light);
        assert!(
            old.is_none(),
            "duplicate LightId {id:?}: already exists in delta_lights"
        );
    }

    /// 添加面积光。`id` 由调用方指定，已存在则 panic。
    ///
    /// `AL` 是泛型——用户构造任意实现 `AreaLight<S>` 的类型传入。
    pub fn add_area_light(&mut self, id: LightId, light: AL) {
        let old = self.area_lights.insert(id, light);
        assert!(
            old.is_none(),
            "duplicate LightId {id:?}: already exists in area_lights"
        );
    }

    /// 添加无限远光源。**无 id**——无限远光源是全局的，逃逸光线遍历
    /// 求和。
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
            // `compute_bounds` 借 `&self.primitives`，借用结束于函数返回。
            compute_bounds(&self.primitives)
        };

        // `into_values()` 消耗 `self.primitives`，把所有权转给 `BruteForce`。
        // 部分 move 后，`self.materials` / `self.delta_lights` / ... 仍可
        // 逐字段 move 到新 struct。
        let accelerator = BruteForce::new(self.primitives.into_values().collect());

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

fn compute_bounds(primitives: &BTreeMap<InstanceId, Primitive>) -> Aabb {
    use crate::accel::Bounded;
    let mut acc = Aabb::empty();
    for p in primitives.values() {
        let b = p.bounds();
        if !(b.min().is_finite() && b.max().is_finite()) {
            return Aabb::empty();
        }
        acc = acc.union(&b);
    }
    acc
}
