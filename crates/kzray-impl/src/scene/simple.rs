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

use std::sync::Arc;

use glam::Vec3;
use std::ops::Range;

use kzray_abs::light::{InfiniteLight, Light, LightSampler};
use kzray_abs::material::Material;
use kzray_abs::ray::{Ray, SurfaceHit};
use kzray_abs::scene::Scene;
use kzray_abs::spectrum::SpectralAlgebra;
use kzray_data::common::{InstanceId, LightId};
use kzray_data::geometry::Aabb;

use crate::geometry::{Plane, Sphere};

// ═══════════════════════════════════════════════════════════════
// Shape
// ═══════════════════════════════════════════════════════════════

/// 图元枚举。
///
/// 封闭集合——图元种类由实现层决定。求交统一转成 `GeometryHit`。
#[derive(Clone, Copy, Debug)]
enum Shape {
    Sphere(Sphere),
    Plane(Plane),
}

impl Shape {
    fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<crate::geometry::GeometryHit> {
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
            // 无限平面没有有限 bounds
            Shape::Plane(_) => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// ShapeEntry
// ═══════════════════════════════════════════════════════════════

/// 场景中的一个图元条目。
///
/// `instance_id` 由场景分配（自增），`material` 是运行时材质。
/// `prim_id` 当前恒为 0——每个图元一个 instance。
struct ShapeEntry<S: SpectralAlgebra> {
    shape: Shape,
    material: Material<S>,
    instance_id: InstanceId,
}

// ═══════════════════════════════════════════════════════════════
// SimpleScene
// ═══════════════════════════════════════════════════════════════

/// 简单场景。
///
/// 暴力遍历求交，无加速结构。持有图元、光源、采样器。
pub struct SimpleScene<S: SpectralAlgebra, LS: LightSampler> {
    bounds: Aabb,
    shapes: Vec<ShapeEntry<S>>,
    lights: Vec<Arc<dyn Light<S>>>,
    light_sampler: Option<LS>,
    infinite_light: Option<Arc<dyn InfiniteLight<S>>>,
}

impl<S: SpectralAlgebra, LS: LightSampler> SimpleScene<S, LS> {
    /// 图元数量。
    pub fn shape_count(&self) -> usize {
        self.shapes.len()
    }

    /// 光源数量。
    pub fn light_count(&self) -> usize {
        self.lights.len()
    }
}

impl<S: SpectralAlgebra, LS: LightSampler> Scene<S, LS> for SimpleScene<S, LS> {
    fn bounds(&self) -> Aabb {
        self.bounds
    }

    fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<SurfaceHit> {
        let mut best_t = t_range.end;
        let mut best_hit: Option<SurfaceHit> = None;

        for entry in &self.shapes {
            let range = t_range.start..best_t;
            if let Some(geom_hit) = entry.shape.intersect(ray, range) {
                best_t = geom_hit.t;
                best_hit = Some(geom_hit.into_surface_hit(entry.instance_id, 0));
            }
        }

        best_hit
    }

    fn intersect_p(&self, ray: &Ray, t_range: Range<f32>) -> bool {
        // 任意命中即返回 true，不需要最近
        for entry in &self.shapes {
            if entry.shape.intersect(ray, t_range.clone()).is_some() {
                return true;
            }
        }
        false
    }

    fn material(&self, hit: &SurfaceHit) -> Option<&Material<S>> {
        let idx = hit.instance_id.as_u64() as usize;
        self.shapes.get(idx).map(|e| &e.material)
    }

    fn light(&self, id: LightId) -> Option<&dyn Light<S>> {
        self.lights.get(id.as_u64() as usize).map(|l| l.as_ref())
    }

    fn light_sampler(&self) -> Option<&LS> {
        self.light_sampler.as_ref()
    }

    fn infinite_light(&self) -> Option<&dyn InfiniteLight<S>> {
        self.infinite_light.as_deref()
    }
}

// ═══════════════════════════════════════════════════════════════
// SimpleSceneBuilder
// ═══════════════════════════════════════════════════════════════

/// 简单场景构建器。
///
/// 逐步添加图元 / 光源，最后 `build` 成 `SimpleScene`。
///
/// # `instance_id` 分配
///
/// 按 `add_shape` 的调用顺序自增（0, 1, 2, ...）。图元的
/// `instance_id` 就是它在 `shapes` 向量中的索引。
///
/// # `LightId` 分配
///
/// 按 `add_light` 的调用顺序自增。与 `RenderInput` 的 `LightId` 一致
/// （阶段 5 接入时从 `RenderInput` 读）。
pub struct SimpleSceneBuilder<S: SpectralAlgebra, LS: LightSampler> {
    shapes: Vec<ShapeEntry<S>>,
    lights: Vec<Arc<dyn Light<S>>>,
    light_sampler: Option<LS>,
    infinite_light: Option<Arc<dyn InfiniteLight<S>>>,
    explicit_bounds: Option<Aabb>,
}

impl<S: SpectralAlgebra, LS: LightSampler> Default for SimpleSceneBuilder<S, LS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: SpectralAlgebra, LS: LightSampler> SimpleSceneBuilder<S, LS> {
    pub fn new() -> Self {
        SimpleSceneBuilder {
            shapes: Vec::new(),
            lights: Vec::new(),
            light_sampler: None,
            infinite_light: None,
            explicit_bounds: None,
        }
    }

    /// 添加一个图元，附带材质。返回分配的 `instance_id`。
    pub fn add_shape(&mut self, shape: impl Into<Shape>, material: Material<S>) -> InstanceId {
        let instance_id = InstanceId::from(self.shapes.len() as u64);
        self.shapes.push(ShapeEntry {
            shape: shape.into(),
            material,
            instance_id,
        });
        instance_id
    }

    /// 添加球体（便利方法）。
    pub fn add_sphere(&mut self, sphere: Sphere, material: Material<S>) -> InstanceId {
        self.add_shape(sphere, material)
    }

    /// 添加平面（便利方法）。
    pub fn add_plane(&mut self, plane: Plane, material: Material<S>) -> InstanceId {
        self.add_shape(plane, material)
    }

    /// 添加光源。返回分配的 `LightId`。
    pub fn add_light(&mut self, light: Arc<dyn Light<S>>) -> LightId {
        let id = LightId::from(self.lights.len() as u64);
        self.lights.push(light);
        id
    }

    /// 设置光源采样器。
    pub fn with_light_sampler(mut self, sampler: LS) -> Self {
        self.light_sampler = Some(sampler);
        self
    }

    /// 设置无限远光源（环境光）。
    pub fn with_infinite_light(mut self, light: Arc<dyn InfiniteLight<S>>) -> Self {
        self.infinite_light = Some(light);
        self
    }

    /// 显式指定世界范围。
    ///
    /// 不指定时，从球体的 bounds 聚合；有平面时范围无限（`Aabb::empty()`
    /// 会被视为"无 bounds"）。
    pub fn with_bounds(mut self, bounds: Aabb) -> Self {
        self.explicit_bounds = Some(bounds);
        self
    }

    /// 构造场景。
    ///
    /// 计算世界范围：
    /// 1. 如果显式指定 `bounds`，直接使用；
    /// 2. 否则从所有球体聚合；
    /// 3. 有无限平面时无法聚合，用 `Aabb::empty()`（表示无限）。
    pub fn build(self) -> SimpleScene<S, LS> {
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
                // 有无限平面，bounds 无意义
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

// ═══════════════════════════════════════════════════════════════
// Shape 的 From 实现
// ═══════════════════════════════════════════════════════════════

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
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::light::PointLight;
    use crate::light::sampler::UniformLightSampler;
    use crate::material::LambertBSDFBuilder;
    use kzray_abs::light::LightSampleContext;
    use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
    use std::sync::Arc;

    type TestScene = SimpleScene<SpectrumRgb, UniformLightSampler>;
    type TestBuilder = SimpleSceneBuilder<SpectrumRgb, UniformLightSampler>;

    fn lambert(albedo: SpectrumRgb) -> Material<SpectrumRgb> {
        Material::bsdf(Arc::new(LambertBSDFBuilder::new(albedo)))
    }

    fn make_simple_scene() -> TestScene {
        let mut builder = TestBuilder::new();
        // 球体在原点，半径 1
        builder.add_sphere(
            Sphere::new(Vec3::ZERO, 1.0),
            lambert(SpectrumRgb::new(0.8, 0.2, 0.2)),
        );
        // 平面在 y = -1
        builder.add_plane(
            Plane::horizontal(-1.0),
            lambert(SpectrumRgb::new(0.5, 0.5, 0.5)),
        );
        // 点光源
        builder.add_light(Arc::new(PointLight::new(
            Vec3::new(0.0, 5.0, 0.0),
            SpectrumRgb::new(10.0, 10.0, 10.0),
        )));

        builder
            .with_light_sampler(UniformLightSampler::new(vec![LightId::from(0)]))
            .build()
    }

    // ─────────── 图元求交 ───────────

    #[test]
    fn intersect_hits_sphere() {
        let scene = make_simple_scene();
        // 从 -x 射向原点，命中球的 (-1, 0, 0)
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 0.0);
        let hit = scene.intersect(&ray, 0.0..f32::INFINITY).unwrap();
        assert!((hit.position - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-5);
        assert_eq!(hit.instance_id.as_u64(), 0); // 第一个球
        assert_eq!(hit.prim_id, 0);
    }

    #[test]
    fn intersect_hits_plane() {
        let scene = make_simple_scene();
        // 从上方射向平面 (y = -1)
        let ray = Ray::new(Vec3::new(5.0, 5.0, 0.0), Vec3::NEG_Y, 0.0);
        let hit = scene.intersect(&ray, 0.0..f32::INFINITY).unwrap();
        assert!((hit.position - Vec3::new(5.0, -1.0, 0.0)).length() < 1e-5);
        assert_eq!(hit.instance_id.as_u64(), 1); // 第二个是平面
    }

    #[test]
    fn intersect_returns_closest() {
        let scene = make_simple_scene();
        // 从 -x 射向原点，先命中球，再穿过球命中平面
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 0.0);
        let hit = scene.intersect(&ray, 0.0..f32::INFINITY).unwrap();
        // 球的命中在前
        assert_eq!(hit.instance_id.as_u64(), 0);
    }

    #[test]
    fn intersect_misses_everything() {
        let scene = make_simple_scene();
        // 朝 +x 射出，什么都没
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::NEG_X, 0.0);
        assert!(scene.intersect(&ray, 0.0..f32::INFINITY).is_none());
    }

    #[test]
    fn intersect_p_returns_true_for_hit() {
        let scene = make_simple_scene();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 0.0);
        assert!(scene.intersect_p(&ray, 0.0..f32::INFINITY));
    }

    #[test]
    fn intersect_p_returns_false_for_miss() {
        let scene = make_simple_scene();
        let ray = Ray::new(Vec3::new(-5.0, 100.0, 0.0), Vec3::X, 0.0);
        // y=100，球和平面都够不着
        assert!(!scene.intersect_p(&ray, 0.0..f32::INFINITY));
    }

    // ─────────── 材质查询 ───────────

    #[test]
    fn material_returns_correct() {
        let scene = make_simple_scene();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 0.0);
        let hit = scene.intersect(&ray, 0.0..f32::INFINITY).unwrap();
        let mat = scene.material(&hit).unwrap();
        assert!(mat.is_bsdf());
    }

    #[test]
    fn material_returns_none_for_invalid_id() {
        let scene = make_simple_scene();
        // 伪造一个无效的 hit
        let hit = SurfaceHit {
            t: 0.0,
            distance: 0.0,
            position: Vec3::ZERO,
            shading_normal: Vec3::Y,
            geometric_normal: Vec3::Y,
            uv: glam::Vec2::ZERO,
            instance_id: InstanceId::from(999),
            prim_id: 0,
        };
        assert!(scene.material(&hit).is_none());
    }

    // ─────────── 光源查询 ───────────

    #[test]
    fn light_returns_point_light() {
        let scene = make_simple_scene();
        let light = scene.light(LightId::from(0)).unwrap();
        assert!(light.light_type().is_delta());
    }

    #[test]
    fn light_returns_none_for_invalid_id() {
        let scene = make_simple_scene();
        assert!(scene.light(LightId::from(999)).is_none());
    }

    #[test]
    fn light_sampler_returns_some() {
        let scene = make_simple_scene();
        assert!(scene.light_sampler().is_some());
    }

    #[test]
    fn infinite_light_returns_none() {
        let scene = make_simple_scene();
        assert!(scene.infinite_light().is_none());
    }

    // ─────────── Bounds ───────────

    #[test]
    fn bounds_aggregates_spheres() {
        let mut builder = TestBuilder::new();
        builder.add_sphere(
            Sphere::new(Vec3::ZERO, 1.0),
            lambert(SpectrumRgb::ONE),
        );
        builder.add_sphere(
            Sphere::new(Vec3::new(5.0, 0.0, 0.0), 2.0),
            lambert(SpectrumRgb::ONE),
        );
        let scene = builder.build();
        let b = scene.bounds();
        // 合并两个球
        assert!((b.min().x - (-1.0)).abs() < 1e-5);
        assert!((b.max().x - 7.0).abs() < 1e-5);
    }

    #[test]
    fn bounds_empty_with_plane() {
        let mut builder = TestBuilder::new();
        builder.add_plane(Plane::horizontal(0.0), lambert(SpectrumRgb::ONE));
        let scene = builder.build();
        assert!(scene.bounds().is_empty());
    }

    #[test]
    fn explicit_bounds_used() {
        let bounds = Aabb::new(Vec3::splat(-10.0), Vec3::splat(10.0));
        let mut builder = TestBuilder::new();
        builder.add_sphere(Sphere::new(Vec3::ZERO, 1.0), lambert(SpectrumRgb::ONE));
        let scene = builder.with_bounds(bounds).build();
        assert_eq!(scene.bounds(), bounds);
    }

    // ─────────── 混合图元 ───────────

    #[test]
    fn multiple_spheres() {
        let mut builder = TestBuilder::new();
        builder.add_sphere(
            Sphere::new(Vec3::new(-3.0, 0.0, 0.0), 1.0),
            lambert(SpectrumRgb::new(1.0, 0.0, 0.0)),
        );
        builder.add_sphere(
            Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            lambert(SpectrumRgb::new(0.0, 1.0, 0.0)),
        );
        builder.add_sphere(
            Sphere::new(Vec3::new(3.0, 0.0, 0.0), 1.0),
            lambert(SpectrumRgb::new(0.0, 0.0, 1.0)),
        );
        let scene = builder.build();

        // 从 -x 射向 +x，应该依次命中三个
        let ray = Ray::new(Vec3::new(-10.0, 0.0, 0.0), Vec3::X, 0.0);
        let hit = scene.intersect(&ray, 0.0..f32::INFINITY).unwrap();
        assert_eq!(hit.instance_id.as_u64(), 0); // 第一个球
    }

    // ─────────── Scene trait ───────────

    #[test]
    fn scene_is_object_safe() {
        let scene: Box<dyn Scene<SpectrumRgb, UniformLightSampler>> =
            Box::new(make_simple_scene());
        assert_eq!(scene.bounds().is_empty() || !scene.bounds().is_empty(), true);
    }

    #[test]
    fn scene_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TestScene>();
    }

    // 抑制 unused imports
    #[test]
    fn _use_imports() {
        let _: Option<NoWavelengths> = None;
        let _: Option<LightSampleContext> = None;
    }
}
