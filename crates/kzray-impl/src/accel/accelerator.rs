//! kzray-impl/src/accel/accelerator.rs

// TODO 并行加速时候，引入 Arc
use std::ops::Range;

use kzray_abs::{Library, Ray, SpectralAlgebra};
use kzray_data::{Aabb, InstanceId};

use crate::{
    accel::{
        animated_shape::AnimatedShapeInstance,
        instance::InstancePrimitive,
        shape::ShapeInstance,
        shape_area_light::{AnimatedShapeAreaLight, ShapeAreaLight},
        InstanceHit,
    },
    geometry::RayA,
};

// ═══════════════════════════════════════════════════════════════
// Bounded
// ═══════════════════════════════════════════════════════════════

pub trait Bounded {
    fn bounds(&self) -> Aabb;
}

// ═══════════════════════════════════════════════════════════════
// Primitive
// ═══════════════════════════════════════════════════════════════

/// 加速图元。BVH 叶子载荷。
///
/// 变体按几何种类分：
///
/// - `Shape` / `AnimatedShape`：普通表面实例，命中返回
///   [`InstanceHit::Shape`]；
/// - `ShapeAreaLight` / `AnimatedShapeAreaLight`：面积光实例，命中
///   返回 [`InstanceHit::AreaLight`]（带 `LightId`，供积分器求 emission）。
///
/// 面积光与普通表面是**不同图元类型**——面积光命中后积分器需要按
/// `LightId` 查 emission 并参与 MIS，普通表面走材质路径。类型区分让
/// 调用方 `match InstanceHit` 时编译器强制处理两种命中。
pub enum Primitive {
    Shape(ShapeInstance),
    AnimatedShape(AnimatedShapeInstance),
    ShapeAreaLight(ShapeAreaLight),
    AnimatedShapeAreaLight(AnimatedShapeAreaLight),
    // MeshTriangle(MeshTriangle),
}

impl Primitive {
    /// 最近命中（世界空间，含 alpha / 动画）。
    ///
    /// `S` 是光谱代数——`Library<S>` 持有光谱纹理（`AlbedoTexture<S>` 等），
    /// 即使本方法内部只用 `lib.alpha(...)`（不依赖 `S`），签名仍需传播
    /// `S` 以接收 `&Library<S>`。类型参数传播零运行时成本。
    pub fn intersect<S: SpectralAlgebra>(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> Option<PrimitiveHit> {
        let instance_hit = match self {
            Self::Shape(shape) => shape.intersect(ray, t_range, lib)?,
            Self::AnimatedShape(shape) => shape.intersect(ray, t_range, lib)?,
            Self::ShapeAreaLight(light) => light.intersect(ray, t_range, lib)?,
            Self::AnimatedShapeAreaLight(light) => light.intersect(ray, t_range, lib)?,
        };
        Some(PrimitiveHit::Instance(instance_hit))
    }

    /// 遮挡查询（世界空间，含 alpha / 动画）。
    ///
    /// 面积光的 `occluded` 需要遵守 `backface` 配置——背面 `Transparent`
    /// 时不遮挡，由 `ShapeAreaLight::occluded` 内部处理。
    pub fn occluded<S: SpectralAlgebra>(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> bool {
        match self {
            Self::Shape(shape) => shape.occluded(ray, t_range, lib),
            Self::AnimatedShape(shape) => shape.occluded(ray, t_range, lib),
            Self::ShapeAreaLight(light) => light.occluded(ray, t_range, lib),
            Self::AnimatedShapeAreaLight(light) => light.occluded(ray, t_range, lib),
        }
    }

    /// 图元对应的 `InstanceId`。
    ///
    /// 各变体内部 `ShapeInstance` / `AnimatedShapeInstance` / `ShapeAreaLight`
    /// 自带 `id`——`add_primitive` 用它作 map key，用户不需重复传 id。
    #[inline]
    pub fn instance_id(&self) -> InstanceId {
        match self {
            Self::Shape(s) => s.id,
            Self::AnimatedShape(s) => s.id,
            Self::ShapeAreaLight(s) => s.instance.id,
            Self::AnimatedShapeAreaLight(s) => s.instance.id,
        }
    }
}

impl Bounded for Primitive {
    #[inline]
    fn bounds(&self) -> Aabb {
        match self {
            Self::Shape(shape) => shape.bounds(),
            Self::AnimatedShape(shape) => shape.bounds(),
            Self::ShapeAreaLight(light) => light.bounds(),
            Self::AnimatedShapeAreaLight(light) => light.bounds(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// PrimitiveHit
// ═══════════════════════════════════════════════════════════════

/// 加速层图元命中。与抽象层 `kzray_abs::ray::Hit` 区分——后者是场景层
/// 统一命中，前者是加速结构返回的图元级结果。
pub enum PrimitiveHit {
    /// 实例相交结果。
    Instance(InstanceHit),
    // 网格三角形相交结果
    // MeshTriangleHit,
}

impl PrimitiveHit {
    /// 命中的仿射参数 `t`。
    #[inline]
    pub fn t(&self) -> f32 {
        match self {
            Self::Instance(h) => h.t(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Accelerator
// ═══════════════════════════════════════════════════════════════

/// 加速结构统一接口。
///
/// 方法泛型化在 `S: SpectralAlgebra` 上——`Library<S>` 持有光谱纹理，
/// 签名需要传播 `S`。`Accelerator` trait 本身不带 `S`，因此 `BruteForce`
/// 等实现是纯几何类型，无 `PhantomData`。
pub trait Accelerator {
    /// 最近命中。
    fn test<S: SpectralAlgebra>(
        &self,
        ray: &Ray,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> Option<PrimitiveHit>;

    /// 遮挡查询：`t_range` 内任意可接受命中存在即 `true`。
    fn occluded<S: SpectralAlgebra>(
        &self,
        ray: &Ray,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> bool;

    /// 所有命中（调试/验证用）。
    fn test_all<S: SpectralAlgebra>(
        &self,
        ray: &Ray,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> Vec<PrimitiveHit>;
}
