//! kzray-impl/src/accel/accelerator.rs

// TODO 并行加速时候，引入 Arc
use std::ops::Range;

use kzray_abs::{Ray, SpectralAlgebra, Library};
use kzray_data::Aabb;

use crate::{
    accel::{
        InstanceHit,
        animated_shape::AnimatedShapeInstance,
        instance::InstancePrimitive,
        shape::ShapeInstance,
    }, geometry::RayA,
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


pub enum Primitive {
    Shape(ShapeInstance),
    AnimatedShape(AnimatedShapeInstance),
    // MeshTriangle(MeshTriangle),
}

/// 加速精确相交结果
pub enum PrimitiveHit {
    /// 实例相交结果
    Instance(InstanceHit),
    // 网格三角形相交结果
    // MeshTriangleHit,
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
        };
        Some(PrimitiveHit::Instance(instance_hit))
    }

    /// 遮挡查询（世界空间，含 alpha / 动画）。
    pub fn occluded<S: SpectralAlgebra>(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> bool {
        match self {
            Self::Shape(shape) => shape.occluded(ray, t_range, lib),
            Self::AnimatedShape(shape) => shape.occluded(ray, t_range, lib),
        }
    }
}

impl Bounded for Primitive {
    #[inline]
    fn bounds(&self) -> Aabb {
        match self {
            Self::Shape( shape ) => shape.bounds(),
            Self::AnimatedShape(shape)=> shape.bounds(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Hit
// ═══════════════════════════════════════════════════════════════


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
