//! kzray-impl/src/accel/shape_area_light.rs
//! 形状灯光, AreaLight

use std::ops::Range;

use glam::Vec3A;
use kzray_abs::{AreaLightBackface, Library, SpectralAlgebra};
use kzray_data::{Aabb, LightId};

use crate::{
    accel::{
        instance::InstancePrimitive, AnimatedShapeInstance, Bounded, InstanceHit, ShapeInstance,
    },
    geometry::RayA,
};

pub struct ShapeAreaLight {
    pub instance: ShapeInstance,
    pub light_id: LightId,
    pub backface: AreaLightBackface,
}

pub struct AnimatedShapeAreaLight {
    pub instance: AnimatedShapeInstance,
    pub light_id: LightId,
    pub backface: AreaLightBackface,
}

// ---------------------------------------------------------------------------
// 公共逻辑抽成自由函数。
// ShapeInstance / AnimatedShapeInstance 都实现了 InstancePrimitive，
// 所以这里用 `&impl InstancePrimitive` 接收，S 由传入的 lib 推断。
// ---------------------------------------------------------------------------
fn area_light_intersect<S: SpectralAlgebra>(
    instance: &impl InstancePrimitive,
    light_id: LightId,
    backface: AreaLightBackface,
    ray: &RayA,
    t_range: Range<f32>,
    lib: &Library<S>,
) -> Option<InstanceHit> {
    let InstanceHit::Shape(id, hit) = instance.intersect(ray, t_range, lib)? else {
        unreachable!("ShapeInstance 只返回 Shape 变体")
    };

    // 正面判定：ray.dir 与几何法线反向 = 从正面打。
    // 注意用 geometric_normal（面朝向），不用 shading_normal。
    let facing = ray.dir.dot(Vec3A::from(hit.geometric_normal)) < 0.0;

    if facing {
        return Some(InstanceHit::AreaLight(id, light_id, hit));
    }
    match backface {
        AreaLightBackface::Transparent => None,
        AreaLightBackface::TwoSided => Some(InstanceHit::AreaLight(id, light_id, hit)),
        AreaLightBackface::Opaque => Some(InstanceHit::Shape(id, hit)),
    }
}

fn area_light_occluded<S: SpectralAlgebra>(
    instance: &impl InstancePrimitive,
    backface: AreaLightBackface,
    ray: &RayA,
    t_range: Range<f32>,
    lib: &Library<S>,
) -> bool {
    // 阴影射线：先做普通遮挡判定，再看是否背面命中且配置为 Transparent。
    // 正面命中总是遮挡；背面命中取决于配置。
    let Some(InstanceHit::Shape(_, hit)) = instance.intersect(ray, t_range, lib) else {
        return false;
    };
    let facing = ray.dir.dot(Vec3A::from(hit.geometric_normal)) < 0.0;
    match (facing, backface) {
        (true, _) => true,
        (false, AreaLightBackface::Transparent) => false,
        (false, _) => true,
    }
}

// ---------------------------------------------------------------------------
// 两个包装类型只做转发。
// ---------------------------------------------------------------------------

impl InstancePrimitive for ShapeAreaLight {
    fn intersect<S: SpectralAlgebra>(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> Option<InstanceHit> {
        area_light_intersect(
            &self.instance,
            self.light_id,
            self.backface,
            ray,
            t_range,
            lib,
        )
    }

    fn occluded<S: SpectralAlgebra>(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> bool {
        area_light_occluded(&self.instance, self.backface, ray, t_range, lib)
    }
}

impl InstancePrimitive for AnimatedShapeAreaLight {
    fn intersect<S: SpectralAlgebra>(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> Option<InstanceHit> {
        area_light_intersect(
            &self.instance,
            self.light_id,
            self.backface,
            ray,
            t_range,
            lib,
        )
    }

    fn occluded<S: SpectralAlgebra>(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> bool {
        area_light_occluded(&self.instance, self.backface, ray, t_range, lib)
    }
}

impl Bounded for ShapeAreaLight {
    fn bounds(&self) -> Aabb {
        self.instance.bounds()
    }
}

impl Bounded for AnimatedShapeAreaLight {
    fn bounds(&self) -> Aabb {
        self.instance.bounds()
    }
}
