//! kzray-impl/src/accel/shape_area_light.rs
//! 形状灯光, AreaLight

use std::ops::Range;

use glam::Vec3A;
use kzray_abs::{AreaLightBackface, Library, SpectralAlgebra};
use kzray_data::LightId;

use crate::{accel::{AnimatedShapeInstance, InstanceHit, ShapeInstance, instance::InstancePrimitive}, geometry::RayA};


pub struct ShapeAreaLight {
    pub instance: ShapeInstance,
    pub light_id: LightId,
    pub backface: AreaLightBackface,
}


pub struct AnimatedShapeAreaLight {
    pub instance: AnimatedShapeInstance,
    pub light_ref: LightId,
    pub backface: AreaLightBackface,
}


impl InstancePrimitive for ShapeAreaLight {
    fn intersect<S: SpectralAlgebra>(
        &self, ray: &RayA, t_range: Range<f32>, lib: &Library<S>,
    ) -> Option<InstanceHit> {
        let InstanceHit::Shape(id, hit) =
            self.instance.intersect(ray, t_range, lib)? else {
            unreachable!("ShapeInstance 只返回 Shape 变体")
        };

        // 正面判定：ray.dir 与几何法线反向 = 从正面打。
        // 注意用 geometric_normal（面朝向），不用 shading_normal。
        let facing = ray.dir.dot(Vec3A::from(hit.geometric_normal)) < 0.0;

        if facing {
            return Some(InstanceHit::AreaLight(id, self.light_id, hit));
        }
        match self.backface {
            AreaLightBackface::Transparent => None,
            AreaLightBackface::TwoSided => Some(InstanceHit::AreaLight(id, self.light_id, hit)),
            AreaLightBackface::Opaque => Some(InstanceHit::Shape(id, hit)),
        }
    }

    fn occluded<S: SpectralAlgebra>(
        &self, ray: &RayA, t_range: Range<f32>, lib: &Library<S>,
    ) -> bool {
        // 阴影射线：先做普通遮挡判定，再看是否背面命中且配置为 Transparent。
        // 正面命中总是遮挡；背面命中取决于配置。
        let Some(InstanceHit::Shape(_, hit)) =
            self.instance.intersect(ray, t_range.clone(), lib) else {
            return false;
        };
        let facing = ray.dir.dot(Vec3A::from(hit.geometric_normal)) < 0.0;
        match (facing, self.backface) {
            (true, _) => true,
            (false, AreaLightBackface::Transparent) => false,
            (false, _) => true,
        }
    }
}
