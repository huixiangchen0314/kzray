//! kzray-impl/src/accel/instance.rs
use std::ops::Range;

use kzray_abs::{Library, SpectralAlgebra};
use kzray_data::{InstanceId, LightId};

use crate::geometry::{RayA, ShapeHit};



pub enum InstanceHit {
    Shape(InstanceId, ShapeHit),
    AreaLight(InstanceId, LightId, ShapeHit),
}

impl InstanceHit {
    #[inline]
    pub fn t(&self) -> f32 {
        match self {
            Self::Shape(_, hit) => hit.t,
            Self::AreaLight(_,_ ,hit )=> hit.t,
        }
    }

    #[inline]
    pub fn instance_id(&self) -> InstanceId {
        match self {
            Self::Shape(id, _) => *id,
            Self::AreaLight(id,_ ,_ ) =>*id,
        }
    }

    /// 面积光命中返回 `Some(light_id)`；普通形状命中返回 `None`。
    /// 积分器用它决定是否求 emission 及是否参与 MIS。
    #[inline]
    pub fn light_ref(&self) -> Option<LightId> {
        match self {
            Self::Shape(_, _) => None,
            Self::AreaLight(_, id, _) => Some(*id),
        }
    }

    /// 提取 `ShapeHit`（两种变体都有）。
    #[inline]
    pub fn shape_hit(&self) -> &ShapeHit {
        match self {
            Self::Shape(_, h) | Self::AreaLight(_, _, h) => h,
        }
    }

}

pub trait InstancePrimitive {
    fn intersect<S: SpectralAlgebra>(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> Option<InstanceHit>;

    fn occluded<S: SpectralAlgebra>(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> bool;
}
