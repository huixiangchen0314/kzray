//! kzray-impl/src/geometry/shape_hit.rs

use std::ops::Range;

use glam::{Vec2, Vec3, Vec3A};
use kzray_abs::SurfaceHit;
use kzray_data::InstanceId;

use crate::geometry::RayA;

pub trait ShapeA {
    fn intersect(&self, ray: &RayA, t_range: Range<f32>) -> Option<ShapeHit>;

    /// 遮挡查询：`t_range` 内**任意**交点存在即返回 `true`。
    ///
    /// 与 [`ShapeA::intersect`] 的区别：
    ///
    /// - 只回答"有没有"，不求最近、不构造 [`ShapeHit`]；
    /// - 短路形状（如球判别式通过、三角形重心坐标通过）立即返回，
    ///   不算法线、位置、UV、`distance`；
    /// - 区间语义与 `intersect` 一致：`t >= t_range.start && t < t_range.end`。
    ///
    /// # 默认实现
    ///
    /// 基于 [`ShapeA::intersect`]，**正确但慢**。热点形状必须覆写为
    /// 短路版本。默认实现存在只为保证新形状可渐进实现——先写
    /// `intersect` 跑通，再补 `occluded` 优化。
    #[inline]
    fn occluded(&self, ray: &RayA, t_range: Range<f32>) -> bool {
        self.intersect(ray, t_range).is_some()
    }

    /// 在表面均匀采样一点。`u ∈ [0,1)²`。
    ///
    /// 返回几何**所在空间**的采样点。对面积光，几何已在世界空间，
    /// 返回值即世界空间。无限平面无面积，返回 `None`。
    fn sample_area(&self, u: Vec2) -> Option<AreaSample>;


}

/// 图元求交的纯几何结果。
///
/// 不含 `instance_id` / `prim_id`——这两个字段由场景层在求交后填充，
/// 转成 [`SurfaceHit`]。
#[derive(Clone, Copy, Debug)]
pub struct ShapeHit {
    /// 光线参数。击中点 = `ray.origin + t × ray.dir`。
    pub t: f32,
    /// 从光线起点到命中点的实际距离，等于 `t × |ray.dir|`。
    pub distance: f32,
    /// 击中点世界坐标。
    pub position: Vec3,

    // 着色法向不该由几何查询出来
    // 着色法线（当前与几何法线相同）。
    // pub shading_normal: Vec3,
    /// 几何法线（面法线）。
    pub geometric_normal: Vec3,
    /// 表面 UV。
    pub uv: Vec2,
}

impl ShapeHit {
    /// 构造为场景层表面求交
    pub fn into_surface_hit(self, instance_id: InstanceId, prim_id: u32) -> SurfaceHit {
        SurfaceHit {
            t: self.t,
            distance: self.distance,
            position: self.position,
            shading_normal: self.geometric_normal,
            geometric_normal: self.geometric_normal,
            uv: self.uv,
            instance_id,
            prim_id,
        }
    }

    /// 解析图元
    pub fn into_surface_hit_prim(self, instance_id: InstanceId) -> SurfaceHit {
        SurfaceHit {
            t: self.t,
            distance: self.distance,
            position: self.position,
            shading_normal: self.geometric_normal,
            geometric_normal: self.geometric_normal,
            uv: self.uv,
            instance_id,
            prim_id: 0,
        }
    }
}

/// 表面均匀采样结果（**几何所在空间**，即世界空间——调用方传入的
/// 几何已经是世界空间）。
pub struct AreaSample {
    /// 采样点。
    pub position: Vec3A,
    /// 采样点法线（单位向量，朝外）。
    pub normal: Vec3A,

    /// 面积测度
    /// 面积概率密度 `1 / 表面积`。
    ///
    /// 转立体角 pdf 由 `ShapeAreaLight::sample_li` 做：
    /// `pdf_solid = pdf_area * distance² / cos_light`。
    pub pdf_area: f32,
}
