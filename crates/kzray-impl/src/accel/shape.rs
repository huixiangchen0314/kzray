//! kzray-impl/src/accel/shape.rs
//! 静态实例

use std::ops::Range;

use glam::{Affine3A, Vec3A};
use kzray_abs::{Library, SpectralAlgebra};
use kzray_data::{Aabb, InstanceId, TextureId};

use crate::{
    accel::{Bounded, InstanceHit, instance::InstancePrimitive}, geometry::{AnyShapeA, RayA, ShapeA, ShapeHit},
};

/// alpha cutout 阈值。`Texture<AlphaChannel>::eval` 低于此值视为透明。
pub const ALPHA_THRESHOLD: f32 = 0.5;



// ═══════════════════════════════════════════════════════════════
// 共享求交逻辑
// ═══════════════════════════════════════════════════════════════

/// 给定已解析到当前时刻的正/逆变换，执行"世界 → 局部 → 求交 → 回世界"。
///
/// `alpha: Option<TextureId>` 按值传——`TextureId` 是 4 字节 `Copy`，
/// 无所有权问题。透明时返回 `None`，BVH 继续搜索其他图元。
#[inline]
pub(crate) fn intersect_with_transform<S: SpectralAlgebra>(
    shape: &AnyShapeA,
    transform: &Affine3A,
    inv_transform: &Affine3A,
    alpha: Option<TextureId>,
    ray: &RayA,
    t_range: Range<f32>,
    lib: &Library<S>,
) -> Option<ShapeHit> {
    let local_origin = inv_transform.transform_point3a(ray.origin);
    let local_dir = inv_transform.transform_vector3a(ray.dir);
    let local_ray = RayA::new(local_origin, local_dir, ray.time);

    let local_hit = shape.intersect(&local_ray, t_range)?;

    // ── alpha 判定：透明则视为未命中 ──
    if let Some(id) = alpha {
        let tex = lib.alpha(id);           // &AlphaTexture
        let a = tex.eval(local_hit.uv, None); // 不需要微分射线
        if a < ALPHA_THRESHOLD {
            return None;
        }
    }

    let t = local_hit.t;
    let position = transform.transform_point3a(local_hit.position.into());
    let geometric_normal =
        (inv_transform.matrix3.transpose() * Vec3A::from(local_hit.geometric_normal)).normalize();
    let distance = t * ray.dir.length();

    Some(ShapeHit {
        t,
        distance,
        position: position.into(),
        geometric_normal: geometric_normal.into(),
        uv: local_hit.uv,
    })
}

#[inline]
pub(crate) fn occluded_with_transform(
    shape: &AnyShapeA,
    inv_transform: &Affine3A,
    ray: &RayA,
    t_range: Range<f32>,
) -> bool {
    let local_origin = inv_transform.transform_point3a(ray.origin);
    let local_dir = inv_transform.transform_vector3a(ray.dir);
    let local_ray = RayA::new(local_origin, local_dir, ray.time);
    shape.occluded(&local_ray, t_range)
}

// ═══════════════════════════════════════════════════════════════
// ShapeInstance（静态）
// ═══════════════════════════════════════════════════════════════

pub struct ShapeInstance {
    pub id: InstanceId,
    pub shape: AnyShapeA,
    pub transform: Affine3A,
    pub inv_transform: Affine3A,
    /// alpha cutout 纹理。`None` = 不透明。
    ///
    /// `TextureId` 是 `Library` 全局句柄，`lib.alpha(id)` 取回
    /// `&Texture<AlphaChannel>`。常量 alpha 也走纹理——构建时注册一个
    /// `ConstantTexture<AlphaChannel>`，其 id 赋给本字段。
    pub alpha: Option<TextureId>,
}

impl ShapeInstance {
    pub fn new(id: InstanceId, shape: AnyShapeA, transform: Affine3A) -> Self {
        let det = transform.matrix3.determinant();
        assert!(
            det.abs() > 1e-12,
            "shape instance transform is not invertible (det = {det})"
        );
        Self {
            id,
            shape,
            transform,
            inv_transform: transform.inverse(),
            alpha: None,
        }
    }

    /// 设置 alpha cutout 纹理（构建期调用）。
    pub fn with_alpha(mut self, alpha: TextureId) -> Self {
        self.alpha = Some(alpha);
        self
    }
}

impl InstancePrimitive for ShapeInstance {
    #[inline]
    fn intersect<S: SpectralAlgebra>(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> Option<InstanceHit> {
        let hit = intersect_with_transform(
            &self.shape,
            &self.transform,
            &self.inv_transform,
            self.alpha,          // Option<TextureId> 按值传
            ray,
            t_range,
            lib,
        )?;
        Some(InstanceHit::Shape(self.id, hit))
    }

    #[inline]
    fn occluded<S: SpectralAlgebra>(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        lib: &Library<S>,
    ) -> bool {
        match self.alpha {
            // 快路径：无 alpha，短路 occluded，不构造 ShapeHit。
            None => occluded_with_transform(&self.shape, &self.inv_transform, ray, t_range),
            // 慢路径：有 alpha，需要 UV 采样，走 intersect 拿候选再判。
            Some(_) => self.intersect(ray, t_range, lib).is_some(),
        }
    }
}

impl Bounded for ShapeInstance {
    #[inline]
    fn bounds(&self) -> Aabb {
        self.shape.world_aabb(&self.transform)
    }
}
