//! kzray-impl/src/accel/instance.rs
//! 静态实例

use std::ops::Range;

use glam::{Affine3A, Vec3A};
use kzray_data::{Aabb, InstanceId};

use crate::{
    accel::{Bounded, alpha::AlphaSource, resource::ResourceTable},
    geometry::{AnyShapeA, RayA, ShapeA, ShapeHit},
};

pub enum InstanceHit {
    Shape(InstanceId, ShapeHit),
}

impl InstanceHit {
    /// 命中的仿射参数 `t`。
    #[inline]
    pub fn t(&self) -> f32 {
        match self {
            Self::Shape(_id, shape_hit) => shape_hit.t,
        }
    }

}

pub trait InstancePrimitive {
    fn intersect(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        table: &ResourceTable,
    ) -> Option<InstanceHit>;

    fn occluded(&self, ray: &RayA, t_range: Range<f32>, table: &ResourceTable) -> bool;
}

// ═══════════════════════════════════════════════════════════════
// 共享求交逻辑
// ═══════════════════════════════════════════════════════════════
//
// 静态实例和动画实例的唯一差别是"如何取得当前时刻的正/逆变换"。
// 变换解析后，后续逻辑（世界→局部、求交、命中回世界）完全一致，
// 抽成模块级函数由两个类型共享。

/// 给定已解析到当前时刻的正/逆变换，执行"世界 → 局部 → 求交 → 回世界"。
///
/// `inv_transform` 用于位置/方向/法线，`transform` 只用于命中位置回世界。
///
/// # alpha 语义
///
/// `alpha` 为 `Some` 时在拿到局部命中后采样 alpha；低于阈值视为透明，
/// 直接返回 `None`——BVH 看到这个图元 miss，自然继续访问其他叶子。
/// 跨图元的透明搜索由 BVH 遍历保证，无需额外推进逻辑。
#[inline]
pub(crate) fn intersect_with_transform(
    shape: &AnyShapeA,
    transform: &Affine3A,
    inv_transform: &Affine3A,
    alpha: Option<&AlphaSource>,
    ray: &RayA,
    t_range: Range<f32>,
    table: &ResourceTable,
) -> Option<ShapeHit> {
    let local_origin = inv_transform.transform_point3a(ray.origin);
    let local_dir = inv_transform.transform_vector3a(ray.dir);
    let local_ray = RayA::new(local_origin, local_dir, ray.time);

    let local_hit = shape.intersect(&local_ray, t_range)?;

    // ── alpha 判定：透明则视为未命中，BVH 自然继续 ──
    if let Some(a) = alpha {
        if !a.is_opaque(local_hit.uv, table) {
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

/// 遮挡查询的共享逻辑。只需 `inv_transform`（世界→局部），无需正向变换。
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

/// 静态解析形状实例。
///
/// 约定 `inv_transform == transform.inverse()`；用 [`ShapeInstance::new`]
/// 构造可保证这一点。
pub struct ShapeInstance {
    pub id: InstanceId,
    /// 内联的局部空间形状。
    pub shape: AnyShapeA,
    /// 局部 → 世界。
    pub transform: Affine3A,
    /// 世界 → 局部（`transform` 的逆）。
    pub inv_transform: Affine3A,
    /// alpha cutout 源。`None` = 不透明（求交零成本）。
    pub alpha: Option<AlphaSource>,
}

impl ShapeInstance {
    /// 从正向变换构造，自动求逆。默认不透明。
    ///
    /// # Panics
    ///
    /// `transform` 不可逆（`|det| ≤ 1e-12`）时 panic。
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

    /// 设置 alpha cutout（构建期调用）。
    ///
    /// 链式风格：`ShapeInstance::new(...).with_alpha(AlphaSource::Constant(0.5))`。
    /// 不调用则保持不透明。
    pub fn with_alpha(mut self, alpha: AlphaSource) -> Self {
        self.alpha = Some(alpha);
        self
    }
}

impl InstancePrimitive for ShapeInstance {
    #[inline]
    fn intersect(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        table: &ResourceTable,
    ) -> Option<InstanceHit> {
        let hit = intersect_with_transform(
            &self.shape,
            &self.transform,
            &self.inv_transform,
            self.alpha.as_ref(),
            ray,
            t_range,
            table,
        )?;
        Some(InstanceHit::Shape(self.id, hit))
    }

    #[inline]
    fn occluded(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        table: &ResourceTable,
    ) -> bool {
        match &self.alpha {
            // 快路径：无 alpha，短路 occluded，不构造 ShapeHit。
            None => occluded_with_transform(&self.shape, &self.inv_transform, ray, t_range),
            // 慢路径：有 alpha，需要 UV 采样，走 intersect 拿候选再判。
            Some(_) => self.intersect(ray, t_range, table).is_some(),
        }
    }
}

impl Bounded for ShapeInstance {
    /// 世界空间 AABB。委托给 `AnyShapeA::world_aabb`——各形状按自身
    /// 几何特征直接构造世界 AABB（球用行范数、三角形用 3 顶点），
    /// 不走 8 角点变换。
    ///
    /// alpha 不影响几何边界——透明实例仍占空间，AABB 不变。
    #[inline]
    fn bounds(&self) -> Aabb {
        self.shape.world_aabb(&self.transform)
    }
}
