//! kzray-impl/src/accel/animated.rs
//! 动画化实例

use std::ops::Range;

use glam::Affine3A;
use kzray_data::{Aabb, InstanceId};

use crate::{
    accel::{
        Bounded,
        alpha::AlphaSource,
        instance::{
            InstanceHit, InstancePrimitive, intersect_with_transform,
            occluded_with_transform,
        },
        resource::ResourceTable,
    },
    geometry::{AnyShapeA, RayA},
};

/// 动画化解析形状实例。
///
/// 与 [`crate::accel::instance::ShapeInstance`] 的区别：变换按 `ray.time`
/// 在 t=0 和 t=1 两个关键帧之间插值。只在运动模糊场景使用；静态场景用
/// `ShapeInstance` 不付动画内存代价（`AnimatedShapeInstance` 是
/// `ShapeInstance` 的两倍大，多出来的正/逆变换各一份）。
///
/// # 插值方式
///
/// `Affine3A` 拆成 `(scale, rotation, translation)` 后分别插值：
///
/// - scale / translation：线性 `lerp`；
/// - rotation：`Quat::slerp`（逐元素 `lerp` 会失去正交性，矩阵不再
///   表示旋转）。
///
/// 正/逆变换**独立插值**——不在运行时求 `inverse()`，避免每光线一次
/// 矩阵求逆，且对非均匀缩放的 `slerp` 保持一致性。
pub struct AnimatedShapeInstance {
    pub id: InstanceId,
    pub shape: AnyShapeA,
    /// t=0 和 t=1 的正变换（局部 → 世界）。
    pub forward: [Affine3A; 2],
    /// t=0 和 t=1 的逆变换（世界 → 局部）。
    pub inverse: [Affine3A; 2],
    /// alpha cutout 源。`None` = 不透明（求交零成本）。
    pub alpha: Option<AlphaSource>,
}

impl AnimatedShapeInstance {
    /// 从两个关键帧的正变换构造，自动求逆。默认不透明。
    ///
    /// # Panics
    ///
    /// 任一变换不可逆（`|det| ≤ 1e-12`）时 panic。
    pub fn new(
        id: InstanceId,
        shape: AnyShapeA,
        transform_t0: Affine3A,
        transform_t1: Affine3A,
    ) -> Self {
        for (label, m) in [("t0", &transform_t0), ("t1", &transform_t1)] {
            let det = m.matrix3.determinant();
            assert!(
                det.abs() > 1e-12,
                "animated instance {label} transform is not invertible (det = {det})"
            );
        }
        Self {
            id,
            shape,
            forward: [transform_t0, transform_t1],
            inverse: [transform_t0.inverse(), transform_t1.inverse()],
            alpha: None,
        }
    }

    /// 设置 alpha cutout（构建期调用）。
    ///
    /// 链式风格：
    /// `AnimatedShapeInstance::new(...).with_alpha(AlphaSource::Constant(0.5))`。
    /// 不调用则保持不透明。
    pub fn with_alpha(mut self, alpha: AlphaSource) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// `time ∈ [0, 1]` 时刻的正变换。
    #[inline]
    fn forward_at(&self, time: f32) -> Affine3A {
        lerp_affine(&self.forward[0], &self.forward[1], time)
    }

    /// `time ∈ [0, 1]` 时刻的逆变换。
    #[inline]
    fn inverse_at(&self, time: f32) -> Affine3A {
        lerp_affine(&self.inverse[0], &self.inverse[1], time)
    }

    /// `time ∈ [0, 1]` 时刻的正逆变换对。
    #[inline]
    fn transform_at(&self, time: f32) -> (Affine3A, Affine3A) {
        (self.forward_at(time), self.inverse_at(time))
    }
}

impl InstancePrimitive for AnimatedShapeInstance {
    #[inline]
    fn intersect(
        &self,
        ray: &RayA,
        t_range: Range<f32>,
        table: &ResourceTable,
    ) -> Option<InstanceHit> {
        let (forward, inverse) = self.transform_at(ray.time);
        let hit = intersect_with_transform(
            &self.shape,
            &forward,
            &inverse,
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
            // 只解析逆变换（世界→局部），正变换对遮挡查询无用。
            None => {
                let inverse = self.inverse_at(ray.time);
                occluded_with_transform(&self.shape, &inverse, ray, t_range)
            }
            // 慢路径：有 alpha，需要 UV 采样，走 intersect 拿候选再判。
            Some(_) => self.intersect(ray, t_range, table).is_some(),
        }
    }
}

impl Bounded for AnimatedShapeInstance {
    /// 时间区间上的保守世界 AABB。
    ///
    /// 在 `[0, 1]` 上取 5 个等距采样点的 AABB 并集：
    ///
    /// - 纯平移动画：两端并集即精确覆盖，多采 3 点是冗余但无害；
    /// - 旋转动画：5 点覆盖常见角度范围，但**不保证绝对紧**——极端的
    ///   旋转（如绕轴 180°）中间时刻可能超出采样并集。若需要绝对保守，
    ///   DCC 应在提供 BVH 时给出精确的运动 AABB，或改用运动轨迹的球面
    ///   包围。
    ///
    /// alpha 不影响几何边界——透明实例仍占空间，AABB 不变。
    ///
    /// 构建期调用，不在热点路径。
    fn bounds(&self) -> Aabb {
        const SAMPLES: usize = 5;
        let mut b = self.shape.world_aabb(&self.forward[0]);
        for i in 1..SAMPLES {
            let t = i as f32 / (SAMPLES - 1) as f32;
            b = b.union(&self.shape.world_aabb(&self.forward_at(t)));
        }
        b
    }
}

// ═══════════════════════════════════════════════════════════════
// 仿射插值
// ═══════════════════════════════════════════════════════════════

/// 两个仿射变换之间的插值。
///
/// `Affine3A` 不能逐元素线性插值——旋转部分直接 `lerp` 会失去正交性。
/// 拆成 `(scale, rotation, translation)` 后：
///
/// - scale：`Vec3::lerp`；
/// - rotation：`Quat::slerp`；
/// - translation：`Vec3::lerp`。
///
/// `t` 由 `ray.time` 提供，已归一化到 `[0, 1]`。越界 `t` 会外推
/// （`slerp` 对 `t < 0` 或 `t > 1` 仍产生合法四元数，但语义上是外推）。
#[inline]
fn lerp_affine(a: &Affine3A, b: &Affine3A, t: f32) -> Affine3A {
    let (sa, qa, ta) = a.to_scale_rotation_translation();
    let (sb, qb, tb) = b.to_scale_rotation_translation();
    Affine3A::from_scale_rotation_translation(
        sa.lerp(sb, t),
        qa.slerp(qb, t),
        ta.lerp(tb, t),
    )
}

