//! 可见表面数据（GBuffer / AOV 输出用）。
//!
//! 本模块定义 `VisibleSurface`——求交产生的、**输出用**的几何缓存。
//! 它不依赖光线（没有 `t` / `distance` / 光线参数），只携带渲染结果
//! 需要写出的表面属性：位置、法线、UV、深度、物体标识。
//!
//! # 与 `kzray_abs::ray::SurfaceHit` 的区别
//!
//! | 类型 | 模块 | 用途 | 携带光线信息 |
//! |---|---|---|---|
//! | `SurfaceHit` | `kzray_abs::ray` | 求交结果，供 BSDF / 光源采样 | 是（`t` / `distance`） |
//! | `VisibleSurface` | 本模块 | 输出缓存，供 GBuffer / AOV | 否 |
//!
//! `SurfaceHit` 是**中间量**，一次求交产生，后续着色用完即弃；
//! `VisibleSurface` 是**持久量**，随样本累加到 GBuffer，最后写出图像。
//! 两者字段重叠（位置、法线、UV、实例 ID、图元 ID），但语义不同，
//! 不合并——`SurfaceHit` 在 `kzray_abs`（依赖 `Ray` / 采样器），
//! `VisibleSurface` 在 `kzray_data`（纯数据，无算法依赖）。
//!
//! # 转换
//!
//! 实现层从 `SurfaceHit` 构造 `VisibleSurface` 是自由函数（组合逻辑，
//! 不进契约层）：
//!
//! ```text
//! let visible = VisibleSurface {
//!     position: hit.position,
//!     normal: hit.shading_normal,
//!     uv: hit.uv,
//!     depth: /* 实现层约定：相机空间 z 或到相机距离 */,
//!     instance_id: hit.instance_id,
//!     prim_id: hit.prim_id,
//! };
//! ```
//!
//! `depth` 的语义由实现层约定，契约层不规定——相机空间 z、世界空间
//! 距离、透视除法后的 NDC 深度都可能，取决于下游 AOV 的消费方。

use glam::{Vec2, Vec3};
use crate::common::InstanceId;

// ═══════════════════════════════════════════════════════════════
// VisibleSurface
// ═══════════════════════════════════════════════════════════════

/// 求交产生的可见表面信息。
///
/// 纯数据 struct，无方法逻辑。用于：
///
/// - **GBuffer 胶片**：`FilmWithVisibleSurface::add_sample_with_visible`
///   接收它，填充深度 / 法线 / 物体 ID 通道；
/// - **AOV 输出**：直接作为附加通道写出；
/// - **后期处理**：景深、运动模糊、SSAO 等需要几何信息的效果。
///
/// 不用于：BSDF 求值、光源采样、路径追踪的中间计算——那些用
/// `kzray_abs::ray::SurfaceHit`，它携带光线参数。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisibleSurface {
    /// 击中点世界坐标。
    pub position: Vec3,

    /// 着色法线（可能经法线贴图或插值扰动）。
    ///
    /// 与 `SurfaceHit::shading_normal` 同义。GBuffer 的 normal 通道
    /// 通常写这个值，而不是几何法线——后期处理需要看到与渲染一致的
    /// 着色法线。
    pub normal: Vec3,

    /// 表面 UV。
    pub uv: Vec2,

    /// 深度。
    ///
    /// **语义由实现层约定**，契约层不规定。常见约定：
    ///
    /// - 相机空间 z（透视相机下线性）；
    /// - 世界空间到相机的欧氏距离；
    /// - 透视除法后的 NDC 深度（`[0, 1]` 或 `[-1, 1]`）。
    ///
    /// 下游 AOV 消费方（景深、SSAO、后处理）必须与产生方使用同一
    /// 约定，否则深度重建会错。约定写进实现层文档，不在这里强制。
    pub depth: f32,

    /// 实例 ID。用于查找物体 / 材质槽 / 物体级 AOV。
    pub instance_id: InstanceId,

    /// 实例内的图元索引。
    pub prim_id: u32,
}

impl VisibleSurface {
    /// 构造可见表面。
    ///
    /// 便捷构造器，字段全部显式传入；没有默认值或派生逻辑。
    pub fn new(
        position: Vec3,
        normal: Vec3,
        uv: Vec2,
        depth: f32,
        instance_id: InstanceId,
        prim_id: u32,
    ) -> Self {
        VisibleSurface {
            position,
            normal,
            uv,
            depth,
            instance_id,
            prim_id,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use crate::EntityId;

use super::*;

    fn sample_surface() -> VisibleSurface {
        VisibleSurface::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec2::new(0.25, 0.75),
            5.5,
            InstanceId::from(EntityId::from_name("kzray")),
            7,
        )
    }

    #[test]
    fn new_preserves_fields() {
        let s = sample_surface();
        assert_eq!(s.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(s.normal, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(s.uv, Vec2::new(0.25, 0.75));
        assert_eq!(s.depth, 5.5);
        assert_eq!(s.prim_id, 7);
    }

    #[test]
    fn copy_semantics() {
        let s = sample_surface();
        let copy = s;
        // Copy 后两者独立（值语义）
        assert_eq!(s, copy);
        // 修改副本不影响原值（验证是 Copy 不是移动）
        let mut another = s;
        another.depth = 10.0;
        assert_eq!(s.depth, 5.5);
        assert_eq!(another.depth, 10.0);
    }

    #[test]
    fn clone_and_debug() {
        let s = sample_surface();
        let cloned = s.clone();
        assert_eq!(s, cloned);
        // Debug 输出非空且包含类型名
        let debug_str = format!("{:?}", s);
        assert!(debug_str.contains("VisibleSurface"));
    }

    #[test]
    fn partial_eq_distinguishes_fields() {
        let a = sample_surface();
        let mut b = sample_surface();
        assert_eq!(a, b);
        b.depth = 99.0;
        assert_ne!(a, b);
    }

    #[test]
    fn send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VisibleSurface>();
    }
}
