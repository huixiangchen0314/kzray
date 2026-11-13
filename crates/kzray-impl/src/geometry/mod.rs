//! 几何求交。
//!
//! 图元（`Sphere` / `Plane`）返回 [`GeometryHit`]——纯几何信息，不含
//! `instance_id` / `prim_id`。这两个字段是场景层的分配信息（图元在场景
//! 中的位置、材质槽索引），图元本身不知道。场景层在求交结果上填充后
//! 转成 `SurfaceHit`。
//!
//! # 为什么分离
//!
//! - **图元是纯几何**：球体不知道自己挂在哪个 instance 下。把 `instance_id`
//!   塞进图元会导致同一球体无法被多个 instance 复用（不同 id）。
//! - **`GeometryHit` → `SurfaceHit` 是场景层的职责**：场景层知道图元与
//!   instance / 材质槽的映射关系。
//!
//! # 法线
//!
//! 图元返回的 `geometric_normal` / `shading_normal` 是**几何法线**（当前
//! 不支持法线贴图）。两者相同。着色时由 `BSDFBuilder` 根据 `wo` 决定
//! 是否翻转（双面材质），不在图元层处理。

use glam::{Vec2, Vec3};

use kzray_abs::ray::SurfaceHit;
use kzray_data::common::InstanceId;

pub mod plane;
pub mod sphere;

pub use plane::Plane;
pub use sphere::Sphere;

// ═══════════════════════════════════════════════════════════════
// GeometryHit
// ═══════════════════════════════════════════════════════════════

/// 图元求交的纯几何结果。
///
/// 不含 `instance_id` / `prim_id`——这两个字段由场景层在求交后填充，
/// 转成 [`SurfaceHit`]。
#[derive(Clone, Copy, Debug)]
pub struct GeometryHit {
    /// 光线参数。击中点 = `ray.origin + t × ray.dir`。
    pub t: f32,
    /// 从光线起点到命中点的实际距离，等于 `t × |ray.dir|`。
    pub distance: f32,
    /// 击中点世界坐标。
    pub position: Vec3,
    /// 着色法线（当前与几何法线相同）。
    pub shading_normal: Vec3,
    /// 几何法线（面法线）。
    pub geometric_normal: Vec3,
    /// 表面 UV。
    pub uv: Vec2,
}

impl GeometryHit {
    /// 转成 `SurfaceHit`，填充场景层分配的 `instance_id` / `prim_id`。
    pub fn into_surface_hit(self, instance_id: InstanceId, prim_id: u32) -> SurfaceHit {
        SurfaceHit {
            t: self.t,
            distance: self.distance,
            position: self.position,
            shading_normal: self.shading_normal,
            geometric_normal: self.geometric_normal,
            uv: self.uv,
            instance_id,
            prim_id,
        }
    }
}
