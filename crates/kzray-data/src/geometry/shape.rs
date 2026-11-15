//! kzray-data/src/geometry/primitive.rs 解析形状

// ═══════════════════════════════════════════════════════════════
// Plane
// ═══════════════════════════════════════════════════════════════

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// 无限平面。
///
/// 由平面上一点和法线定义。**请用 [`Plane::new`] 构造**，它会归一化法线；
/// 直接写结构体字面量时法线不会自动归一化，调用方需自行保证 `normal_*`
/// 是单位向量。
///
/// # UV 参数化
///
/// 用世界坐标的 `(x, z)` 投影（假设平面大致水平，法线接近 ±Y）。
/// 这是一个 hack——正确的平面 UV 需要切线 + 副切线，但无限平面没有
/// 自然的 UV 原点。对 PT 调试场景（地面 / 天花板 / 墙面）够用。
///
/// 未来如果平面需要纹理，加 `tangent` / `bitangent` 字段，按面内坐标
/// 计算 UV。
///
/// # 字段布局
///
/// 为兼容无 `serde` feature 的 `glam`，`Vec3` 拆成独立 `f32` 字段。
/// 因此**不再实现 `Eq`**（`f32` 无 `Eq`），只保留 `PartialEq`。
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct Plane {
    /// 平面上一点的 x。
    pub point_x: f32,
    /// 平面上一点的 y。
    pub point_y: f32,
    /// 平面上一点的 z。
    pub point_z: f32,
    /// 单位法线的 x。
    pub normal_x: f32,
    /// 单位法线的 y。
    pub normal_y: f32,
    /// 单位法线的 z。
    pub normal_z: f32,
}

impl Plane {
    /// 构造平面，并把 `normal` 归一化。
    ///
    /// # Panics
    ///
    /// `normal` 为零向量时 `normalize()` 产生 NaN——传零向量是调用方 bug。
    pub fn new(point: Vec3, normal: Vec3) -> Self {
        let n = normal.normalize();
        Self {
            point_x: point.x,
            point_y: point.y,
            point_z: point.z,
            normal_x: n.x,
            normal_y: n.y,
            normal_z: n.z,
        }
    }

    /// 平面上一点（由独立字段重组）。
    #[inline]
    pub fn point(&self) -> Vec3 {
        Vec3::new(self.point_x, self.point_y, self.point_z)
    }

    /// 单位法线（由独立字段重组）。
    #[inline]
    pub fn normal(&self) -> Vec3 {
        Vec3::new(self.normal_x, self.normal_y, self.normal_z)
    }
}

// ═══════════════════════════════════════════════════════════════
// Sphere
// ═══════════════════════════════════════════════════════════════

/// 球体。
///
/// 解析求交，二次方程。`ray.dir` 可以不归一化——`a = |dir|²` 处理，
/// `distance = t × |dir|`。
///
/// # UV 参数化
///
/// 标准球面坐标：
///
/// - `u = atan2(z, x) / (2π) + 0.5`；
/// - `v = acos(y / radius) / π`。
///
/// `y` 轴为球体北极。接缝在 `x = -radius, z = 0`（`u = 0/1` 处）。
///
/// # 字段布局
///
/// 同 [`Plane`]，`Vec3` 拆成独立 `f32` 字段，故不实现 `Eq`。
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct Sphere {
    pub center_x: f32,
    pub center_y: f32,
    pub center_z: f32,
    pub radius: f32,
}

impl Sphere {
    #[inline]
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self {
            center_x: center.x,
            center_y: center.y,
            center_z: center.z,
            radius,
        }
    }

    /// 球心（由独立字段重组）。
    #[inline]
    pub fn center(&self) -> Vec3 {
        Vec3::new(self.center_x, self.center_y, self.center_z)
    }
}
