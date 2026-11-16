//! kzray-impl/src/light/hdr_mapping.rs
//! 等距柱状投影（equirectangular）：方向 ↔ UV 映射。
//!
//! HDR 环境贴图的标准格式（Radiance `.hdr` / OpenEXR）。UV 定义：
//!
//! - `u = atan2(z, x) / (2π) + 0.5`，`u ∈ [0, 1)`，`u = 0.5` 对应 `+X`；
//! - `v = acos(y) / π`，`v ∈ [0, 1]`，`v = 0` 对应 `+Y`（天空 / 北极），
//!   `v = 1` 对应 `-Y`（地面 / 南极）。
//!
//! 与 `Sphere::uv` 的球面坐标一致——HDR 纹理本质是球面纹理。图像顶部
//! 像素对应天空，底部对应地面。这是 `.hdr` 文件的标准约定。
//!
//! # 边界
//!
//! `u` 在 `[-π, π] → [0, 1]` 映射下，`atan2` 返回 `(-π, π]`，`-π`
//! 对应 `u = 0.0`。`u = 0` 和 `u = 1` 是同一条经线（HDR 图像左右
//! 边缘是接缝）。采样时 `Wrap::Repeat` 处理接缝——`u = 1.0` 视作
//! `u = 0.0`。
//!
//! `v` 在 `[0, π] → [0, 1]` 映射下，极点（`±Y`）对应 `v = 0` / `v = 1`，
//! 双线性采样在极点行是水平线——极点的 UV 方位角不确定，但该行的
//! 像素值应一致（HDR 极点行通常是同色）。

use std::f32::consts::PI;

use glam::{Vec2, Vec3};

// ═══════════════════════════════════════════════════════════════
// 方向 → UV
// ═══════════════════════════════════════════════════════════════

/// 单位方向 → 等距柱状投影 UV。
///
/// # 参数
///
/// `dir` 必须归一化（`debug_assert` 检查）。
///
/// # 返回
///
/// `(u, v)`，`u ∈ [0, 1]`，`v ∈ [0, 1]`。`u` 的边界值 `1.0` 与 `0.0`
/// 是同一条经线——调用方用 `Wrap::Repeat` 处理。
#[inline]
pub fn direction_to_uv(dir: Vec3) -> Vec2 {
    debug_assert!(
        (dir.length_squared() - 1.0).abs() < 1e-4,
        "direction must be normalized, |dir|² = {}",
        dir.length_squared()
    );
    let u = dir.z.atan2(dir.x) / (2.0 * PI) + 0.5;
    let v = dir.y.clamp(-1.0, 1.0).acos() / PI;
    Vec2::new(u, v)
}

// ═══════════════════════════════════════════════════════════════
// UV → 方向
// ═══════════════════════════════════════════════════════════════

/// 等距柱状投影 UV → 单位方向。
///
/// `direction_to_uv` 的逆（`u` 的边界 `0/1` 共享同一条经线，往返时
/// 保持 `u` 的值）。测试用；未来若需要"从 UV 生成方向光"等场景也能用。
#[inline]
pub fn uv_to_direction(uv: Vec2) -> Vec3 {
    let phi = (uv.x - 0.5) * 2.0 * PI;
    let theta = uv.y * PI;
    let sin_theta = theta.sin();
    Vec3::new(
        sin_theta * phi.cos(),
        theta.cos(),
        sin_theta * phi.sin(),
    )
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn approx_uv(a: Vec2, b: Vec2) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y)
    }

    // ── 主方向 → UV ──

    #[test]
    fn plus_x_maps_to_center() {
        let uv = direction_to_uv(Vec3::X);
        assert!(approx_uv(uv, Vec2::new(0.5, 0.5)));
    }

    #[test]
    fn minus_x_maps_to_seam() {
        let uv = direction_to_uv(Vec3::NEG_X);
        // atan2(0, -1) = π → u = 1.0；或某些实现返回 -π → u = 0.0
        assert!(approx(uv.y, 0.5));
        assert!(
            approx(uv.x, 0.0) || approx(uv.x, 1.0),
            "u at -X should be 0 or 1, got {}",
            uv.x
        );
    }

    #[test]
    fn plus_z_maps_to_three_quarters() {
        let uv = direction_to_uv(Vec3::Z);
        assert!(approx_uv(uv, Vec2::new(0.75, 0.5)));
    }

    #[test]
    fn minus_z_maps_to_one_quarter() {
        let uv = direction_to_uv(Vec3::NEG_Z);
        assert!(approx_uv(uv, Vec2::new(0.25, 0.5)));
    }

    #[test]
    fn plus_y_maps_to_top() {
        let uv = direction_to_uv(Vec3::Y);
        assert!(approx(uv.y, 0.0), "v at +Y should be 0, got {}", uv.y);
    }

    #[test]
    fn minus_y_maps_to_bottom() {
        let uv = direction_to_uv(Vec3::NEG_Y);
        assert!(approx(uv.y, 1.0), "v at -Y should be 1, got {}", uv.y);
    }

    // ── UV 边界 ──

    #[test]
    fn uv_in_valid_range() {
        for dir in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(-1.0, 2.0, -3.0).normalize(),
        ] {
            let uv = direction_to_uv(dir);
            assert!(uv.x >= 0.0 && uv.x <= 1.0, "u = {} out of range", uv.x);
            assert!(uv.y >= 0.0 && uv.y <= 1.0, "v = {} out of range", uv.y);
        }
    }

    // ── 往返：direction → uv → direction ──

    #[test]
    fn roundtrip_direction_uv_direction() {
        let dirs = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::NEG_X,
            Vec3::NEG_Y,
            Vec3::NEG_Z,
            Vec3::new(1.0, 1.0, 0.0).normalize(),
            Vec3::new(1.0, 0.0, 1.0).normalize(),
            Vec3::new(-0.5, 0.3, 0.8).normalize(),
            Vec3::new(0.1, -0.9, -0.3).normalize(),
        ];
        for dir in dirs {
            let uv = direction_to_uv(dir);
            let back = uv_to_direction(uv);
            assert!(
                (dir - back).length() < 1e-4,
                "roundtrip dir = {:?} → uv = {:?} → back = {:?}",
                dir,
                uv,
                back
            );
        }
    }

    #[test]
    fn roundtrip_uv_direction_uv() {
        let uvs = [
            Vec2::new(0.5, 0.5),
            Vec2::new(0.25, 0.5),
            Vec2::new(0.75, 0.5),
            Vec2::new(0.5, 0.0),
            Vec2::new(0.5, 1.0),
            Vec2::new(0.3, 0.7),
            Vec2::new(0.8, 0.2),
        ];
        for uv in uvs {
            let dir = uv_to_direction(uv);
            let back = direction_to_uv(dir);
            // u=0.5, v=0/1 是极点——方位角任意，跳过
            if uv.y > 1e-4 && uv.y < 1.0 - 1e-4 && uv.x > 1e-4 && uv.x < 1.0 - 1e-4 {
                assert!(approx_uv(uv, back), "uv {:?} → dir {:?} → {:?}", uv, dir, back);
            }
        }
    }

    // ── 极端方向 ──

    #[test]
    fn near_pole_no_nan() {
        for dir in [
            Vec3::new(1e-8, 1.0, 0.0).normalize(),
            Vec3::new(0.0, 0.99999, 1e-8).normalize(),
            Vec3::new(1e-6, -0.99999, 1e-6).normalize(),
        ] {
            let uv = direction_to_uv(dir);
            assert!(!uv.x.is_nan(), "u is NaN for {:?}", dir);
            assert!(!uv.y.is_nan(), "v is NaN for {:?}", dir);
        }
    }

    #[test]
    fn direction_length_is_preserved_by_inverse() {
        // uv_to_direction 总是返回单位向量
        for uv in [
            Vec2::new(0.0, 0.0),
            Vec2::new(0.5, 0.5),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.3, 0.9),
        ] {
            let dir = uv_to_direction(uv);
            assert!(
                (dir.length_squared() - 1.0).abs() < 1e-5,
                "|dir|² = {} at uv = {:?}",
                dir.length_squared(),
                uv
            );
        }
    }
}
