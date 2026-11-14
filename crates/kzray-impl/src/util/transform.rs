//! 变换工具
//!
//! 与具体相机模型无关的构造工具，放在这里供 `pinhole` / 未来 `ortho` /
//! `panorama` 等模块复用。**不依赖 image crate 或任何外部 IO**——纯数学。

use glam::{Affine3A, Mat3, Vec3};

/// 构造 look-at 变换。
///
/// 返回的 `Affine3A` 满足：局部空间 +Z 指向 `target - eye`，+Y 接近 `up`，
/// +X 为 `forward × up`（右手系）。
///
/// # 参数
///
/// - `eye`：相机位置（世界空间）；
/// - `target`：注视点（世界空间）；
/// - `up`：上方向参考（世界空间），通常是 `Vec3::Y`。
///
/// # Panics
///
/// - `eye == target`（退化，无方向）；
/// - `up` 平行于 `target - eye`（右向量退化）。
///
/// 这两个都是调用方错误，panic 比静默 fallback 更安全——静默 fallback
/// 会让相机朝向出人意料，调试成本高。
///
/// # 示例
///
/// ```text
/// let transform = look_at(
///     Vec3::new(0.0, 2.0, 5.0),
///     Vec3::ZERO,
///     Vec3::Y,
/// );
/// ```
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Affine3A {
    let forward = target - eye;
    let forward_len = forward.length();
    assert!(
        forward_len > 1e-6,
        "look_at: eye and target are too close (distance {forward_len})"
    );
    let forward = forward / forward_len;

    let right = forward.cross(up);
    let right_len = right.length();
    assert!(
        right_len > 1e-6,
        "look_at: up {up:?} is parallel to view direction {forward:?}"
    );
    let right = right / right_len;

    let new_up = right.cross(forward);
    let rot = Mat3::from_cols(right, new_up, forward);
    Affine3A::from_mat3_translation(rot, eye)
}

