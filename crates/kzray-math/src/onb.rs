//! 正交基。给定归一化法线 `n`，构造右手系 `(u, v, w)`，`w = n`。
//!
//! 实现用 Duff et al. 2017, "Building an Orthonormal Basis, Revisited"。
//! Branchless，任意归一化 `n` 都鲁棒，包括 `n = (0, 0, -1)` 这个经典退化方向。
//! PBRT-v4 / Mitsuba 3 均采用此版本。

use glam::Vec3;

/// 正交基。`w` 是输入法线本身，`u` / `v` 是与 `w` 正交的单位向量。
///
/// 右手系：`u × v = w`。
#[derive(Clone, Copy, Debug)]
pub struct Onb {
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
}

impl Onb {
    /// 从归一化法线构造正交基。
    ///
    /// **输入必须已归一化**（debug_assert 检查，release 不检查）。
    #[inline]
    pub fn from_w(n: Vec3) -> Self {
        debug_assert!(
            (n.length_squared() - 1.0).abs() < 1e-4,
            "Onb::from_w 需要归一化法线，实际 |n|² = {}",
            n.length_squared()
        );
        let sign = 1.0f32.copysign(n.z);
        let a = -1.0 / (sign + n.z);
        let b = n.x * n.y * a;
        Onb {
            u: Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
            v: Vec3::new(b, sign + n.y * n.y * a, -n.y),
            w: n,
        }
    }

    /// 局部空间 → 世界空间。`local = (x, y, z)` 在基 `(u, v, w)` 下的表示。
    #[inline]
    pub fn to_world(&self, local: Vec3) -> Vec3 {
        self.u * local.x + self.v * local.y + self.w * local.z
    }

    /// 世界空间 → 局部空间。`to_local(to_world(x)) == x`。
    #[inline]
    pub fn to_local(&self, world: Vec3) -> Vec3 {
        Vec3::new(world.dot(self.u), world.dot(self.v), world.dot(self.w))
    }
}
