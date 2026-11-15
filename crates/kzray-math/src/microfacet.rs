//! 微表面分布。GGX 和 Beckmann 两种 NDF，Smith 遮蔽函数，可见法线采样。
//!
//! 粗糙度参数 `alpha` 是分布形状参数（不是 `roughness`）。
//! 常见映射：`alpha = roughness²`（UE4 / PBRT-v4），由调用方决定。
//!
//! 所有方向都是局部空间（`z` 轴为宏观法线 `n`）。世界空间变换用 `Onb`。

use glam::Vec3;
use std::f32::consts::PI;

// ═══════════════════════════════════════════════════════════════
// 公共辅助
// ═══════════════════════════════════════════════════════════════

/// 关于单位法线 `n` 的反射。`n` 通常是半向量 `h`。
/// 用于 `wi = reflect(wo, h)`，`wo` / `wi` 都是离开表面的方向。
#[inline]
pub fn reflect(wo: Vec3, n: Vec3) -> Vec3 {
    2.0 * wo.dot(n) * n - wo
}

/// 计算半向量并归一化。`wo` / `wi` 同向表面外。
#[inline]
pub fn half_vector(wo: Vec3, wi: Vec3) -> Vec3 {
    (wo + wi).normalize_or_zero()
}

// ═══════════════════════════════════════════════════════════════
// GGX (Trowbridge-Reitz)
// ═══════════════════════════════════════════════════════════════

pub mod ggx {
    use super::*;

    /// GGX NDF 求值。`m` 是微表面法线（局部空间，归一化），
    /// `cos_m = m.z` 是 `n·m`。`alpha` 是形状参数。
    ///
    /// `D(m) = α² / (π · ((n·m)²·(α² - 1) + 1)²)`
    #[inline]
    pub fn d(cos_m: f32, alpha: f32) -> f32 {
        if cos_m <= 0.0 {
            return 0.0;
        }
        let a2 = alpha * alpha;
        let c2 = cos_m * cos_m;
        let denom = c2 * (a2 - 1.0) + 1.0;
        a2 / (PI * denom * denom)
    }

    /// Smith Λ 函数（GGX）。
    /// `Λ(v) = (-1 + √(1 + α²·tan²θv)) / 2`
    #[inline]
    fn lambda(cos_v: f32, alpha: f32) -> f32 {
        let tan2 = (1.0 - cos_v * cos_v) / (cos_v * cos_v).max(1e-12);
        (-1.0 + (1.0 + alpha * alpha * tan2).sqrt()) * 0.5
    }

    /// Smith G1（遮蔽 + 阴影的单向项）。
    /// `G1(v) = 1 / (1 + Λ(v))`
    #[inline]
    pub fn g1(cos_v: f32, alpha: f32) -> f32 {
        if cos_v <= 0.0 {
            return 0.0;
        }
        let tan2 = (1.0 - cos_v * cos_v) / (cos_v * cos_v).max(1e-12);
        let lam = (-1.0 + (1.0 + alpha * alpha * tan2).sqrt()) * 0.5;
        1.0 / (1.0 + lam)
    }

    /// Smith G2，高度相关形式（Heitz 2014）。
    /// `G2(wo, wi) = 1 / (1 + Λ(wo) + Λ(wi))`
    #[inline]
    pub fn g2(cos_o: f32, cos_i: f32, alpha: f32) -> f32 {
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return 0.0;
        }
        let lo = lambda(cos_o, alpha);
        let li = lambda(cos_i, alpha);
        1.0 / (1.0 + lo + li)
    }

    /// Smith G2，可分离形式。`G2 = G1(wo) · G1(wi)`。
    /// 更便宜但精度略低，适合实时渲染；离线渲染用 `g2`。
    #[inline]
    pub fn g2_separable(cos_o: f32, cos_i: f32, alpha: f32) -> f32 {
        g1(cos_o, alpha) * g1(cos_i, alpha)
    }

    /// 可见法线采样（Heitz 2018）。输入 `wo`（局部空间，`wo.z > 0`）
    /// 和随机数 `u1, u2`，返回采样到的半向量 `m`（局部空间，归一化）。
    ///
    /// PDF 用 `pdf_vndf` 计算，**不由 `sample_vndf` 返回**——保证
    /// 采样和 PDF 独立实现，测试可以验证两者一致。
    pub fn sample_vndf(wo: Vec3, alpha: f32, u1: f32, u2: f32) -> Vec3 {
        let vh = Vec3::new(alpha * wo.x, alpha * wo.y, wo.z).normalize();

        let lensq = vh.x * vh.x + vh.y * vh.y;
        let t1 = if lensq > 0.0 {
            Vec3::new(-vh.y, vh.x, 0.0) / lensq.sqrt()
        } else {
            Vec3::X
        };
        let t2 = vh.cross(t1);

        let r = u1.sqrt();
        let phi = 2.0 * PI * u2;
        let p1 = r * phi.cos();
        let mut p2 = r * phi.sin();
        let s = 0.5 * (1.0 + vh.z);
        p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;

        let nh = p1 * t1 + p2 * t2 + (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt() * vh;
        Vec3::new(alpha * nh.x, alpha * nh.y, nh.z.max(1e-6)).normalize()
    }

    /// VNDF 采样半向量 `m` 的 PDF（关于 m 的立体角密度）。
    ///
    /// `pdf(m) = D(m) · G1(wo, m) · max(0, wo·m) / (n·wo)`
    pub fn pdf_vndf(wo: Vec3, m: Vec3, alpha: f32) -> f32 {
        if wo.z <= 0.0 || m.z <= 0.0 {
            return 0.0;
        }
        let cos_om = wo.dot(m);
        if cos_om <= 0.0 {
            return 0.0;
        }
        let d = d(m.z, alpha);
        let g1 = g1(wo.z, alpha);
        d * g1 * cos_om / wo.z
    }
}

// ═══════════════════════════════════════════════════════════════
// Beckmann
// ═══════════════════════════════════════════════════════════════

pub mod beckmann {
    use super::*;

    #[inline]
    pub fn d(cos_m: f32, alpha: f32) -> f32 {
        if cos_m <= 0.0 {
            return 0.0;
        }
        let c2 = cos_m * cos_m;
        let tan2 = (1.0 - c2) / c2;
        let a2 = alpha * alpha;
        (-tan2 / a2).exp() / (PI * a2 * c2 * c2)
    }

    #[inline]
    fn lambda(cos_v: f32, alpha: f32) -> f32 {
        if cos_v <= 0.0 {
            return f32::INFINITY;
        }
        if cos_v >= 1.0 {
            return 0.0;
        }
        let sin_v = (1.0 - cos_v * cos_v).max(0.0).sqrt();
        let tan_v = sin_v / cos_v;
        let a = 1.0 / (alpha * tan_v.max(1e-9));
        let erf_a = erf_approx(a);
        (erf_a - 1.0) * 0.5 + (-a * a).exp() / (2.0 * a * PI.sqrt()).max(1e-9)
    }

    #[inline]
    pub fn g1(cos_v: f32, alpha: f32) -> f32 {
        if cos_v <= 0.0 {
            return 0.0;
        }
        if cos_v >= 1.0 {
            return 1.0;
        }
        1.0 / (1.0 + lambda(cos_v, alpha))
    }

    #[inline]
    pub fn g2(cos_o: f32, cos_i: f32, alpha: f32) -> f32 {
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return 0.0;
        }
        if cos_o >= 1.0 && cos_i >= 1.0 {
            return 1.0;
        }
        let lo = lambda(cos_o, alpha);
        let li = lambda(cos_i, alpha);
        1.0 / (1.0 + lo + li)
    }

    /// **NDF 采样**（不是 VNDF）。从 `D(m)·cosθ_m` 采样微表面法线 `m`。
    ///
    /// 公式：`tan²θ = -α²·ln(1-u1)`，`φ = 2π·u2`。
    ///
    /// **这不是可见法线采样**。采样的 `m` 可能不可见（`wo·m < 0`），
    /// 下游 BSDF 必须检查 `wo·m > 0`，无效样本返回零贡献。
    ///
    /// Beckmann 没有解析的 VNDF 采样（需要 `erfinv` 或 rejection sampling），
    /// 这是 PBRT-v3、Cycles、Mitsuba 早期版本共用的近似。
    /// 2018 方法我们之后再做
    #[inline]
    pub fn sample_ndf(alpha: f32, u1: f32, u2: f32) -> Vec3 {
        let phi = 2.0 * PI * u2;
        let tan2 = -alpha * alpha * (1.0 - u1).max(1e-12).ln();
        let cos_theta = 1.0 / (1.0 + tan2).sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
    }

    /// NDF 采样 `sample_ndf` 的 PDF：`p(m) = D(m)·cosθ_m`。
    #[inline]
    pub fn pdf_ndf(m: Vec3, alpha: f32) -> f32 {
        if m.z <= 0.0 {
            return 0.0;
        }
        d(m.z, alpha) * m.z
    }

    fn erf_approx(x: f32) -> f32 {
        let sign = x.signum() as f64;
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x) as f64;
        let y = 1.0
            - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t
                * (-x * x).exp() as f64;
        (sign * y) as f32
    }
}
