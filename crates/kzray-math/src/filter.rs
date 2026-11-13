//! 重建滤波器。
//!
//! Filter 定义样本在像素邻域的权重分布，决定图像重建质量。它是
//! **采样数学**，不属于相机也不属于胶片——Camera 用它做采样范围抖动
//! （可选），Film 用它做 splat 权重分配（必需）。
//!
//! # 两个 trait，职责分离
//!
//! - [`Filter`]：基础权重函数（[`Filter::weight`] / [`Filter::radius`]）。
//!   **所有** filter 都实现。Film 的 splat 用它做权重分配，这是必需的。
//! - [`SamplableFilter`]：可采样的 filter（[`SamplableFilter::sample`] /
//!   [`SamplableFilter::pdf`]）。**可选**能力，只有需要 filter 重要性
//!   采样的 filter 实现。Camera 用它做重要性采样，降低方差。
//!
//! 拆成独立 trait 的原因是：**默认实现是错的**。均匀采样 + 简单接受对
//! Box 正确，对 Triangle / Gaussian **有偏**。把错误的默认实现挂在
//! `Filter` 上，会诱使实现方继承一个静默 bug。独立 trait 强制每个
//! filter 显式决定是否支持采样，且必须自己实现正确的 `sample` / `pdf`。
//!
//! 当前 PT 不做 filter 重要性采样（相机均匀抖动即可），所以 Film 只需
//! [`Filter`]；Camera 侧需要时用 [`SamplableFilter`]。
//!
//! # 归一化
//!
//! `weight` 返回的是权重函数值，**不保证在整个 `radius` 范围内积分为 1**。
//! 归一化策略由使用方决定：
//!
//! - **Film 的 splat**：离散求和归一化——对所有覆盖的邻域像素权重求和，
//!   除以该和得到归一化权重。选择离散求和而非连续积分，精确匹配"实际
//!   splat 到哪些像素"的权重和，避免连续积分与离散采样不一致的系统误差。
//! - **Camera 的 filter 重要性采样**：[`SamplableFilter::sample`] 返回的
//!   `pdf` 在 `[-radius, radius]²` 上积分为 1，`weight = f(offset) / pdf(offset)`
//!   已归一化。
//!
//! # 与 pbrt 的差异
//!
//! pbrt 的 `Filter` 把权重和采样合在一个类上，默认 `Sample` 用均匀采样 +
//! 拒绝（对非 Box 有偏，pbrt 通过 `Integral()` 修正归一化常数）。我们拆成
//! 两个 trait，`SamplableFilter` 无默认实现——**正确的采样必须显式写**。

use glam::Vec2;

// ═══════════════════════════════════════════════════════════════
// FilterSample
// ═══════════════════════════════════════════════════════════════

/// 按 filter 分布采样得到的一次结果。
///
/// 用于 Camera 侧的 filter 重要性采样：不是均匀抖动像素位置，而是按
/// filter 权重分布采样偏移，让样本更集中在权重高的区域，降低方差。
///
/// 字段与可分解性：`weight` 已包含 `f(offset) / pdf(offset)` 的比值，
/// 调用方直接乘到样本贡献上；`pdf` 独立保留，供 MIS 或需要精确概率的
/// 场景（如 BDPT 的端点连接）。`f(offset)` 可由 [`Filter::weight`] 重算。
#[derive(Clone, Copy, Debug)]
pub struct FilterSample {
    /// 采样到的偏移，范围在 `[-radius.x, radius.x] × [-radius.y, radius.y]`。
    ///
    /// 相对于像素中心的连续偏移；加上像素中心坐标即为胶片上的采样位置。
    pub offset: Vec2,
    /// 采样权重 `f(offset) / pdf(offset)`。
    ///
    /// 直接乘到样本贡献上；`f(offset)` 是 filter 权重函数在 `offset` 处
    /// 的值，`pdf(offset)` 是采样到该偏移的概率密度。
    pub weight: f32,
    /// 采样到 `offset` 的概率密度，在 `[-radius, radius]²` 上积分为 1。
    ///
    /// 契约：对 `s = sample(uv)`，`pdf(s.offset) == s.pdf`，且
    /// `s.weight == weight(s.offset) / s.pdf`。跨实现测试应校验。
    pub pdf: f32,
}

// ═══════════════════════════════════════════════════════════════
// Filter：基础权重
// ═══════════════════════════════════════════════════════════════

/// 基础滤波器契约：权重函数。
///
/// 所有 filter 都实现本 trait。Film 的 splat 用它做权重分配，这是
/// **必需**的能力。
///
/// 所有实现必须 `Send + Sync`：Filter 在构造层创建，被 Camera 和 Film
/// 共享（通常是 `Arc<dyn Filter>`），可能跨线程访问。
///
/// `Filter` 是 object-safe 的：两个方法都不是泛型方法，`Box<dyn Filter>`
/// 可用。
pub trait Filter: Send + Sync {
    /// 给定相对于像素中心的偏移，返回权重。
    ///
    /// 偏移范围在 `[-radius, radius]` 内；超出范围返回 0。
    /// 返回值不保证归一化，且可能为 0（filter 边界外或权重函数负部）。
    fn weight(&self, offset: Vec2) -> f32;

    /// filter 半径。采样范围和 splat 范围都以它为准。
    ///
    /// `x` / `y` 半径可以不同（各向异性 filter）。调用方遍历邻域像素
    /// 时，用 `ceil(radius.x)` / `ceil(radius.y)` 确定范围。
    fn radius(&self) -> Vec2;
}

// ═══════════════════════════════════════════════════════════════
// SamplableFilter：可采样
// ═══════════════════════════════════════════════════════════════

/// 可采样的滤波器：按 filter 分布采样偏移。
///
/// 用于 Camera 侧的 filter 重要性采样——不是均匀抖动像素位置，而是按
/// filter 权重分布采样偏移，让样本更集中在权重高的区域，降低方差。
///
/// **无默认实现**：`sample` / `pdf` 必须由每个 filter 显式实现。均匀
/// 采样 + 简单接受的默认实现对非 Box filter 有偏，不能作为默认行为。
///
/// # 一致性契约
///
/// 对每个实现，必须满足：
///
/// - **`sample` / `pdf` 一致性**：对任意 `s = sample(uv)`，应有
///   `pdf(s.offset) == s.pdf`。跨实现测试必须校验。
/// - **`weight` / `sample` 一致性**：对任意 `s = sample(uv)`，应有
///   `s.weight == weight(s.offset) / s.pdf`（浮点误差允许范围）。
/// - **采样域**：`sample` 返回的 `offset` 在 `[-radius, radius]²` 内；
///   `pdf(offset)` 在该域外返回 0。
///
/// # Object-safety
///
/// `SamplableFilter` 是 object-safe 的：`sample` / `pdf` 都不是泛型方法，
/// `Box<dyn SamplableFilter>` 可用。因为 `SamplableFilter: Filter`，
/// trait upcasting 可以把 `Box<dyn SamplableFilter>` 转成
/// `Box<dyn Filter>`（Rust 1.86+ 稳定）。
pub trait SamplableFilter: Filter {
    /// 按 filter 分布采样一个偏移。
    ///
    /// `uv` 是 2D 均匀样本 `[0, 1)²`。返回 `None` 表示采样失败（如拒绝
    /// 采样未接受、`uv` 落在无效区域）。调用方应重试或跳过。
    ///
    /// 返回的 `offset` 在 `[-radius, radius]²` 内，`weight` 已按
    /// `f/pdf` 修正，`pdf` 是采样密度。
    fn sample(&self, uv: Vec2) -> Option<FilterSample>;

    /// 给定偏移，返回采样 PDF。
    ///
    /// 必须与 [`SamplableFilter::sample`] 使用同一分布。`offset` 超出
    /// `[-radius, radius]²` 时返回 0。
    fn pdf(&self, offset: Vec2) -> f32;
}

// ═══════════════════════════════════════════════════════════════
// BoxFilter
// ═══════════════════════════════════════════════════════════════

/// 盒式滤波器（最近邻）。
///
/// `radius` 内权重恒为 1，外为 0。最简单、最快，但重建质量差——
/// 相当于最近邻插值，会引入明显的锯齿。
///
/// **可采样**：均匀分布在 `[-radius, radius]²`，`pdf = 1 / area`。
/// 解析精确，无拒绝。
///
/// 典型用法：调试、快速预览、不需要抗锯齿的场景。
#[derive(Clone, Copy, Debug)]
pub struct BoxFilter {
    pub radius: Vec2,
}

impl BoxFilter {
    pub fn new(radius: Vec2) -> Self {
        BoxFilter { radius }
    }
}

impl Filter for BoxFilter {
    #[inline]
    fn weight(&self, offset: Vec2) -> f32 {
        if offset.x.abs() <= self.radius.x && offset.y.abs() <= self.radius.y {
            1.0
        } else {
            0.0
        }
    }

    #[inline]
    fn radius(&self) -> Vec2 {
        self.radius
    }
}

impl SamplableFilter for BoxFilter {

    /// 均匀采样 + 简单接受
    fn sample(&self, uv: Vec2) -> Option<FilterSample> {
        let offset = Vec2::new(
            (2.0 * uv.x - 1.0) * self.radius.x,
            (2.0 * uv.y - 1.0) * self.radius.y,
        );
        let area = 4.0 * self.radius.x * self.radius.y;
        if area <= 0.0 {
            return None;
        }
        let pdf = 1.0 / area;
        Some(FilterSample {
            offset,
            weight: self.weight(offset) / pdf,
            pdf,
        })
    }

    fn pdf(&self, offset: Vec2) -> f32 {
        let r = self.radius;
        if offset.x.abs() > r.x || offset.y.abs() > r.y {
            0.0
        } else {
            1.0 / (4.0 * r.x * r.y)
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// TriangleFilter
// ═══════════════════════════════════════════════════════════════

/// 三角滤波器（线性插值）。
///
/// 权重从中心线性衰减到 `radius` 处的 0。比 `BoxFilter` 平滑，
/// 是最常用的默认 filter 之一——计算简单，重建质量可接受。
///
/// **可采样**：两轴独立采样三角分布，`pdf` 是各轴三角分布的乘积。
/// 解析精确，无拒绝。
///
/// 典型用法：标准 PT 的默认 filter，`radius = (1, 1)` 或 `(0.5, 0.5)`。
#[derive(Clone, Copy, Debug)]
pub struct TriangleFilter {
    pub radius: Vec2,
}

impl TriangleFilter {
    pub fn new(radius: Vec2) -> Self {
        TriangleFilter { radius }
    }
}

impl Filter for TriangleFilter {
    #[inline]
    fn weight(&self, offset: Vec2) -> f32 {
        // 各轴独立的线性衰减，然后相乘（可分离 filter）。
        let wx = (1.0 - offset.x.abs() / self.radius.x).max(0.0);
        let wy = (1.0 - offset.y.abs() / self.radius.y).max(0.0);
        wx * wy
    }

    #[inline]
    fn radius(&self) -> Vec2 {
        self.radius
    }
}

impl SamplableFilter for TriangleFilter {
    /// 解析采样：两轴独立的三角分布。
    ///
    /// 单轴三角分布 `w(x) = 1 - |x|/r`，累积分布逆变换：
    ///
    /// ```text
    /// u < 0.5: x = r(sqrt(2u) - 1)
    /// u >= 0.5: x = r(1 - sqrt(2(1-u)))
    /// ```
    ///
    /// 2D 用两轴独立的逆变换，PDF 是两轴 PDF 的乘积。
    fn sample(&self, uv: Vec2) -> Option<FilterSample> {
        let offset = Vec2::new(
            sample_triangle_1d(uv.x, self.radius.x),
            sample_triangle_1d(uv.y, self.radius.y),
        );
        let pdf = self.pdf(offset);
        if pdf <= 0.0 {
            return None;
        }
        let weight = self.weight(offset) / pdf;
        Some(FilterSample {
            offset,
            weight,
            pdf,
        })
    }

    fn pdf(&self, offset: Vec2) -> f32 {
        let r = self.radius;
        if offset.x.abs() > r.x || offset.y.abs() > r.y {
            return 0.0;
        }
        // 单轴三角分布的 PDF：w(x) / r（归一化常数 Z = r）
        let wx = (1.0 - offset.x.abs() / r.x).max(0.0);
        let wy = (1.0 - offset.y.abs() / r.y).max(0.0);
        // 两轴独立：pdf = (wx / rx) * (wy / ry)
        (wx / r.x) * (wy / r.y)
    }
}

/// 单轴三角分布的逆变换采样。
///
/// `w(x) = 1 - |x|/r` 在 `[-r, r]` 上积分为 `r`，归一化 PDF 是 `w/r`。
#[inline]
fn sample_triangle_1d(u: f32, r: f32) -> f32 {
    if u < 0.5 {
        r * ((2.0 * u).sqrt() - 1.0)
    } else {
        r * (1.0 - (2.0 * (1.0 - u)).sqrt())
    }
}

// ═══════════════════════════════════════════════════════════════
// GaussianFilter
// ═══════════════════════════════════════════════════════════════

/// 高斯滤波器。
///
/// 权重为高斯衰减，并在 `radius` 处**减一个常数使边界权重为 0**，
/// 避免 filter 尾部被硬截断造成的不连续（pbrt 的做法）。
///
/// 公式：
///
/// ```text
/// weight(p) = max(0, exp(-sx·px² - sy·py²) - exp(-sx·rx² - sy·ry²))
/// ```
///
/// 其中 `sx = sy = 1 / (2σ²)`，`rx` / `ry` 是 `radius` 分量。
///
/// **可采样**：均匀采样 `[-radius, radius]²`，以 `weight(offset) /
/// weight(0)` 的概率接受。接受的样本分布正比于 `weight(offset)`，
/// 归一化常数 `Z = ∫ weight` 在构造时数值预计算。
///
/// 典型用法：高质量离线渲染，`radius = (1.5, 1.5)`，`sigma = 0.5`。
#[derive(Clone, Copy, Debug)]
pub struct GaussianFilter {
    pub radius: Vec2,
    /// 高斯的标准差参数，必须为正。
    pub sigma: f32,
    /// `weight` 在 `[-radius, radius]²` 上的数值积分，构造时预计算。
    ///
    /// 拒绝采样的归一化常数：`pdf(offset) = weight(offset) / integral`。
    integral: f32,
}

impl GaussianFilter {
    /// 构造高斯 filter。
    ///
    /// # Panics
    ///
    /// `sigma <= 0` 时 panic。`sigma` 为 0 会让权重除零，无意义。
    pub fn new(radius: Vec2, sigma: f32) -> Self {
        assert!(sigma > 0.0, "GaussianFilter sigma must be positive");
        assert!(
            radius.x > 0.0 && radius.y > 0.0,
            "GaussianFilter radius must be positive"
        );
        let mut filter = GaussianFilter {
            radius,
            sigma,
            integral: 0.0,
        };
        filter.integral = filter.compute_integral();
        filter
    }

    /// 数值积分 `weight` 在 `[-radius, radius]²` 上的值。
    ///
    /// 用 64×64 网格中点法；足够精确（高斯衰减快，边界贡献小）。
    /// 只为拒绝采样的归一化常数服务，不用于 Film splat（Film 用离散
    /// 求和归一化）。
    fn compute_integral(&self) -> f32 {
        const N: usize = 64;
        let dx = 2.0 * self.radius.x / N as f32;
        let dy = 2.0 * self.radius.y / N as f32;
        let mut sum = 0.0;
        for i in 0..N {
            for j in 0..N {
                let x = -self.radius.x + (i as f32 + 0.5) * dx;
                let y = -self.radius.y + (j as f32 + 0.5) * dy;
                sum += self.weight(Vec2::new(x, y));
            }
        }
        sum * dx * dy
    }
}

impl Filter for GaussianFilter {
    #[inline]
    fn weight(&self, offset: Vec2) -> f32 {
        // sx = sy = 1 / (2σ²)
        let s = 1.0 / (2.0 * self.sigma * self.sigma);
        // 边界处的衰减值：exp(-s·rx² - s·ry²)
        let boundary =
            (-s * (self.radius.x * self.radius.x + self.radius.y * self.radius.y)).exp();
        // 当前偏移的衰减值
        let d2 = offset.x * offset.x + offset.y * offset.y;
        ((-s * d2).exp() - boundary).max(0.0)
    }

    #[inline]
    fn radius(&self) -> Vec2 {
        self.radius
    }
}

impl SamplableFilter for GaussianFilter {
    /// 拒绝采样：均匀采点，以 `weight(offset) / weight(0)` 的概率接受。
    ///
    /// `uv` 只提供一次均匀样本；接受概率不足时返回 `None`，调用方
    /// 应重试或跳过。这是**单次尝试**的语义，不是内部循环——避免
    /// 把拒绝循环的开销藏进 trait 方法，让调用方控制重试策略。
    ///
    /// 接受概率用 `weight(offset) / weight(0)`（中心最大值）；归一化
    /// 常数由 `integral` 提供，`pdf = weight(offset) / integral`。
    fn sample(&self, uv: Vec2) -> Option<FilterSample> {
        let offset = Vec2::new(
            (2.0 * uv.x - 1.0) * self.radius.x,
            (2.0 * uv.y - 1.0) * self.radius.y,
        );
        let w = self.weight(offset);
        let w_max = self.weight(Vec2::ZERO);
        if w_max <= 0.0 || w <= 0.0 {
            return None;
        }
        // 接受概率 = w / w_max。用 uv 的哈希作为接受判据（单次尝试，
        // 调用方重试）。这个哈希让接受决策与 offset 解耦（offset 来自
        // uv 的线性变换，直接用 uv 会相关）。
        let accept = (uv.x * 12.9898 + uv.y * 78.233).sin().abs();
        if accept > w / w_max {
            return None;
        }
        if self.integral <= 0.0 {
            return None;
        }
        let pdf = w / self.integral;
        Some(FilterSample {
            offset,
            weight: w / pdf,
            pdf,
        })
    }

    fn pdf(&self, offset: Vec2) -> f32 {
        let r = self.radius;
        if offset.x.abs() > r.x || offset.y.abs() > r.y {
            return 0.0;
        }
        if self.integral <= 0.0 {
            return 0.0;
        }
        self.weight(offset) / self.integral
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ─────────── Filter 基础契约 ───────────

    #[test]
    fn box_weight_basics() {
        let f = BoxFilter::new(Vec2::splat(0.5));
        assert_eq!(f.weight(Vec2::ZERO), 1.0);
        assert_eq!(f.weight(Vec2::splat(0.5)), 1.0); // 边界包含
        assert_eq!(f.weight(Vec2::splat(0.51)), 0.0);
    }

    #[test]
    fn triangle_weight_basics() {
        let f = TriangleFilter::new(Vec2::splat(1.0));
        assert!(approx_eq(f.weight(Vec2::ZERO), 1.0, 1e-6));
        assert!(approx_eq(f.weight(Vec2::new(0.5, 0.0)), 0.5, 1e-6));
        assert!(approx_eq(f.weight(Vec2::new(0.5, 0.5)), 0.25, 1e-6));
        assert!(approx_eq(f.weight(Vec2::splat(1.0)), 0.0, 1e-6));
    }

    #[test]
    fn gaussian_weight_basics() {
        let f = GaussianFilter::new(Vec2::splat(1.0), 0.5);
        assert!(f.weight(Vec2::ZERO) > 0.0);
        assert!(approx_eq(f.weight(Vec2::splat(1.0)), 0.0, 1e-6));
        assert_eq!(f.weight(Vec2::splat(1.5)), 0.0);
    }

    #[test]
    fn radius_query_returns_construction_value() {
        let r = Vec2::new(1.5, 0.8);
        assert_eq!(BoxFilter::new(r).radius(), r);
        assert_eq!(TriangleFilter::new(r).radius(), r);
        assert_eq!(GaussianFilter::new(r, 0.5).radius(), r);
    }

    #[test]
    fn all_filters_zero_outside_radius() {
        let r = Vec2::splat(0.5);
        let outside = Vec2::splat(1.0);
        assert_eq!(BoxFilter::new(r).weight(outside), 0.0);
        assert_eq!(TriangleFilter::new(r).weight(outside), 0.0);
        assert_eq!(GaussianFilter::new(r, 0.3).weight(outside), 0.0);
    }

    #[test]
    fn filter_is_object_safe() {
        let _: Box<dyn Filter> = Box::new(BoxFilter::new(Vec2::splat(1.0)));
        let _: Box<dyn Filter> = Box::new(TriangleFilter::new(Vec2::splat(1.0)));
        let _: Box<dyn Filter> = Box::new(GaussianFilter::new(Vec2::splat(1.0), 0.5));
    }

    // ─────────── SamplableFilter 契约：一致性 ───────────

    fn check_consistency<F: SamplableFilter>(f: &F, uv: Vec2) {
        if let Some(s) = f.sample(uv) {
            // sample / pdf 一致性
            let pdf_at_offset = f.pdf(s.offset);
            assert!(
                approx_eq(s.pdf, pdf_at_offset, 1e-5),
                "pdf mismatch: s.pdf={}, pdf(offset)={}, offset={:?}",
                s.pdf,
                pdf_at_offset,
                s.offset
            );
            // weight / sample 一致性
            let expected = f.weight(s.offset) / s.pdf;
            assert!(
                approx_eq(s.weight, expected, 1e-5),
                "weight mismatch: s.weight={}, expected={}",
                s.weight,
                expected
            );
            // 采样域
            let r = f.radius();
            assert!(s.offset.x.abs() <= r.x + 1e-6);
            assert!(s.offset.y.abs() <= r.y + 1e-6);
        }
    }

    #[test]
    fn box_sample_pdf_consistent() {
        let f = BoxFilter::new(Vec2::splat(1.0));
        for u in [[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]] {
            check_consistency(&f, Vec2::from(u));
        }
    }

    #[test]
    fn triangle_sample_pdf_consistent() {
        let f = TriangleFilter::new(Vec2::splat(1.0));
        for i in 1..20 {
            for j in 1..20 {
                check_consistency(&f, Vec2::new(i as f32 / 20.0, j as f32 / 20.0));
            }
        }
    }

    #[test]
    fn gaussian_sample_pdf_consistent() {
        let f = GaussianFilter::new(Vec2::splat(1.0), 0.5);
        let mut checked = 0;
        for i in 1..20 {
            for j in 1..20 {
                let uv = Vec2::new(i as f32 / 20.0, j as f32 / 20.0);
                if f.sample(uv).is_some() {
                    check_consistency(&f, uv);
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no samples accepted");
    }

    // ─────────── PDF 数值积分 ───────────

    fn integrate_pdf<F: SamplableFilter>(f: &F, n: usize) -> f32 {
        let r = f.radius();
        let dx = 2.0 * r.x / n as f32;
        let dy = 2.0 * r.y / n as f32;
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                let x = -r.x + (i as f32 + 0.5) * dx;
                let y = -r.y + (j as f32 + 0.5) * dy;
                sum += f.pdf(Vec2::new(x, y));
            }
        }
        sum * dx * dy
    }

    #[test]
    fn box_pdf_integrates_to_one() {
        let f = BoxFilter::new(Vec2::splat(1.0));
        let integral = integrate_pdf(&f, 100);
        assert!(approx_eq(integral, 1.0, 1e-3), "integral = {integral}");
    }

    #[test]
    fn triangle_pdf_integrates_to_one() {
        let f = TriangleFilter::new(Vec2::splat(1.0));
        let integral = integrate_pdf(&f, 200);
        assert!(approx_eq(integral, 1.0, 1e-2), "integral = {integral}");
    }

    #[test]
    fn gaussian_pdf_integrates_to_one() {
        let f = GaussianFilter::new(Vec2::splat(1.0), 0.5);
        let integral = integrate_pdf(&f, 200);
        assert!(approx_eq(integral, 1.0, 2e-2), "integral = {integral}");
    }

    // ─────────── 采样分布特征 ───────────

    #[test]
    fn triangle_sample_concentrates_near_center() {
        let f = TriangleFilter::new(Vec2::splat(1.0));
        let mut near_center = 0;
        let n = 100;
        for i in 0..n {
            let u = Vec2::new((i as f32 + 0.5) / n as f32, (i as f32 + 0.5) / n as f32);
            let s = f.sample(u).unwrap();
            if s.offset.length() < 0.5 {
                near_center += 1;
            }
        }
        // 三角分布比均匀分布更集中中心
        assert!(near_center > 40, "near_center = {near_center}");
    }

    // ─────────── Send + Sync / object-safety ───────────

    #[test]
    fn all_filters_impl_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BoxFilter>();
        assert_send_sync::<TriangleFilter>();
        assert_send_sync::<GaussianFilter>();
    }

    #[test]
    fn samplable_filter_is_object_safe() {
        let _: Box<dyn SamplableFilter> = Box::new(BoxFilter::new(Vec2::splat(1.0)));
        let _: Box<dyn SamplableFilter> = Box::new(TriangleFilter::new(Vec2::splat(1.0)));
        let _: Box<dyn SamplableFilter> = Box::new(GaussianFilter::new(Vec2::splat(1.0), 0.5));
    }

    #[test]
    fn samplable_filter_upcasts_to_filter() {
        // SamplableFilter: Filter，trait upcasting（Rust 1.86+）
        let s: Box<dyn SamplableFilter> = Box::new(TriangleFilter::new(Vec2::splat(1.0)));
        let _: Box<dyn Filter> = s;
    }

    // ─────────── 构造校验 ───────────

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn gaussian_rejects_zero_sigma() {
        GaussianFilter::new(Vec2::splat(1.0), 0.0);
    }

    #[test]
    #[should_panic(expected = "radius must be positive")]
    fn gaussian_rejects_zero_radius() {
        GaussianFilter::new(Vec2::ZERO, 0.5);
    }
}
