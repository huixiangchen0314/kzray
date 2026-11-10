//! 离散概率分布。用于光源选择、环境贴图重要性采样。
//!
//! - `Distribution1D`：一维离散分布，给定权重，构造 CDF，提供离散/连续采样。
//! - `Distribution2D`：二维离散分布，行方向 + 行内列方向两级采样。
//!
//! **精度**：CDF 用 `f64` 累积，权重 / PDF 用 `f32`。
//! **错误处理**：空 / NaN / Inf / 负权重 / 全零 → `panic!`（上游 bug，不可恢复）。
//! **边界**：`sample_discrete(u)` 的 `u` clamp 到 `[0, 1-ε]`，避免 `u=1` 时越界。
//! **序列化**：`into_raw` / `from_raw` 提供 `(weights, cdf, sum)` 的往返。

// ── Distribution1D ──

#[derive(Clone, Debug)]
pub struct Distribution1D {
    weights: Vec<f32>,
    cdf: Vec<f64>,   // 长度 = weights.len() + 1
    sum: f64,
}

impl Distribution1D {
    /// 从权重构造。空、含 NaN/Inf、含负权重、全零 → panic。
    pub fn new(weights: &[f32]) -> Self {
        assert!(!weights.is_empty(), "Distribution1D 不能为空");
        let mut sum = 0.0f64;
        let mut cdf = Vec::with_capacity(weights.len() + 1);
        cdf.push(0.0);
        for (i, &w) in weights.iter().enumerate() {
            assert!(
                w.is_finite() && w >= 0.0,
                "权重[{i}] = {w} 非有限或为负"
            );
            sum += w as f64;
            cdf.push(sum);
        }
        assert!(sum > 0.0, "Distribution1D 至少需要一个正权重");
        Distribution1D { weights: weights.to_vec(), cdf, sum }
    }

    pub fn len(&self) -> usize { self.weights.len() }
    pub fn is_empty(&self) -> bool { self.weights.is_empty() }
    pub fn sum(&self) -> f32 { self.sum as f32 }
    pub fn weights(&self) -> &[f32] { &self.weights }
    pub fn cdf(&self) -> &[f64] { &self.cdf }

    /// 离散采样：`u ∈ [0, 1)`，返回索引和该索引的 PDF。
    pub fn sample_discrete(&self, u: f32) -> (usize, f32) {
        let n = self.weights.len();
        if n == 1 {
            return (0, 1.0);
        }
        let u = u.clamp(0.0, 1.0 - f32::EPSILON);
        let target = u as f64 * self.sum;
        let idx = self
            .cdf
            .partition_point(|&c| c <= target)
            .saturating_sub(1)
            .min(n - 1);
        let pdf = (self.weights[idx] as f64 / self.sum) as f32;
        (idx, pdf)
    }

    /// 某索引的 PDF。
    pub fn pdf_discrete(&self, index: usize) -> f32 {
        assert!(index < self.weights.len(), "index {index} 越界（len {}）", self.weights.len());
        (self.weights[index] as f64 / self.sum) as f32
    }

    /// 连续采样：`u ∈ [0, 1)`，返回 `x ∈ [0, n)` 和密度 `p(x)`。
    pub fn sample_continuous(&self, u: f32) -> (f32, f32) {
        let n = self.weights.len();
        if n == 1 {
            let u = u.clamp(0.0, 1.0);
            return (u, 1.0);
        }
        let u = u.clamp(0.0, 1.0 - f32::EPSILON);
        let target = u as f64 * self.sum;
        let idx = self
            .cdf
            .partition_point(|&c| c <= target)
            .saturating_sub(1)
            .min(n - 1);
        let lo = self.cdf[idx];
        let hi = self.cdf[idx + 1];
        let width = hi - lo;
        let frac = if width > 0.0 {
            ((target - lo) / width) as f32
        } else {
            0.0
        };
        let x = idx as f32 + frac.clamp(0.0, 1.0);
        let pdf = (self.weights[idx] as f64 / self.sum) as f32;
        (x, pdf)
    }

    /// 连续 PDF（关于 x 的密度）。`x ∉ [0, n)` 返回 0。
    pub fn pdf_continuous(&self, x: f32) -> f32 {
        let n = self.weights.len();
        if !(0.0..n as f32).contains(&x) {
            return 0.0;
        }
        let idx = (x as usize).min(n - 1);
        (self.weights[idx] as f64 / self.sum) as f32
    }

    /// 从 raw 权重和 CDF 重建。形状检查，不做权重合理性检查。
    pub fn from_raw(weights: Vec<f32>, cdf: Vec<f64>, sum: f64) -> Self {
        assert!(!weights.is_empty(), "Distribution1D::from_raw 收到空权重");
        assert_eq!(cdf.len(), weights.len() + 1, "cdf 长度必须 = weights 长度 + 1");
        assert!(cdf[0].abs() < 1e-12, "cdf[0] 必须为 0");
        assert!(sum > 0.0, "sum 必须为正");
        assert!(
            (cdf[cdf.len() - 1] - sum).abs() <= 1e-6 * sum.max(1.0),
            "cdf 末项必须等于 sum：cdf_end = {}, sum = {}",
            cdf[cdf.len() - 1],
            sum
        );
        for i in 1..cdf.len() {
            assert!(cdf[i] >= cdf[i - 1], "cdf 必须单调非递减（i={i}）");
        }
        Distribution1D { weights, cdf, sum }
    }

    pub fn into_raw(self) -> (Vec<f32>, Vec<f64>, f64) {
        (self.weights, self.cdf, self.sum)
    }
}

// ── Distribution2D ──

#[derive(Clone, Debug)]
pub struct Distribution2D {
    row_dists: Vec<Distribution1D>,
    marginal: Distribution1D,
    total: f32,
}

impl Distribution2D {
    /// 从二维权重构造。`weights[row][col]`。
    /// 空、行内列数不一致、全零 → panic。
    pub fn new(weights: &[Vec<f32>]) -> Self {
        assert!(!weights.is_empty(), "Distribution2D 不能为空");
        let cols = weights[0].len();
        assert!(cols > 0, "每行不能为空");
        for (i, row) in weights.iter().enumerate() {
            assert_eq!(row.len(), cols, "第 {i} 行列数 {} ≠ {cols}", row.len());
        }

        let row_dists: Vec<Distribution1D> = weights
            .iter()
            .map(|row| Distribution1D::new(row))
            .collect();
        let row_sums: Vec<f32> = row_dists.iter().map(|d| d.sum()).collect();
        let marginal = Distribution1D::new(&row_sums);
        let total = marginal.sum();

        Distribution2D { row_dists, marginal, total }
    }

    pub fn rows(&self) -> usize { self.row_dists.len() }
    pub fn cols(&self) -> usize { self.row_dists[0].len() }
    pub fn total(&self) -> f32 { self.total }
    pub fn row_dists(&self) -> &[Distribution1D] { &self.row_dists }
    pub fn marginal(&self) -> &Distribution1D { &self.marginal }

    /// 离散采样：`u1` 选行，`u2` 选列。返回 `((row, col), p(row,col))`。
    pub fn sample_discrete(&self, u1: f32, u2: f32) -> ((usize, usize), f32) {
        let (row, pdf_row) = self.marginal.sample_discrete(u1);
        let (col, pdf_col) = self.row_dists[row].sample_discrete(u2);
        ((row, col), pdf_row * pdf_col)
    }

    /// 某格子的联合 PDF。
    pub fn pdf_discrete(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.rows(), "row {row} 越界（rows {}）", self.rows());
        assert!(col < self.cols(), "col {col} 越界（cols {}）", self.cols());
        self.marginal.pdf_discrete(row) * self.row_dists[row].pdf_discrete(col)
    }

    /// 连续采样：`u1, u2 ∈ [0, 1)`，返回 `((u, v), p(u,v))`，`u, v ∈ [0, 1)`。
    /// PDF 是**关于 (u, v) 的密度**，`∫∫ p du dv = 1`。
    pub fn sample_continuous(&self, u1: f32, u2: f32) -> ((f32, f32), f32) {
        let rows = self.rows();
        let cols = self.cols();
        let (row_f, _) = self.marginal.sample_continuous(u1);
        let row = (row_f as usize).min(rows - 1);
        let (col_f, _) = self.row_dists[row].sample_continuous(u2);
        let col = (col_f as usize).min(cols - 1);

        let u = col_f / cols as f32;
        let v = row_f / rows as f32;

        let w = self.row_dists[row].weights()[col];
        let pdf = w * rows as f32 * cols as f32 / self.total;
        ((u, v), pdf)
    }

    /// 连续 PDF。`(u, v) ∈ [0, 1)²`。
    pub fn pdf_continuous(&self, u: f32, v: f32) -> f32 {
        let rows = self.rows();
        let cols = self.cols();
        let u = u.clamp(0.0, 1.0 - f32::EPSILON);
        let v = v.clamp(0.0, 1.0 - f32::EPSILON);
        let row = ((v * rows as f32) as usize).min(rows - 1);
        let col = ((u * cols as f32) as usize).min(cols - 1);
        let w = self.row_dists[row].weights()[col];
        w * rows as f32 * cols as f32 / self.total
    }

    pub fn from_raw(row_dists: Vec<Distribution1D>, marginal: Distribution1D) -> Self {
        assert!(!row_dists.is_empty(), "Distribution2D::from_raw 收到空行");
        assert_eq!(
            marginal.len(),
            row_dists.len(),
            "marginal 长度 {} ≠ 行数 {}",
            marginal.len(),
            row_dists.len()
        );
        let total = marginal.sum();
        Distribution2D { row_dists, marginal, total }
    }

    pub fn into_raw(self) -> (Vec<Distribution1D>, Distribution1D) {
        (self.row_dists, self.marginal)
    }
}
