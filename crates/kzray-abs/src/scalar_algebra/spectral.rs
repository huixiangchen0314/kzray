//! 全光谱实现。启用 `spectral` feature 后编译。

use super::ScalarAlgebra;

/// 定长采样光谱能量，`N` 为采样点数（编译期确定）。
///
/// `mul` 为逐采样点乘，**不是**光谱内积；内积不是双线性代数的 `mul`。
/// 波长采样点的具体位置不属于本类型——若需要，由上层以常量或
/// 外部表提供，不要塞进 `Spectral` 破坏 `Copy`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spectral<const N: usize>(pub [f32; N]);

impl<const N: usize> Spectral<N> {
    pub const fn new(vals: [f32; N]) -> Self { Spectral(vals) }
    pub const fn to_array(self) -> [f32; N] { self.0 }
}

impl<const N: usize> ScalarAlgebra for Spectral<N> {
    type Scalar = f32;

    #[inline]
    fn zero() -> Self { Spectral([0.0; N]) }

    #[inline]
    fn one() -> Self { Spectral([1.0; N]) }

    #[inline]
    fn add(&self, o: &Self) -> Self {
        Spectral(std::array::from_fn(|i| self.0[i] + o.0[i]))
    }

    #[inline]
    fn neg(&self) -> Self {
        Spectral(std::array::from_fn(|i| -self.0[i]))
    }

    #[inline]
    fn mul(&self, o: &Self) -> Self {
        Spectral(std::array::from_fn(|i| self.0[i] * o.0[i]))
    }

    #[inline]
    fn scale(&self, k: f32) -> Self {
        Spectral(std::array::from_fn(|i| self.0[i] * k))
    }

    /// 存在 NaN 分量时返回 NaN，保证保守上界语义。
    /// 注：`f32::max` 会忽略 NaN，故显式传播。
    #[inline]
    fn max_abs(&self) -> f32 {
        self.0.iter().fold(0.0_f32, |m, &x| {
            let a = x.abs();
            if a.is_nan() || m.is_nan() { f32::NAN } else { m.max(a) }
        })
    }
}
