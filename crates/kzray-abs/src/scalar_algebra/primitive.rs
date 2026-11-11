use super::ScalarAlgebra;

impl ScalarAlgebra for f32 {
    type Scalar = f32;

    #[inline]
    fn zero() -> Self { 0.0 }

    #[inline]
    fn one() -> Self { 1.0 }

    #[inline]
    fn add(&self, other: &Self) -> Self { *self + *other }

    #[inline]
    fn neg(&self) -> Self { -*self }

    #[inline]
    fn mul(&self, other: &Self) -> Self { *self * *other }

    #[inline]
    fn scale(&self, k: f32) -> Self { *self * k }

    /// `NaN.abs() == NaN`，故含 `NaN` 时返回 `NaN`；
    /// 调用方用 `max_abs() < eps` 早退时 `NaN` 不会通过，行为安全。
    #[inline]
    fn max_abs(&self) -> f32 { self.abs() }
}
