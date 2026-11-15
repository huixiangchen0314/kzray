//! 标量代数与光谱代数：`ScalarAlgebra` / `SpectralAlgebra` trait 定义与所有实现。
//!
//! 分层：
//!
//! - [`ScalarAlgebra`] :: 纯算术。只保证逐分量加、减、乘、数乘、零元、幺元；
//!   不承担波长、色彩空间、物理量纲等语义。
//! - [`SpectralAlgebra`] :: 光谱表示的波长上下文。用关联类型
//!   [`SpectralAlgebra::WavelengthContext`] 把“光谱类型”和“配套波长上下文”
//!   在类型层面强绑定，N 的一致性和波长语义由编译器保证。
//!
//! 实现：
//!
//! - `f32` :: 单标量，只实现 [`ScalarAlgebra`]；不是光谱表示，不实现
//!   [`SpectralAlgebra`]。
//! - [`Rgb`] :: 三通道 RGB，隐含固定色彩空间；波长上下文为 [`NoWavelengths`]，
//!   零大小，零运行时成本。
//! - [`Spectral<N>`] :: N 个采样波长上的能量值；波长上下文为
//!   [`SampledWavelengths<N>`]，N 强绑定。`spectral` feature 启用后编译。

use glam::Vec3;

// ═══════════════════════════════════════════════════════════════
// ScalarAlgebra：纯算术
// ═══════════════════════════════════════════════════════════════

/// 标量域上的算术。
///
/// 只保证逐分量加、减、乘、数乘、零元、幺元；不承担波长、
/// 色彩空间、物理量纲等语义。物理语义由上下文承担。
pub trait ScalarAlgebra: Copy + Send + Sync + 'static {
    /// 标量域。
    ///
    /// `From<f32>` 是渲染器的合理假设：所有标量域都能从 `f32` 构造
    /// （`f32` 自反，`f64: From<f32>` 标准库提供，半精度 crate 通常也实现）。
    type Scalar: Copy + From<f32>;

    /// 加法单位元（零元）。不是乘法单位元，见 [`Self::one`]。
    fn zero() -> Self;

    /// 乘法单位元（幺元）。
    ///
    /// 物理上对应「透射率 1 / 常数谱 1」，不是「能量 1」。
    fn one() -> Self;

    /// 交换、结合。
    fn add(&self, other: &Self) -> Self;

    /// `add` 的逆。存在是因为运算承认负值。
    fn neg(&self) -> Self;

    /// 逐分量（Hadamard）乘：交换、结合、双线性。
    fn mul(&self, other: &Self) -> Self;

    /// 标量乘，与 `add` / `mul` 满足分配律。
    fn scale(&self, k: Self::Scalar) -> Self;

   /// 分量绝对值的最大值，返回 `f32`。
    ///
    /// 非范数、非公理、非代数结构的一部分。用于**运行时控制流**——
    /// RR 存活概率、早退优化。这些场景需要与 `f32` 阈值比较、需要
    /// 转 `f32` 做概率采样，`Self::Scalar` 的 `Copy + From<f32>` 不够。
    ///
    /// 返回 `f32` 而非 `Self::Scalar`：消费者只有 `f32` 需求，且所有
    /// 实现（`f32` / `Rgb` / `Spectrum`）内部本来就算出 `f32`。
    /// 让 `Self::Scalar` 只为 `scale` 服务，`max_abs` 独立为 `f32`，
    /// 避免给 `Self::Scalar` 加一堆只为 `max_abs` 用的 bound。
    /// 这是合理的工程简化
    fn max_abs(&self) -> f32;

    /// 由 `add` + `neg` 导出，非公理，仅为调用便利。
    fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// 线性插值：`a + (b - a) * t`。
    ///
    /// **由代数定义导出**——`add` / `sub` / `scale` 组合，不引入新公理。
    /// 默认实现，所有 `ScalarAlgebra` 实现自动获得。`t` 通过
    /// `S::Scalar: From<f32>` 转换到标量域。
    #[inline]
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        a.add(&b.sub(&a).scale(t.into()))
    }

}


// ═══════════════════════════════════════════════════════════════
// SpectralAlgebra：波长上下文
// ═══════════════════════════════════════════════════════════════

/// 需要波长上下文的光谱表示。
///
/// [`ScalarAlgebra`] 只保证算术；本 trait 表达“这个光谱表示配套什么样
/// 的波长上下文”。关联类型 [`Self::WavelengthContext`] 让
/// `Spectral<N>` 的分量数和 `SampledWavelengths<N>` 的波长数在类型
/// 层面强绑定，编译器保证一致。
///
/// 只在需要波长的地方用（光源采样、BSDF 求值、光谱到颜色的转换）；
/// 只做算术累积的地方（throughput、权重）约束 [`ScalarAlgebra`] 即可。
pub trait SpectralAlgebra: ScalarAlgebra {
    /// 配套的波长上下文。
    ///
    /// - [`Rgb`] :: [`NoWavelengths`]，零大小，隐含固定色彩空间，
    ///   不需要运行时波长。
    /// - `Spectral<N>` :: `SampledWavelengths<N>`，分量数和 N 强绑定。
    type WavelengthContext: Copy + Send + Sync + 'static;
}

// ═══════════════════════════════════════════════════════════════
// 波长上下文类型
// ═══════════════════════════════════════════════════════════════

/// `Rgb` 的波长上下文：零大小类型，编译器优化掉。
///
/// `Rgb` 隐含固定色彩空间，采样、求值都不需要运行时波长；
/// 函数签名要求 `&S::WavelengthContext` 时传 `&NoWavelengths` 即可，
/// 运行时零成本。
#[derive(Clone, Copy, Debug, Default)]
pub struct NoWavelengths;

/// N 个采样波长的上下文。
///
/// - `wavelengths` :: N 个具体波长（纳米）。
/// - `pdfs` :: 每个波长的采样 PDF，用于光谱到 XYZ/RGB 的积分权重。
///
/// 不塞进 `Spectral<N>`，因为波长每样本不同；`Spectral<N>` 只持有
/// 能量值，保持 `Copy`。波长值通过 `&SampledWavelengths<N>` 传递，
/// 由 `SpectralAlgebra::WavelengthContext` 保证和 `Spectral<N>` 配套。
#[derive(Clone, Copy, Debug)]
pub struct SampledWavelengths<const N: usize> {
    pub wavelengths: [f32; N],
    pub pdfs: [f32; N],
}

impl<const N: usize> SampledWavelengths<N> {
    pub const fn new(wavelengths: [f32; N], pdfs: [f32; N]) -> Self {
        SampledWavelengths { wavelengths, pdfs }
    }
}

// ═══════════════════════════════════════════════════════════════
// 实现：f32（只实现 ScalarAlgebra）
// ═══════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════
// 实现：Rgb
// ═══════════════════════════════════════════════════════════════

/// 三通道 RGB。隐含固定色彩空间，不需要运行时波长。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb(pub [f32; 3]);

impl Rgb {
    pub const ZERO: Rgb = Rgb([0.0; 3]);
    pub const ONE: Rgb = Rgb([1.0; 3]);

    pub const fn new(r: f32, g: f32, b: f32) -> Self { Rgb([r, g, b]) }
    pub const fn from_array(a: [f32; 3]) -> Self { Rgb(a) }
    pub const fn to_array(self) -> [f32; 3] { self.0 }
    pub fn from_vec3(v: Vec3) -> Self { Rgb([v.x, v.y, v.z]) }
    pub fn to_vec3(self) -> Vec3 { Vec3::from_array(self.0) }
}

impl ScalarAlgebra for Rgb {
    type Scalar = f32;

    #[inline]
    fn zero() -> Self { Rgb::ZERO }

    #[inline]
    fn one() -> Self { Rgb::ONE }

    #[inline]
    fn add(&self, o: &Self) -> Self {
        Rgb([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]])
    }

    #[inline]
    fn neg(&self) -> Self {
        Rgb([-self.0[0], -self.0[1], -self.0[2]])
    }

    #[inline]
    fn mul(&self, o: &Self) -> Self {
        Rgb([self.0[0] * o.0[0], self.0[1] * o.0[1], self.0[2] * o.0[2]])
    }

    #[inline]
    fn scale(&self, k: f32) -> Self {
        Rgb([self.0[0] * k, self.0[1] * k, self.0[2] * k])
    }

    #[inline]
    fn max_abs(&self) -> f32 {
        self.0[0].abs().max(self.0[1].abs()).max(self.0[2].abs())
    }
}

impl SpectralAlgebra for Rgb {
    /// `Rgb` 隐含固定色彩空间，不需要运行时波长。
    type WavelengthContext = NoWavelengths;
}

// ═══════════════════════════════════════════════════════════════
// 实现：Spectral<N>（`spectral` feature）
// ═══════════════════════════════════════════════════════════════

/// 定长采样光谱能量，`N` 为采样点数（编译期确定）。
///
/// `mul` 为逐采样点乘，**不是**光谱内积；内积不是双线性代数的 `mul`。
/// 波长采样点的具体位置不属于本类型——若需要，由上层以常量或
/// 外部表提供，不要塞进 `Spectral` 破坏 `Copy`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spectrum<const N: usize>(pub [f32; N]);

impl<const N: usize> Spectrum<N> {
    pub const fn new(vals: [f32; N]) -> Self { Spectrum(vals) }
    pub const fn to_array(self) -> [f32; N] { self.0 }
}

impl<const N: usize> ScalarAlgebra for Spectrum<N> {
    type Scalar = f32;

    #[inline]
    fn zero() -> Self { Spectrum([0.0; N]) }

    #[inline]
    fn one() -> Self { Spectrum([1.0; N]) }

    #[inline]
    fn add(&self, o: &Self) -> Self {
        Spectrum(std::array::from_fn(|i| self.0[i] + o.0[i]))
    }

    #[inline]
    fn neg(&self) -> Self {
        Spectrum(std::array::from_fn(|i| -self.0[i]))
    }

    #[inline]
    fn mul(&self, o: &Self) -> Self {
        Spectrum(std::array::from_fn(|i| self.0[i] * o.0[i]))
    }

    #[inline]
    fn scale(&self, k: f32) -> Self {
        Spectrum(std::array::from_fn(|i| self.0[i] * k))
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

impl<const N: usize> SpectralAlgebra for Spectrum<N> {
    /// 分量数和 `SampledWavelengths<N>` 的波长数在类型层面强绑定。
    type WavelengthContext = SampledWavelengths<N>;
}

