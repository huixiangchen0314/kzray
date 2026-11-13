//! 像素传感器：光谱值 `S` → 输出 RGB 的转换契约。
//!
//! `PixelSensor` 负责把 [`SpectralAlgebra`] 值转换成
//! [`kzray_data::color::Rgb`]（纯数据，用于图像存储）。它是**胶片
//! resolve 阶段的组件**，不是渲染主循环的一部分——每个像素只调用
//! 一次，不在热路径。
//!
//! # 职责边界
//!
//! 本模块**只负责 S → 输出 RGB 的转换**。**不做**：
//!
//! - 曝光调整、tone mapping、白平衡——图像 I/O 或后期处理的职责；
//! - sRGB gamma 编码——由调用方写出 8 位图像时用
//!   [`kzray_data::color::Rgb::to_srgb_u8`] 完成；
//! - 图像存储——[`crate::film::Film`] 的职责。
//!
//! # 与 `kzray_math::color` 的关系
//!
//! 颜色数学（`spectral_to_xyz` / `WhitePoint` / `SpectralSamples`）在
//! `kzray_math::color`。本模块是**薄包装**：把 `SpectralAlgebra` 的
//! 具体类型桥接到 `kzray_math::color` 的工具上，输出
//! `kzray_data::color::Rgb`。
//!
//! # 为什么 trait 带 `S` 泛型参数
//!
//! `PixelSensor<S>` 的 `S` 是**光谱类型**，不是方法的泛型参数。这样：
//!
//! - 每种 `(sensor 实现, S 类型)` 组合是一个独立 `impl`，可以访问
//!   `S` 的具体字段（`Rgb::0`、`Spectrum::0`），不需要 `spectrum` 提供
//!   泛化的"光谱访问接口"；
//! - `RgbSensor` 只对 `Rgb` 实现——传 `Spectrum<N>` 直接**编译错**，
//!   类型系统阻止错误配置；
//! - `process` 不是泛型方法，`PixelSensor<S>` 是 object-safe 的，
//!   `Box<dyn PixelSensor<S>>` 可用。
//!
//! 这是"用类型系统代替运行时分派"的直接体现：不需要 `SpectralData`
//! 枚举、不需要 `spectral_data` 访问方法，`spectrum` 保持纯光谱代数。
//!
//! # 未实现的功能（TODO）
//!
//! 当前 [`CieSensor`] 对 `Spectrum<N>` 的处理**假设输入是覆盖全谱的
//! 稠密采样**，且值已按波长采样 PDF 加权（或无需加权）。这覆盖了
//! RGB 渲染模式和"稠密光谱"渲染模式，但**不覆盖**每路径随机采样
//! 少量波长的稀疏光谱渲染。
//!
//! TODO(sparse-spectrum-sensor): 稀疏采样的正确做法是 Monte Carlo
//! 估计 `E[value(λ) × CIE(λ) / pdf(λ)]`，需要对每个采样波长 `λ_i`
//! 查 CIE 匹配函数 `x̄(λ_i)` / `ȳ(λ_i)` / `z̄(λ_i)`，乘 `value_i / pdf_i`
//! 求和。实现需要：
//!
//! 1. `kzray_math::color` 暴露 CIE 1931 匹配函数查询接口
//!    （`cie_xyz(wavelength: f32) -> [f32; 3]`），当前 `spectral_to_xyz`
//!    假设稠密光谱、内部走 `colorimetry` 的积分；
//! 2. `CieSensor::process` 对 `Spectrum<N>` 走 MC 路径，用
//!    `SampledWavelengths::pdfs` 加权；
//! 3. 处理 `N` 小时方差大的问题（可能需要多波长或分层采样）。
//!
//! 当前实现不做 MC 估计，直接用 `spectral_to_xyz` 的稠密积分。对
//! `N` 较大（如 32+）且波长覆盖全谱的情况，误差可接受；对 `N` 小
//! 的稀疏采样，结果有偏。文档明确这一限制，实现层选择 `N` 时需注意。
//!
//! TODO(spectral-film-output): 当前 `Film` 的 `Output` 由实现层决定；
//! `SpectralFilm` 输出原始光谱数据时不需要 `PixelSensor`，但当前没有
//! 光谱输出类型（`SpectralImage`）。等 `Film` 实现层需要光谱输出时，
//! 在 `kzray-data` 加 `SpectralImage` 类型，无需改本模块。

use kzray_data::Rgb;
use kzray_math::color::{spectral_to_xyz, SpectralSamples, WhitePoint};

use crate::spectrum::{
    NoWavelengths, Rgb as SpectrumRgb, SampledWavelengths, SpectralAlgebra, Spectrum,
};

// ═══════════════════════════════════════════════════════════════
// PixelSensor trait
// ═══════════════════════════════════════════════════════════════

/// 像素传感器契约：光谱值 `S` → 输出 `Rgb`。
///
/// `S` 是 trait 参数，`process` 不是泛型方法——每种 `(sensor, S)` 组合
/// 是独立 `impl`，可以访问 `S` 的具体字段。
///
/// 所有实现必须 `Send + Sync`：Sensor 在构造层创建，被 Film 持有，
/// 可能跨线程访问。
///
/// `PixelSensor<S>` 是 object-safe 的：`process` 不泛型、没有 `Self`
/// 的 `Sized` 要求之外的特殊约束，`Box<dyn PixelSensor<S>>` 可用。
pub trait PixelSensor<S: SpectralAlgebra>: Send + Sync {
    /// 把光谱值 `S` 处理成输出 RGB。
    ///
    /// 返回的是**线性 RGB**，无 gamma 编码。调用方写出 8 位图像时
    /// 用 [`Rgb::to_srgb_u8`] 做 sRGB 编码。
    ///
    /// `wavelengths` 是采样波长上下文：
    ///
    /// - 对 `Rgb` 类型（[`NoWavelengths`]），被忽略；
    /// - 对 `Spectrum<N>` 类型（[`SampledWavelengths<N>`]），用于
    ///   光谱积分。
    fn process(&self, value: &S, wavelengths: &S::WavelengthContext) -> Rgb;
}

// ═══════════════════════════════════════════════════════════════
// RgbSensor：仅处理 Rgb
// ═══════════════════════════════════════════════════════════════

/// 直通传感器：仅处理 [`crate::spectrum::Rgb`]。
///
/// 用于 RGB 渲染模式。`Rgb` 直接转成输出 `Rgb`，不做任何颜色科学计算。
///
/// **不对 `Spectrum<N>` 实现 `PixelSensor`**：RGB 渲染模式下 `S = Rgb`，
/// 如果代码里出现 `Spectrum<N>`，是配置错误——传 `Spectrum<N>` 给
/// `RgbSensor::process` 会在**编译期**报错，而不是运行时静默错误。
///
/// 典型用法：
///
/// ```text
/// // RGB 渲染模式
/// let sensor = RgbSensor;
/// let color = sensor.process(&rgb_value, &NoWavelengths);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct RgbSensor;

impl PixelSensor<SpectrumRgb> for RgbSensor {
    #[inline]
    fn process(&self, value: &SpectrumRgb, _wavelengths: &NoWavelengths) -> Rgb {
        Rgb::new(value.0[0], value.0[1], value.0[2])
    }
}

// ═══════════════════════════════════════════════════════════════
// CieSensor：Rgb 直通 + Spectrum<N> 积分
// ═══════════════════════════════════════════════════════════════

/// CIE 传感器：用 CIE 匹配函数把光谱积分成 XYZ，再转线性 RGB。
///
/// 对两种 `S` 都实现 `PixelSensor`：
///
/// - `Rgb`：直接透传（RGB 值本身已经是颜色，不需要 CIE 积分）；
/// - `Spectrum<N>`：用 [`kzray_math::color::spectral_to_xyz`] 做积分。
///
/// # 白点
///
/// 构造时可以指定参考白点（[`WhitePoint::D65`] / [`WhitePoint::D50`]）。
/// 光谱积分的输出是 `target` 白点参考下的 XYZ，`to_lin_srgb` 会自动
/// 适配回 D65（`LinSrgb` 原色固定 D65），因此**最终 RGB 与白点选择
/// 无关**（物理正确）。白点只影响中间表示。
///
/// 默认 `D65`，与标准 sRGB 显示一致。
///
/// # 稀疏采样限制（TODO）
///
/// 对 `Spectrum<N>` 的处理**假设输入是覆盖全谱的稠密采样**，且值已按
/// 波长采样 PDF 加权（或无需加权）。`N` 为小值、波长是每路径随机采样
/// 的稀疏点时，直接调用会有偏差——正确做法是 Monte Carlo 估计
/// `E[value × CIE / pdf]`。详见模块文档"未实现的功能（TODO）"。
///
/// # 典型用法
///
/// ```text
/// // 光谱渲染模式
/// let sensor = CieSensor::new();  // D65
/// let color = sensor.process(&spectrum_value, &sampled_wavelengths);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CieSensor {
    /// 参考白点。仅影响中间 XYZ 表示，不影响最终输出 RGB。
    white: WhitePoint,
}

impl CieSensor {
    /// 构造 D65 白点的 CIE 传感器。与标准 sRGB 显示一致。
    pub fn new() -> Self {
        CieSensor {
            white: WhitePoint::D65,
        }
    }

    /// 构造指定白点的 CIE 传感器。
    ///
    /// 白点只影响光谱积分的中间 XYZ 表示；`to_lin_srgb` 会自动适配
    /// 回 D65，最终 RGB 与白点选择无关。
    pub fn with_white(white: WhitePoint) -> Self {
        CieSensor { white }
    }

    /// 当前白点。
    pub fn white(&self) -> WhitePoint {
        self.white
    }
}

impl Default for CieSensor {
    fn default() -> Self {
        Self::new()
    }
}

impl PixelSensor<SpectrumRgb> for CieSensor {
    #[inline]
    fn process(&self, value: &SpectrumRgb, _wavelengths: &NoWavelengths) -> Rgb {
        // RGB 值本身已是颜色，不需要 CIE 积分。
        Rgb::new(value.0[0], value.0[1], value.0[2])
    }
}

impl<const N: usize> PixelSensor<Spectrum<N>> for CieSensor {
    fn process(&self, value: &Spectrum<N>, wavelengths: &SampledWavelengths<N>) -> Rgb {
        // TODO(sparse-spectrum-sensor): 当前走稠密光谱积分路径
        // （spectral_to_xyz 内部假设输入覆盖全谱）。稀疏采样
        // （N 小、波长每路径随机）时结果有偏，需要 MC 估计。
        // 详见模块文档"未实现的功能（TODO）"。
        //
        // 当前实现**不使用** `wavelengths.pdfs`——spectral_to_xyz 的
        // API 假设输入已是稠密光谱或已加权。如果调用方传入了 pdfs
        // 非均匀的稀疏采样，这里会忽略它们。
        let samples = SpectralSamples::new(
            wavelengths.wavelengths.to_vec(),
            value.0.to_vec(),
        );
        let lin = spectral_to_xyz(&samples, self.white).to_lin_srgb();
        Rgb::new(lin.red, lin.green, lin.blue)
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────── RgbSensor ───────────

    #[test]
    fn rgb_sensor_passes_through() {
        let sensor = RgbSensor;
        let value = SpectrumRgb::new(0.25, 0.5, 0.75);
        let result = sensor.process(&value, &NoWavelengths);
        assert_eq!(result, Rgb::new(0.25, 0.5, 0.75));
    }

    #[test]
    fn rgb_sensor_preserves_negative_and_over_one() {
        let sensor = RgbSensor;
        let value = SpectrumRgb::new(-0.5, 1.5, 2.0);
        let result = sensor.process(&value, &NoWavelengths);
        assert_eq!(result, Rgb::new(-0.5, 1.5, 2.0));
    }

    // RgbSensor 不实现 PixelSensor<Spectrum<N>>，传 Spectrum 是编译错。
    // 不需要运行时测试——类型系统保证。

    // ─────────── CieSensor + Rgb ───────────

    #[test]
    fn cie_sensor_passes_through_rgb() {
        let sensor = CieSensor::new();
        let value = SpectrumRgb::new(0.25, 0.5, 0.75);
        let result = sensor.process(&value, &NoWavelengths);
        assert_eq!(result, Rgb::new(0.25, 0.5, 0.75));
    }

    #[test]
    fn cie_sensor_preserves_negative_and_over_one_for_rgb() {
        let sensor = CieSensor::new();
        let value = SpectrumRgb::new(-0.5, 1.5, 2.0);
        let result = sensor.process(&value, &NoWavelengths);
        assert_eq!(result, Rgb::new(-0.5, 1.5, 2.0));
    }

    // ─────────── CieSensor + Spectrum<N> ───────────

    #[test]
    fn cie_sensor_spectrum_returns_finite() {
        let sensor = CieSensor::new();
        let wavelengths = [
            400.0, 430.0, 460.0, 490.0, 520.0, 550.0, 580.0, 610.0, 640.0, 670.0,
        ];
        let values = [1.0; 10];
        let value = Spectrum::new(values);
        let wl = SampledWavelengths::new(wavelengths, [1.0 / 10.0; 10]);
        let result = sensor.process(&value, &wl);
        assert!(result.r.is_finite());
        assert!(result.g.is_finite());
        assert!(result.b.is_finite());
    }

    #[test]
    fn cie_sensor_white_spectrum_is_roughly_white() {
        let sensor = CieSensor::new();
        let wavelengths = [
            400.0, 430.0, 460.0, 490.0, 520.0, 550.0, 580.0, 610.0, 640.0, 670.0,
        ];
        let values = [1.0; 10];
        let value = Spectrum::new(values);
        let wl = SampledWavelengths::new(wavelengths, [1.0 / 10.0; 10]);
        let result = sensor.process(&value, &wl);
        let max = result.r.max(result.g).max(result.b);
        let min = result.r.min(result.g).min(result.b);
        assert!(
            max <= min * 3.0 + 0.01,
            "expected roughly neutral white, got {:?}",
            result
        );
    }

    #[test]
    fn cie_sensor_default_is_d65() {
        let sensor = CieSensor::default();
        assert_eq!(sensor.white(), WhitePoint::D65);
    }

    #[test]
    fn cie_sensor_with_white() {
        let sensor = CieSensor::with_white(WhitePoint::D50);
        assert_eq!(sensor.white(), WhitePoint::D50);
    }

    #[test]
    fn cie_sensor_different_whites_same_rgb_output() {
        let d65 = CieSensor::new();
        let d50 = CieSensor::with_white(WhitePoint::D50);
        let wavelengths = [
            400.0, 430.0, 460.0, 490.0, 520.0, 550.0, 580.0, 610.0, 640.0, 670.0,
        ];
        let values = [1.0; 10];
        let value = Spectrum::new(values);
        let wl = SampledWavelengths::new(wavelengths, [1.0 / 10.0; 10]);
        let r65 = d65.process(&value, &wl);
        let r50 = d50.process(&value, &wl);
        assert!((r65.r - r50.r).abs() < 1e-3, "r: {:?} vs {:?}", r65, r50);
        assert!((r65.g - r50.g).abs() < 1e-3, "g: {:?} vs {:?}", r65, r50);
        assert!((r65.b - r50.b).abs() < 1e-3, "b: {:?} vs {:?}", r65, r50);
    }

    // ─────────── 通用契约 ───────────

    #[test]
    fn sensors_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RgbSensor>();
        assert_send_sync::<CieSensor>();
    }

    #[test]
    fn pixel_sensor_is_object_safe() {
        let _: Box<dyn PixelSensor<SpectrumRgb>> = Box::new(RgbSensor);
        let _: Box<dyn PixelSensor<SpectrumRgb>> = Box::new(CieSensor::new());
    }
}
