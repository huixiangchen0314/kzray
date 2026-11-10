//! 颜色数学。薄包装 palette / colorimetry / linear-srgb。
//! 公共 API 不出现任何 colorimetry 或 palette 类型。
//!
//! 职责边界：本模块只负责**颜色表示与转换**，不负责曝光、tonemap、HDR 映射。
//! `LinSrgb` 不 clamp，允许负值与 >1 值，钳制责任在下游。
//!
//! 白点语义：
//! - `Xyz` / `Yxy` 携带运行时白点标签（`white()` 可读），表示"在哪个参考白点下的坐标"。
//! - `spectral_to_xyz(samples, target)` 的 `target` 生效：返回 `target` 参考下的 XYZ。
//! - `Xyz::to_lin_srgb()` 自动适配到 D65 再转矩阵（`LinSrgb` 原色固定 D65）。
//! - `spectral_to_lin_srgb(samples)` 不接受白点：最终 LinSrgb 与中间白点无关。

use glam::Vec3;

use palette::white_point::D65;
use palette::IntoColor;
use palette::{LinSrgb as PalLinSrgb, Srgb as PalSrgb, Xyz as PalXyz, Yxy as PalYxy};

pub use linear_srgb::default::{linear_to_srgb, srgb_to_linear};
pub use linear_srgb::default::{linear_to_srgb_slice, srgb_to_linear_slice};

use colorimetry::traits::Light;
use colorimetry::spectrum::NS;

// ── 白点 ──

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhitePoint {
    pub x: f32,
    pub y: f32,
}

impl WhitePoint {
    pub const D65: WhitePoint = WhitePoint { x: 0.3127, y: 0.3290 };
    pub const D50: WhitePoint = WhitePoint { x: 0.3457, y: 0.3585 };

    pub const fn new(x: f32, y: f32) -> Self {
        WhitePoint { x, y }
    }

    pub fn xy(self) -> (f32, f32) {
        (self.x, self.y)
    }
}

impl Default for WhitePoint {
    fn default() -> Self { WhitePoint::D65 }
}

#[inline]
fn wp_eq(a: WhitePoint, b: WhitePoint) -> bool {
    (a.x - b.x).abs() < 1e-6 && (a.y - b.y).abs() < 1e-6
}

// ── 白点适配（Bradford） ──

const BRADFORD: [[f32; 3]; 3] = [
    [0.8951,  0.2664, -0.1614],
    [-0.7502, 1.7135,  0.0367],
    [0.0389, -0.0685,  1.0296],
];

const BRADFORD_INV: [[f32; 3]; 3] = [
    [ 0.9869929, -0.1470543, 0.1599627],
    [ 0.4323053,  0.5183603, 0.0492912],
    [-0.0085287,  0.0400428, 0.9684867],
];

#[inline]
fn mat3_mul_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// 白点 xy 色度 → XYZ（Y = 1）
#[inline]
fn xy_to_xyz_y1(wp: WhitePoint) -> [f32; 3] {
    [wp.x / wp.y, 1.0, (1.0 - wp.x - wp.y) / wp.y]
}

/// Bradford 色适应：把 XYZ 从 `src` 白点参考适配到 `dst` 白点参考。
fn bradford_adapt(xyz: [f32; 3], src: WhitePoint, dst: WhitePoint) -> [f32; 3] {
    let s_cone = mat3_mul_vec(BRADFORD, xy_to_xyz_y1(src));
    let d_cone = mat3_mul_vec(BRADFORD, xy_to_xyz_y1(dst));
    let scale = [
        d_cone[0] / s_cone[0],
        d_cone[1] / s_cone[1],
        d_cone[2] / s_cone[2],
    ];
    let v_cone = mat3_mul_vec(BRADFORD, xyz);
    let adapted = [
        v_cone[0] * scale[0],
        v_cone[1] * scale[1],
        v_cone[2] * scale[2],
    ];
    mat3_mul_vec(BRADFORD_INV, adapted)
}

// ── 颜色类型 ──

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinSrgb {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
}

impl LinSrgb {
    pub const fn new(red: f32, green: f32, blue: f32) -> Self {
        Self { red, green, blue }
    }
    pub const fn from_array(a: [f32; 3]) -> Self {
        Self { red: a[0], green: a[1], blue: a[2] }
    }
    pub const fn to_array(self) -> [f32; 3] {
        [self.red, self.green, self.blue]
    }
    pub fn from_vec3(v: Vec3) -> Self {
        Self::from_array(v.to_array())
    }
    pub fn to_vec3(self) -> Vec3 {
        Vec3::from_array(self.to_array())
    }

    /// 线性 sRGB → CIE XYZ（白点 D65，因为 LinSrgb 原色矩阵固定 D65）。
    pub fn to_xyz(self) -> Xyz {
        let p = PalLinSrgb::new(self.red, self.green, self.blue);
        let x: PalXyz<D65, f32> = p.into_color();
        Xyz { x: x.x, y: x.y, z: x.z, white: WhitePoint::D65 }
    }

    /// 线性 sRGB → 编码 sRGB。
    pub fn to_srgb(self) -> Srgb {
        let p = PalLinSrgb::new(self.red, self.green, self.blue);
        let s: PalSrgb<f32> = p.into_color();
        Srgb { red: s.red, green: s.green, blue: s.blue }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Srgb {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
}

impl Srgb {
    pub const fn new(red: f32, green: f32, blue: f32) -> Self {
        Self { red, green, blue }
    }
    pub const fn from_array(a: [f32; 3]) -> Self {
        Self { red: a[0], green: a[1], blue: a[2] }
    }
    pub const fn to_array(self) -> [f32; 3] {
        [self.red, self.green, self.blue]
    }
    pub fn to_lin_srgb(self) -> LinSrgb {
        let p = PalSrgb::new(self.red, self.green, self.blue);
        let l: PalLinSrgb<f32> = p.into_color();
        LinSrgb { red: l.red, green: l.green, blue: l.blue }
    }
}

/// CIE XYZ，携带参考白点标签。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xyz {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    white: WhitePoint,
}

impl Xyz {
    /// 构造 XYZ，白点默认 D65。
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, white: WhitePoint::D65 }
    }

    /// 构造 XYZ，显式指定参考白点。
    pub const fn new_with_white(x: f32, y: f32, z: f32, white: WhitePoint) -> Self {
        Self { x, y, z, white }
    }

    /// 从数组构造，白点默认 D65。
    pub const fn from_array(a: [f32; 3]) -> Self {
        Self { x: a[0], y: a[1], z: a[2], white: WhitePoint::D65 }
    }

    /// 从数组构造，显式指定白点。
    pub const fn from_array_with_white(a: [f32; 3], white: WhitePoint) -> Self {
        Self { x: a[0], y: a[1], z: a[2], white }
    }

    /// 当前参考白点。
    pub fn white(&self) -> WhitePoint {
        self.white
    }

    /// 数值数组（丢弃白点标签，调用方自行记录）。
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Bradford 适配到目标白点。已是目标白点时直接返回自身。
    pub fn adapt_to(self, target: WhitePoint) -> Xyz {
        if wp_eq(self.white, target) {
            return self;
        }
        let a = bradford_adapt([self.x, self.y, self.z], self.white, target);
        Xyz { x: a[0], y: a[1], z: a[2], white: target }
    }

    /// 转线性 sRGB。`LinSrgb` 原色固定 D65，因此先适配到 D65 再转矩阵。
    /// 这意味着最终 `LinSrgb` 与中间白点无关——白点只是中间表示。
    pub fn to_lin_srgb(self) -> LinSrgb {
        let d65 = self.adapt_to(WhitePoint::D65);
        let p = PalXyz::<D65, f32>::new(d65.x, d65.y, d65.z);
        let l: PalLinSrgb<f32> = p.into_color();
        LinSrgb { red: l.red, green: l.green, blue: l.blue }
    }

    /// 转 Yxy，保持当前白点标签（Yxy 色度含义依赖白点）。
    pub fn to_yxy(self) -> Yxy {
        let p = PalXyz::<D65, f32>::new(self.x, self.y, self.z);
        let y: PalYxy<D65, f32> = p.into_color();
        Yxy { x: y.x, y: y.y, luma: y.luma, white: self.white }
    }
}

/// Yxy（色度 + 亮度），携带参考白点标签。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yxy {
    pub x: f32,
    pub y: f32,
    pub luma: f32,
    white: WhitePoint,
}

impl Yxy {
    /// 白点默认 D65。
    pub const fn new(x: f32, y: f32, luma: f32) -> Self {
        Self { x, y, luma, white: WhitePoint::D65 }
    }

    pub const fn new_with_white(x: f32, y: f32, luma: f32, white: WhitePoint) -> Self {
        Self { x, y, luma, white }
    }

    pub fn white(&self) -> WhitePoint {
        self.white
    }

    /// 转回 XYZ，保持白点标签（纯坐标变换，不做色适应）。
    pub fn to_xyz(self) -> Xyz {
        let p = PalYxy::<D65, f32>::new(self.x, self.y, self.luma);
        let x: PalXyz<D65, f32> = p.into_color();
        Xyz { x: x.x, y: x.y, z: x.z, white: self.white }
    }
}

// ── 光谱采样 ──

#[derive(Clone, Debug)]
pub struct SpectralSamples {
    pub wavelengths: Vec<f32>,
    pub values: Vec<f32>,
    spectrum: colorimetry::spectrum::Spectrum,
}

impl SpectralSamples {
    pub fn new(wavelengths: Vec<f32>, values: Vec<f32>) -> Self {
        assert_eq!(wavelengths.len(), values.len(), "波长和值的长度必须一致");
        let map: Vec<(usize, f64)> = wavelengths
            .iter()
            .zip(values.iter())
            .map(|(&w, &v)| {
                assert!((380.0..=780.0).contains(&w), "波长 {w} 超出 380–780 nm");
                (w.round() as usize, v as f64)
            })
            .collect();

        SpectralSamples {
            wavelengths,
            values,
            spectrum: colorimetry::spectrum::Spectrum::from_wavelength_map(&map),
        }
    }

    pub fn from_pairs(pairs: &[(f32, f32)]) -> Self {
        let wavelengths = pairs.iter().map(|&(w, _)| w).collect();
        let values = pairs.iter().map(|&(_, v)| v).collect();
        Self::new(wavelengths, values)
    }

    fn from_illuminant(ill: colorimetry::illuminant::Illuminant) -> Self {
        let spectrum_cow = ill.spectrum();
        let spectrum_ref: &colorimetry::spectrum::Spectrum = spectrum_cow.as_ref();
        let slice: &[f64] = spectrum_ref.as_ref();
        let values: Vec<f32> = slice.iter().map(|&v| v as f32).collect();
        assert_eq!(values.len(), NS, "Illuminant 应提供 {NS} 个样本");

        let wavelengths: Vec<f32> = (380..=780).map(|w| w as f32).collect();
        SpectralSamples { wavelengths, values, spectrum: spectrum_cow.into_owned() }
    }

    fn to_illuminant(&self) -> colorimetry::illuminant::Illuminant {
        colorimetry::illuminant::Illuminant::new(self.spectrum)
    }
}

// ── 光谱 → XYZ / 线性 RGB ──

/// 光谱积分得到 XYZ，并适配到 `target` 白点参考。
///
/// `colorimetry` 返回的是 **D65 参考** XYZ，因此本函数 = 原始 D65 XYZ 经
/// Bradford 适配到 `target`。`target = D65` 时恒等。
///
/// 返回的 `Xyz` 携带 `target` 白点标签，`Xyz::to_lin_srgb()` 会自动适配回
/// D65 再转矩阵，因此最终 RGB 与 `target` 无关（物理正确）。
///
/// 不归一化、不 clamp——尺度与负值原样传给下游。
pub fn spectral_to_xyz(samples: &SpectralSamples, target: WhitePoint) -> Xyz {
    let ill = samples.to_illuminant();
    let raw = ill.xyz(None);
    let base = Xyz::new(raw.x() as f32, raw.y() as f32, raw.z() as f32); // D65 标签
    base.adapt_to(target)
}

/// 光谱 → 线性 sRGB（D65 参考）。
///
/// 不接受白点参数：`LinSrgb` 原色固定 D65，中间白点不改变最终 RGB。
/// 如需中间表示，用 `spectral_to_xyz(samples, target)`。
pub fn spectral_to_lin_srgb(samples: &SpectralSamples) -> LinSrgb {
    spectral_to_xyz(samples, WhitePoint::D65).to_lin_srgb()
}

// ── 标准光源样本 ──

pub fn d65_samples() -> SpectralSamples {
    SpectralSamples::from_illuminant(colorimetry::illuminant::Illuminant::d65())
}

pub fn d50_samples() -> SpectralSamples {
    SpectralSamples::from_illuminant(colorimetry::illuminant::Illuminant::d50())
}

// ── 数组 / Vec3 适配 ──

#[inline]
pub fn array_to_lin_srgb(a: [f32; 3]) -> LinSrgb { LinSrgb::from_array(a) }

#[inline]
pub fn lin_srgb_to_array(c: LinSrgb) -> [f32; 3] { c.to_array() }

#[inline]
pub fn vec3_to_lin_srgb(v: Vec3) -> LinSrgb { LinSrgb::from_vec3(v) }

#[inline]
pub fn lin_srgb_to_vec3(c: LinSrgb) -> Vec3 { c.to_vec3() }

// ── 编码 / 解码 ──

#[inline]
pub fn lin_srgb_to_srgb_array(linear: [f32; 3]) -> [f32; 3] {
    [
        linear_to_srgb(linear[0]),
        linear_to_srgb(linear[1]),
        linear_to_srgb(linear[2]),
    ]
}

#[inline]
pub fn srgb_to_lin_srgb_array(srgb: [f32; 3]) -> [f32; 3] {
    [
        srgb_to_linear(srgb[0]),
        srgb_to_linear(srgb[1]),
        srgb_to_linear(srgb[2]),
    ]
}

// ── 便捷构造 ──

#[inline]
pub fn lin_srgb(r: f32, g: f32, b: f32) -> LinSrgb { LinSrgb::new(r, g, b) }

#[inline]
pub fn xyz(x: f32, y: f32, z: f32) -> Xyz { Xyz::new(x, y, z) }

#[inline]
pub fn srgb(r: f32, g: f32, b: f32) -> Srgb { Srgb::new(r, g, b) }
