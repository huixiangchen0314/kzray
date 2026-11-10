//! `kzray-math::color` 的集成测试。只通过公共 API 验证行为。

use glam::Vec3;
use kzray_math::color::{
    array_to_lin_srgb, d50_samples, d65_samples, lin_srgb, lin_srgb_to_array,
    lin_srgb_to_srgb_array, lin_srgb_to_vec3, linear_to_srgb, srgb, srgb_to_lin_srgb_array,
    srgb_to_linear, srgb_to_linear_slice, spectral_to_lin_srgb, spectral_to_xyz,
    vec3_to_lin_srgb, xyz, SpectralSamples, WhitePoint,
};
use kzray_math::color::{LinSrgb, Xyz};

// ── WhitePoint ──

#[test]
fn d65_constant_is_correct() {
    assert!((WhitePoint::D65.x - 0.3127).abs() < 1e-4);
    assert!((WhitePoint::D65.y - 0.3290).abs() < 1e-4);
}

#[test]
fn d50_constant_is_correct() {
    assert!((WhitePoint::D50.x - 0.3457).abs() < 1e-4);
    assert!((WhitePoint::D50.y - 0.3585).abs() < 1e-4);
}

#[test]
fn default_is_d65() {
    assert_eq!(WhitePoint::default(), WhitePoint::D65);
}

#[test]
fn new_and_xy() {
    let wp = WhitePoint::new(0.3127, 0.3290);
    assert_eq!(wp.xy(), (0.3127, 0.3290));
}

// ── sRGB ↔ 线性往返 ──

#[test]
fn srgb_linear_roundtrip() {
    for v in [0.0f32, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        let linear = srgb_to_linear(v);
        let back = linear_to_srgb(linear);
        assert!((back - v).abs() < 1e-5, "v={v} back={back}");
    }
}

#[test]
fn known_linear_to_srgb_values() {
    assert!((linear_to_srgb(0.5) - 0.7354).abs() < 1e-3);
    assert!(linear_to_srgb(0.0).abs() < 1e-6);
    assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-5);
}

#[test]
fn known_srgb_to_linear_values() {
    assert!((srgb_to_linear(0.5) - 0.2140).abs() < 1e-3);
    assert!(srgb_to_linear(0.0).abs() < 1e-6);
    assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-5);
}

#[test]
fn array_roundtrip() {
    let srgb_in = [0.2f32, 0.5, 0.8];
    let lin = srgb_to_lin_srgb_array(srgb_in);
    let back = lin_srgb_to_srgb_array(lin);
    for i in 0..3 {
        assert!((back[i] - srgb_in[i]).abs() < 1e-5, "i={i}");
    }
}

#[test]
fn slice_matches_scalar() {
    let src = [0.1f32, 0.3, 0.5, 0.7, 0.9];
    let mut dst = src;
    srgb_to_linear_slice(&mut dst);
    for i in 0..src.len() {
        let scalar = srgb_to_linear(src[i]);
        assert!((dst[i] - scalar).abs() < 1e-6, "i={i}");
    }
}

// ── 自定义颜色类型适配 ──

#[test]
fn array_palette_roundtrip() {
    let arr = [0.1f32, 0.5, 0.9];
    assert_eq!(lin_srgb_to_array(array_to_lin_srgb(arr)), arr);
}

#[test]
fn vec3_palette_roundtrip() {
    let v = Vec3::new(0.1, 0.5, 0.9);
    let back = lin_srgb_to_vec3(vec3_to_lin_srgb(v));
    assert!((back - v).length() < 1e-6);
}

#[test]
fn d65_chromaticity_xyz_maps_to_white_in_lin_srgb() {
    let d65 = WhitePoint::D65;
    let xyz = Xyz::new(d65.x / d65.y, 1.0, (1.0 - d65.x - d65.y) / d65.y);
    let rgb = xyz.to_lin_srgb();
    assert!((rgb.red - 1.0).abs() < 5e-3, "r={}", rgb.red);
    assert!((rgb.green - 1.0).abs() < 5e-3, "g={}", rgb.green);
    assert!((rgb.blue - 1.0).abs() < 5e-3, "b={}", rgb.blue);
}

#[test]
fn d50_chromaticity_xyz_produces_finite_positive_lin_srgb() {
    let d50 = WhitePoint::D50;
    // 白点 D65 下的 Xyz，数值取 D50 色度 → 偏色但有限正
    let xyz = Xyz::new(d50.x / d50.y, 1.0, (1.0 - d50.x - d50.y) / d50.y);
    let rgb = xyz.to_lin_srgb();
    assert!(rgb.red.is_finite() && rgb.green.is_finite() && rgb.blue.is_finite());
    assert!(rgb.red > 0.0 && rgb.green > 0.0 && rgb.blue > 0.0);
}

#[test]
fn lin_srgb_to_xyz_roundtrip_is_stable() {
    let original = LinSrgb::new(0.2, 0.5, 0.8);
    let back = original.to_xyz().to_lin_srgb();
    assert!((back.red - original.red).abs() < 1e-4);
    assert!((back.green - original.green).abs() < 1e-4);
    assert!((back.blue - original.blue).abs() < 1e-4);
}

#[test]
fn lin_srgb_to_srgb_roundtrip_is_stable() {
    let original = LinSrgb::new(0.2, 0.5, 0.8);
    let back = original.to_srgb().to_lin_srgb();
    assert!((back.red - original.red).abs() < 1e-4);
    assert!((back.green - original.green).abs() < 1e-4);
    assert!((back.blue - original.blue).abs() < 1e-4);
}

// ── 白点标签与适配 ──

#[test]
fn xyz_default_white_is_d65() {
    assert_eq!(Xyz::new(0.1, 0.2, 0.3).white(), WhitePoint::D65);
    assert_eq!(xyz(0.1, 0.2, 0.3).white(), WhitePoint::D65);
    assert_eq!(LinSrgb::new(0.5, 0.5, 0.5).to_xyz().white(), WhitePoint::D65);
}

#[test]
fn xyz_new_with_white_retains_label() {
    let x = Xyz::new_with_white(0.5, 0.4, 0.3, WhitePoint::D50);
    assert_eq!(x.white(), WhitePoint::D50);
    assert_eq!(x.to_yxy().white(), WhitePoint::D50);
}

#[test]
fn adapt_to_roundtrip() {
    let orig = Xyz::new(0.5, 0.4, 0.3);
    let back = orig.adapt_to(WhitePoint::D50).adapt_to(WhitePoint::D65);
    assert!((back.x - orig.x).abs() < 1e-4);
    assert!((back.y - orig.y).abs() < 1e-4);
    assert!((back.z - orig.z).abs() < 1e-4);
    assert_eq!(back.white(), WhitePoint::D65);
}

#[test]
fn adapt_to_is_idempotent() {
    let orig = Xyz::new(0.5, 0.4, 0.3);
    let once = orig.adapt_to(WhitePoint::D50);
    let twice = once.adapt_to(WhitePoint::D50);
    assert!((twice.x - once.x).abs() < 1e-6);
    assert!((twice.y - once.y).abs() < 1e-6);
    assert!((twice.z - once.z).abs() < 1e-6);
    assert_eq!(twice.white(), WhitePoint::D50);
}

// ── SpectralSamples ──

#[test]
fn spectral_samples_new() {
    let s = SpectralSamples::new(vec![400.0, 500.0, 600.0], vec![0.1, 0.5, 0.9]);
    assert_eq!(s.wavelengths.len(), 3);
    assert_eq!(s.values.len(), 3);
}

#[test]
#[should_panic]
fn spectral_samples_mismatched_length_panics() {
    SpectralSamples::new(vec![400.0, 500.0], vec![0.1]);
}

#[test]
#[should_panic]
fn spectral_samples_out_of_range_panics() {
    SpectralSamples::new(vec![300.0], vec![0.5]);
}

// ── 标准光源样本 ──

#[test]
fn d65_samples_non_empty() {
    let s = d65_samples();
    assert!(!s.wavelengths.is_empty());
    assert_eq!(s.wavelengths.len(), s.values.len());
    assert!(s.wavelengths[0] >= 380.0);
    assert!(*s.wavelengths.last().unwrap() <= 780.0);
    assert!(s.values.iter().all(|&v| v >= 0.0));
}

#[test]
fn d50_samples_non_empty() {
    let s = d50_samples();
    assert!(!s.wavelengths.is_empty());
    assert_eq!(s.wavelengths.len(), s.values.len());
}

#[test]
fn d65_and_d50_samples_differ() {
    let d65 = d65_samples();
    let d50 = d50_samples();
    let same = d65.values.len() == d50.values.len()
        && d65
            .values
            .iter()
            .zip(d50.values.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
    assert!(!same, "D65 和 D50 光谱不应完全相同");
}

// ── 光谱积分与白点语义 ──

#[test]
fn d65_spectral_to_xyz_finite() {
    let xyz = spectral_to_xyz(&d65_samples(), WhitePoint::D65);
    assert!(xyz.x.is_finite() && xyz.y.is_finite() && xyz.z.is_finite());
    assert!(xyz.y > 0.0);
}

#[test]
fn d65_spectral_to_xyz_chromaticity_matches_d65() {
    let xyz = spectral_to_xyz(&d65_samples(), WhitePoint::D65);
    let sum = xyz.x + xyz.y + xyz.z;
    let cx = xyz.x / sum;
    let cy = xyz.y / sum;
    assert!((cx - 0.3127).abs() < 0.02, "cx={cx}, 期望 ≈ 0.3127");
    assert!((cy - 0.3290).abs() < 0.02, "cy={cy}, 期望 ≈ 0.3290");
}

#[test]
fn spectral_to_xyz_preserves_target_white_label() {
    let x65 = spectral_to_xyz(&d65_samples(), WhitePoint::D65);
    let x50 = spectral_to_xyz(&d65_samples(), WhitePoint::D50);
    assert_eq!(x65.white(), WhitePoint::D65);
    assert_eq!(x50.white(), WhitePoint::D50);
}

/// D65 光谱在 D50 参考下：色貌不变，中性白在 D50 坐标系的色度应为 D50。
/// 这是验证 `adapt_to` Bradford 实现正确性的核心测试。
#[test]
fn d65_spectrum_in_d50_reference_has_d50_chromaticity() {
    let xyz = spectral_to_xyz(&d65_samples(), WhitePoint::D50);
    let sum = xyz.x + xyz.y + xyz.z;
    let cx = xyz.x / sum;
    let cy = xyz.y / sum;
    assert!((cx - 0.3457).abs() < 0.02, "cx={cx}, 期望 ≈ 0.3457");
    assert!((cy - 0.3585).abs() < 0.02, "cy={cy}, 期望 ≈ 0.3585");
}

#[test]
fn d65_spectral_to_lin_srgb_is_approximately_white() {
    let rgb = spectral_to_lin_srgb(&d65_samples());
    assert!(rgb.red.is_finite() && rgb.green.is_finite() && rgb.blue.is_finite());
    let max = rgb.red.max(rgb.green).max(rgb.blue);
    let min = rgb.red.min(rgb.green).min(rgb.blue);
    assert!((max - min) / max < 0.1,
        "D65 转线性 sRGB 应接近中性白：r={}, g={}, b={}",
        rgb.red, rgb.green, rgb.blue);
}

/// D50 光谱经 colorimetry 积分后，转 LinSrgb 也应为中性白。
/// 注意：这里不要求中间 XYZ 精确等于 D50 白点——实际光谱积分有色偏，
/// 最终 RGB 接近白即满足可用性。
#[test]
fn d50_spectral_to_lin_srgb_is_approximately_white() {
    let rgb = spectral_to_lin_srgb(&d50_samples());
    assert!(rgb.red.is_finite() && rgb.green.is_finite() && rgb.blue.is_finite());
    let max = rgb.red.max(rgb.green).max(rgb.blue);
    let min = rgb.red.min(rgb.green).min(rgb.blue);
    assert!((max - min) / max < 0.1,
        "D50 光谱转线性 sRGB 应接近中性白：r={}, g={}, b={}",
        rgb.red, rgb.green, rgb.blue);
}

#[test]
fn d65_and_d50_white_samples_agree() {
    let norm = |c: LinSrgb| -> [f32; 3] {
        let y = 0.2126 * c.red + 0.7152 * c.green + 0.0722 * c.blue;
        [c.red / y, c.green / y, c.blue / y]
    };
    let a = norm(spectral_to_lin_srgb(&d65_samples()));
    let b = norm(spectral_to_lin_srgb(&d50_samples()));
    for i in 0..3 {
        assert!((a[i] - b[i]).abs() < 0.05, "i={i} d65={} d50={}", a[i], b[i]);
    }
}

/// 最终 LinSrgb 与中间白点无关：同一光谱经不同白点表示的 XYZ，
/// 经 `to_lin_srgb()` 自动适配回 D65，结果应一致。
#[test]
fn lin_srgb_independent_of_intermediate_white() {
    let via_d65 = spectral_to_xyz(&d65_samples(), WhitePoint::D65).to_lin_srgb();
    let via_d50 = spectral_to_xyz(&d65_samples(), WhitePoint::D50).to_lin_srgb();
    assert!((via_d65.red - via_d50.red).abs() < 1e-3);
    assert!((via_d65.green - via_d50.green).abs() < 1e-3);
    assert!((via_d65.blue - via_d50.blue).abs() < 1e-3);
}

#[test]
fn black_spectrum_integrates_to_black() {
    let pairs: Vec<(f32, f32)> = (0..401).map(|i| (380.0 + i as f32, 0.0)).collect();
    let xyz = spectral_to_xyz(&SpectralSamples::from_pairs(&pairs), WhitePoint::D65);
    assert!(xyz.x.abs() < 1e-6);
    assert!(xyz.y.abs() < 1e-6);
    assert!(xyz.z.abs() < 1e-6);
}

/// Bradford 是线性变换：零入零出，任意白点下黑仍是黑。
#[test]
fn black_spectrum_adaptation_preserves_black() {
    let pairs: Vec<(f32, f32)> = (0..401).map(|i| (380.0 + i as f32, 0.0)).collect();
    let x65 = spectral_to_xyz(&SpectralSamples::from_pairs(&pairs), WhitePoint::D65);
    let x50 = spectral_to_xyz(&SpectralSamples::from_pairs(&pairs), WhitePoint::D50);
    assert!(x65.x.abs() < 1e-6 && x65.y.abs() < 1e-6 && x65.z.abs() < 1e-6);
    assert!(x50.x.abs() < 1e-6 && x50.y.abs() < 1e-6 && x50.z.abs() < 1e-6);
}

#[test]
fn spectral_samples_from_pairs() {
    let s = SpectralSamples::from_pairs(&[(400.0, 0.1), (500.0, 0.5), (600.0, 0.9)]);
    let _ = s;
}

// ── 不 clamp 契约 ──

#[test]
fn lin_srgb_allows_out_of_gamut_values() {
    let out_of_gamut = Xyz::new(0.9, 0.1, 0.1).to_lin_srgb();
    assert!(out_of_gamut.red.is_finite());
    assert!(out_of_gamut.green.is_finite());
    assert!(out_of_gamut.blue.is_finite());

    let c = LinSrgb::new(-0.5, 1.5, 2.0);
    assert_eq!(c.red, -0.5);
    assert_eq!(c.green, 1.5);
    assert_eq!(c.blue, 2.0);
}

// ── 便捷构造 ──

#[test]
fn constructor_helpers() {
    let c = lin_srgb(0.1, 0.2, 0.3);
    assert_eq!((c.red, c.green, c.blue), (0.1, 0.2, 0.3));
    let x = xyz(0.1, 0.2, 0.3);
    assert_eq!((x.x, x.y, x.z), (0.1, 0.2, 0.3));
    let s = srgb(0.1, 0.2, 0.3);
    assert_eq!((s.red, s.green, s.blue), (0.1, 0.2, 0.3));
}
