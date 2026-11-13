//! 纯数据的 RGB 颜色类型。
//!
//! 本模块定义 `Rgb`——**纯数据**的三通道颜色，用于图像存储、元数据、
//! AOV 输出等场景。它没有算术操作（加法、乘法、lerp），那些属于
//! `kzray_abs::spectrum` 的 RGB 光谱类型。
//!
//! # 与 `kzray_abs::spectrum::Rgb` 的区别
//!
//! | 类型 | 模块 | 用途 | 算术 |
//! |---|---|---|---|
//! | `spectrum::Rgb` | `kzray-abs` | 光谱算术、BSDF 积分、MIS 加权 | 是 |
//! | `data::color::Rgb` | 本模块 | 图像像素存储、元数据、AOV | 否（仅数据构造/序列化） |
//!
//! 两个类型字段相同（`r` / `g` / `b` 各 `f32`），但**不合并**：
//! 算术类型依赖光谱学（波长、采样、积分），数据类型的依赖应该尽量
//! 轻（只依赖 `glam` 或不依赖）。`Image` 存 `data::color::Rgb`，
//! 积分器算 `spectrum::Rgb`，两者在实现层边界处显式转换。
//!
//! # 线性 vs sRGB
//!
//! `Rgb` 存储的是**线性 RGB**（未做 gamma 编码）。显示 / 写出到
//! 8 位图像时，用 [`Rgb::to_srgb_u8`] 做 sRGB 编码。`Rgb` 本身不做
//! tone mapping、exposure 调整——那些是图像 I/O 或后期处理的职责。

// ═══════════════════════════════════════════════════════════════
// Rgb
// ═══════════════════════════════════════════════════════════════

/// 线性 RGB 颜色。纯数据，无算术操作。
///
/// 字段全部 `pub`，允许直接读写。构造用 [`Rgb::new`] 或常量
/// [`Rgb::BLACK`] / [`Rgb::WHITE`]。
///
/// `PartialEq` 用逐字段比较，`NaN != NaN`——这是 `f32` 的固有语义，
/// 不做特殊处理。需要 `Eq` / `Hash` 时，调用方应先把 `f32` 转成
/// `u32` 位模式。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Rgb {
    /// 黑色 `(0, 0, 0)`。也是 [`Default`] 的返回值。
    pub const BLACK: Rgb = Rgb::new(0.0, 0.0, 0.0);

    /// 白色 `(1, 1, 1)`。
    pub const WHITE: Rgb = Rgb::new(1.0, 1.0, 1.0);

    /// 构造颜色。无 clamp，字段原样存储。
    #[inline]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Rgb { r, g, b }
    }

    /// 三通道同值构造（灰度）。
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Rgb { r: v, g: v, b: v }
    }

    /// 从 `[f32; 3]` 构造。数组顺序为 `[r, g, b]`。
    #[inline]
    pub const fn from_array(arr: [f32; 3]) -> Self {
        Rgb::new(arr[0], arr[1], arr[2])
    }

    /// 转为 `[f32; 3]`。数组顺序为 `[r, g, b]`。
    #[inline]
    pub const fn to_array(self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    /// 转换到 sRGB 编码的 8 位整数，范围 `[0, 255]`。
    ///
    /// 流程：
    ///
    /// 1. 线性通道 clamp 到 `[0, 1]`（负值截 0，超亮截 1）；
    /// 2. 应用 sRGB gamma 编码（IEC 61966-2-1）；
    /// 3. 乘 255 并四舍五入到 `u8`。
    ///
    /// **不做** tone mapping、exposure 调整、白平衡。这些应在调用本
    /// 方法前由图像 I/O 或后期处理完成。本方法只负责“线性 → sRGB 编码”
    /// 这一步，因为它是数据序列化的标准末段。
    ///
    /// 使用 `f32::round` 而非截断，保证 `0.5` 线性（sRGB ≈ 0.735）
    /// 编码为 188 而非 187。
    pub fn to_srgb_u8(self) -> [u8; 3] {
        [
            srgb_encode_channel(self.r),
            srgb_encode_channel(self.g),
            srgb_encode_channel(self.b),
        ]
    }
}

/// 单通道线性 → sRGB 编码的 u8。
///
/// 线性值先 clamp 到 `[0, 1]`，再按 IEC 61966-2-1 做 sRGB 编码，
/// 最后乘 255 四舍五入。非有限值（NaN / ±∞）按 clamp 规则处理：
/// `NaN.clamp(0, 1)` 返回 `NaN`（Rust 语义），但 `NaN as u8` 在
/// release 下是 UB 风险——因此先判断有限性。
#[inline]
fn srgb_encode_channel(linear: f32) -> u8 {
    // NaN 和 ±∞：clamp 后仍非有限，直接按边界处理
    let c = if linear.is_finite() {
        linear.clamp(0.0, 1.0)
    } else if linear.is_nan() {
        0.0 // NaN 无意义，按黑处理
    } else if linear > 0.0 {
        1.0 // +∞
    } else {
        0.0 // -∞
    };

    // sRGB 编码（线性 → sRGB 非线性）
    let encoded = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };

    // 编码后范围仍在 [0, 1]（数值误差可能略超），乘 255 四舍五入
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────── 构造与常量 ───────────

    #[test]
    fn new_preserves_fields() {
        let c = Rgb::new(0.1, 0.2, 0.3);
        assert_eq!(c.r, 0.1);
        assert_eq!(c.g, 0.2);
        assert_eq!(c.b, 0.3);
    }

    #[test]
    fn black_and_white_constants() {
        assert_eq!(Rgb::BLACK, Rgb::new(0.0, 0.0, 0.0));
        assert_eq!(Rgb::WHITE, Rgb::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn default_is_black() {
        assert_eq!(Rgb::default(), Rgb::BLACK);
    }

    #[test]
    fn splat_sets_all_channels() {
        assert_eq!(Rgb::splat(0.5), Rgb::new(0.5, 0.5, 0.5));
    }

    // ─────────── 数组往返 ───────────

    #[test]
    fn from_array_to_array_roundtrip() {
        let arr = [0.1, 0.2, 0.3];
        let c = Rgb::from_array(arr);
        assert_eq!(c.to_array(), arr);
    }

    #[test]
    fn array_order_is_rgb() {
        let c = Rgb::from_array([1.0, 2.0, 3.0]);
        assert_eq!(c.r, 1.0);
        assert_eq!(c.g, 2.0);
        assert_eq!(c.b, 3.0);
    }

    // ─────────── sRGB 编码 ───────────

    #[test]
    fn srgb_black_is_zero() {
        assert_eq!(Rgb::BLACK.to_srgb_u8(), [0, 0, 0]);
    }

    #[test]
    fn srgb_white_is_255() {
        assert_eq!(Rgb::WHITE.to_srgb_u8(), [255, 255, 255]);
    }

    #[test]
    fn srgb_mid_gray_is_188() {
        // 0.5 线性 sRGB ≈ 0.7354 → 255 * 0.7354 ≈ 187.5 → 188
        // 验证用 round 而非截断（截断会得 187）
        let c = Rgb::splat(0.5);
        let encoded = c.to_srgb_u8();
        assert_eq!(encoded, [188, 188, 188]);
    }

    #[test]
    fn srgb_linear_segment() {
        // 0.003 线性在 sRGB 线性段（≤ 0.0031308）
        // 12.92 * 0.003 = 0.03876 → 255 * 0.03876 = 9.88 → 10
        let c = Rgb::splat(0.003);
        let encoded = c.to_srgb_u8();
        assert_eq!(encoded[0], 10);
    }

    #[test]
    fn srgb_negative_clamps_to_zero() {
        let c = Rgb::new(-1.0, -0.5, -0.001);
        assert_eq!(c.to_srgb_u8(), [0, 0, 0]);
    }

    #[test]
    fn srgb_over_one_clamps_to_255() {
        let c = Rgb::new(2.0, 1.5, 1.001);
        assert_eq!(c.to_srgb_u8(), [255, 255, 255]);
    }

    #[test]
    fn srgb_per_channel_independent() {
        let c = Rgb::new(0.0, 0.5, 1.0);
        let encoded = c.to_srgb_u8();
        assert_eq!(encoded[0], 0);
        assert_eq!(encoded[1], 188);
        assert_eq!(encoded[2], 255);
    }

    #[test]
    fn srgb_nan_becomes_zero() {
        let c = Rgb::splat(f32::NAN);
        assert_eq!(c.to_srgb_u8(), [0, 0, 0]);
    }

    #[test]
    fn srgb_positive_infinity_becomes_255() {
        let c = Rgb::new(f32::INFINITY, f32::NEG_INFINITY, 0.5);
        let encoded = c.to_srgb_u8();
        assert_eq!(encoded[0], 255); // +∞ → 1.0
        assert_eq!(encoded[1], 0);   // -∞ → 0.0
        assert_eq!(encoded[2], 188); // 0.5 → 188
    }

    // ─────────── derive 行为 ───────────

    #[test]
    fn copy_semantics() {
        let c = Rgb::new(0.1, 0.2, 0.3);
        let copy = c;
        assert_eq!(c, copy);
        // 修改副本不影响原值
        let mut another = c;
        another.r = 0.9;
        assert_eq!(c.r, 0.1);
        assert_eq!(another.r, 0.9);
    }

    #[test]
    fn partial_eq_distinguishes_fields() {
        let a = Rgb::new(0.1, 0.2, 0.3);
        let b = Rgb::new(0.1, 0.2, 0.3);
        let c = Rgb::new(0.1, 0.2, 0.4);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn debug_output_contains_type_name() {
        let c = Rgb::new(0.1, 0.2, 0.3);
        let s = format!("{:?}", c);
        assert!(s.contains("Rgb"));
    }

    #[test]
    fn send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Rgb>();
    }
}
