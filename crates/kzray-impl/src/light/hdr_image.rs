//! HDR 环境贴图（光谱代数）。
//!
//! [`HdrImage<S>`] 存储 [`SpectralAlgebra`] 像素，按等距柱状投影采样。
//!
//! # 不负责图像加载
//!
//! **图像加载与 RGB → S 转换由用户完成**。`.hdr` / `.exr` 文件是 RGB
//! 格式，转成任意 `S` 的规则因表示而异：
//!
//! - `Rgb`：直接 `Rgb([r, g, b])`，恒等映射；
//! - `Spectrum<N>`：RGB → 光谱上采样，需要 CIE 色彩匹配函数积分；
//! - 自定义光谱表示：用户自己的转换规则。
//!
//! 把加载交给用户，`HdrImage` 不依赖 `image` crate，也不假设任何转换
//! 规则。用户加载后调 [`HdrImage::from_pixels`] 构造。
//!
//! # 采样
//!
//! - **u 方向 repeat**：HDR 图像左右边缘是同一条经线（接缝），
//!   `u = 1.0` 视作 `u = 0.0`；
//! - **v 方向 clamp**：`v = 0` 是天空（+Y），`v = 1` 是地面（−Y），
//!   极点行不 wrap。
//!
//! 双线性插值四角像素，用 [`SpectralAlgebra`] 的 `add` / `sub` / `scale`
//! 导出 `lerp`——不要求 `S` 实现 `lerp` 方法。
//! TOD  重命名为光谱图像

use glam::Vec2;

use kzray_abs::SpectralAlgebra;

// ═══════════════════════════════════════════════════════════════
// HdrImage
// ═══════════════════════════════════════════════════════════════

/// HDR 环境贴图。行主序存储，`data.len() == width * height`。
///
/// 像素是光谱代数 `S`——值域 `[0, ∞)`，不做任何 clamp。曝光控制交给
/// 上层（`HdrInfiniteLight.scale`）。
pub struct HdrImage<S: SpectralAlgebra> {
    width: u32,
    height: u32,
    data: Vec<S>,
}

impl<S: SpectralAlgebra> HdrImage<S> {
    /// 从已解码的像素构造。
    ///
    /// # Panics
    ///
    /// - `width == 0` 或 `height == 0`；
    /// - `data.len() != width * height`。
    pub fn from_pixels(width: u32, height: u32, data: Vec<S>) -> Self {
        assert!(
            width > 0 && height > 0,
            "HDR dimensions must be non-zero (got {width}x{height})"
        );
        assert_eq!(
            data.len() as u64,
            width as u64 * height as u64,
            "HDR data length ({}) must equal width * height ({width} * {height})",
            data.len()
        );
        Self {
            width,
            height,
            data,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// 双线性采样。`uv ∈ [0, 1]²`。
    ///
    /// - u 方向：`u - u.floor()` 做 repeat，处理左右接缝；
    /// - v 方向：`v.clamp(0, 1)`，处理上下极点。
    ///
    /// 单像素图像（`1×1`）直接返回该像素。
    pub fn sample(&self, uv: Vec2) -> S {
        let u = uv.x - uv.x.floor();
        let v = uv.y.clamp(0.0, 1.0);

        let x = u * (self.width - 1) as f32;
        let y = v * (self.height - 1) as f32;

        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1) % self.width;
        let y1 = (y0 + 1).min(self.height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let at = |x: u32, y: u32| self.data[y as usize * self.width as usize + x as usize];

        let top = S::lerp(at(x0, y0), at(x1, y0), fx);
        let bottom = S::lerp(at(x0, y1), at(x1, y1), fx);
        S::lerp(top, bottom, fy)
    }
}


// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use kzray_abs::Rgb;

    fn approx3(a: Rgb, b: Rgb) -> bool {
        (a.0[0] - b.0[0]).abs() < 1e-5
            && (a.0[1] - b.0[1]).abs() < 1e-5
            && (a.0[2] - b.0[2]).abs() < 1e-5
    }

    // ── 单像素 ──

    #[test]
    fn single_pixel_returns_value() {
        let px = Rgb::new(2.0, 3.0, 4.0);
        let img = HdrImage::from_pixels(1, 1, vec![px]);
        for uv in [
            Vec2::new(0.0, 0.0),
            Vec2::new(0.5, 0.5),
            Vec2::new(1.0, 1.0),
        ] {
            assert!(approx3(img.sample(uv), px));
        }
    }

    // ── 2×1 水平双线性 ──

    #[test]
    fn horizontal_bilinear() {
        let black = Rgb::new(0.0, 0.0, 0.0);
        let white = Rgb::new(1.0, 1.0, 1.0);
        let img = HdrImage::from_pixels(2, 1, vec![black, white]);

        assert!(approx3(img.sample(Vec2::new(0.0, 0.5)), black));
        assert!(approx3(img.sample(Vec2::new(1.0, 0.5)), white));
        assert!(approx3(
            img.sample(Vec2::new(0.5, 0.5)),
            Rgb::new(0.5, 0.5, 0.5)
        ));
    }

    // ── u 方向 repeat ──

    #[test]
    fn u_wraps_at_seam() {
        let black = Rgb::new(0.0, 0.0, 0.0);
        let white = Rgb::new(1.0, 1.0, 1.0);
        let img = HdrImage::from_pixels(2, 1, vec![black, white]);

        // u 略小于 1：接近右边界（白）
        let s = img.sample(Vec2::new(0.9999, 0.5));
        assert!(s.0[0] > 0.99);

        // u 超过 1.0 后 wrap 到 u - 1
        let s_wrap = img.sample(Vec2::new(1.5, 0.5));
        let s_direct = img.sample(Vec2::new(0.5, 0.5));
        assert!(approx3(s_wrap, s_direct));
    }

    // ── v 方向 clamp ──

    #[test]
    fn v_clamps_at_poles() {
        let top = Rgb::new(1.0, 0.0, 0.0);
        let bottom = Rgb::new(0.0, 1.0, 0.0);
        let img = HdrImage::from_pixels(1, 2, vec![top, bottom]);

        assert!(approx3(img.sample(Vec2::new(0.5, -0.5)), top));
        assert!(approx3(img.sample(Vec2::new(0.5, 1.5)), bottom));
    }

    // ── 2×2 四角 ──

    #[test]
    fn four_corners() {
        let red = Rgb::new(1.0, 0.0, 0.0);
        let green = Rgb::new(0.0, 1.0, 0.0);
        let blue = Rgb::new(0.0, 0.0, 1.0);
        let yellow = Rgb::new(1.0, 1.0, 0.0);
        let img = HdrImage::from_pixels(2, 2, vec![red, green, blue, yellow]);

        assert!(approx3(img.sample(Vec2::new(0.0, 0.0)), red));
        assert!(approx3(img.sample(Vec2::new(1.0, 0.0)), green));
        assert!(approx3(img.sample(Vec2::new(0.0, 1.0)), blue));
        assert!(approx3(img.sample(Vec2::new(1.0, 1.0)), yellow));
        // 中心：四角平均
        assert!(approx3(
            img.sample(Vec2::new(0.5, 0.5)),
            Rgb::new(0.5, 0.5, 0.25)
        ));
    }

    // ── HDR 值域不 clamp ──

    #[test]
    fn hdr_values_above_one() {
        let bright = Rgb::new(10.0, 20.0, 30.0);
        let img = HdrImage::from_pixels(2, 1, vec![bright, bright]);
        assert!(approx3(img.sample(Vec2::new(0.5, 0.5)), bright));
    }

    // ── 构造校验 ──

    #[test]
    #[should_panic(expected = "dimensions must be non-zero")]
    fn reject_zero_width() {
        HdrImage::<Rgb>::from_pixels(0, 1, vec![]);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn reject_wrong_len() {
        HdrImage::<Rgb>::from_pixels(2, 2, vec![Rgb::ZERO]);
    }
}
