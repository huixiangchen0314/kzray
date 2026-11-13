//! 图像数据结构。
//!
//! 本模块定义 `Image`——**纯数据的像素数组**，用于胶片解析后的输出、
//! 中间图像的存储、AOV 的容器。它不负责 I/O（EXR / PNG 编解码）、
//! 不做 tone mapping、不做 gamma 编码、不做 mipmap。
//!
//! # 数据格式
//!
//! 像素值用 [`Rgb`]（线性 RGB，无 gamma 编码）。写出到 8 位图像时，
//! 调用方用 [`Rgb::to_srgb_u8`] 做 sRGB 编码；tone mapping 和曝光
//! 调整在调用方完成。
//!
//! # 内存布局
//!
//! **行主序**（row-major）：索引 `i = y * width + x`。
//!
//! ```text
//! pixels = [
//!     (0, 0), (1, 0), ..., (width-1, 0),   // 第 0 行
//!     (0, 1), (1, 1), ..., (width-1, 1),   // 第 1 行
//!     ...
//! ]
//! ```
//!
//! 行主序与 C / C++ / Rust `Vec` 的二维数组约定一致，便于与外部图像
//! 库（image-rs、OpenEXR 等）互操作。
//!
//! # 不变量
//!
//! `pixels.len() == resolution.x * resolution.y`。字段私有，由构造器
//! 保证；`pixels_mut()` / `row_mut()` 返回切片，调用方不能改长度。
//!
//! # 与 pbrt `Image` 的差异
//!
//! pbrt 的 `Image` 携带 `ImageMetadata`（相机变换、曝光时间等），且
//! 直接以 `Image` 作为 `Film::GetImage` 的返回。我们把两者拆开：
//! `Image` 是纯像素数组，`ImageMetadata` 是独立的元数据容器，调用方
//! 需要时自行组合。这样 `Image` 的依赖保持最轻（只依赖 `glam` +
//! `Rgb`），元数据字段按需扩展。

use crate::common::rgb::Rgb;
use glam::UVec2;


// ═══════════════════════════════════════════════════════════════
// Image
// ═══════════════════════════════════════════════════════════════

/// 线性 RGB 图像。纯数据，无 I/O、无 tone mapping、无 gamma 编码。
///
/// 构造后尺寸固定，像素值可读写。行主序存储，见模块文档。
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    resolution: UVec2,
    pixels: Vec<Rgb>,
}

impl Image {
    /// 构造指定分辨率的图像，所有像素初始化为黑色 [`Rgb::BLACK`]。
    ///
    /// `resolution` 为 `(0, 0)` 时允许，得到空图像（`pixel_count() == 0`）。
    /// 实际使用中应避免零尺寸图像，但契约层不强制。
    pub fn new(resolution: UVec2) -> Self {
        let count = (resolution.x as usize) * (resolution.y as usize);
        Image {
            resolution,
            pixels: vec![Rgb::BLACK; count],
        }
    }

    /// 构造指定分辨率的图像，所有像素初始化为 `color`。
    ///
    /// 比 `new()` + 遍历 `set()` 快：一次性填充，无逐像素索引。
    pub fn new_filled(resolution: UVec2, color: Rgb) -> Self {
        let count = (resolution.x as usize) * (resolution.y as usize);
        Image {
            resolution,
            pixels: vec![color; count],
        }
    }

    /// 图像分辨率。
    #[inline]
    pub fn resolution(&self) -> UVec2 {
        self.resolution
    }

    /// 图像宽度（像素）。
    #[inline]
    pub fn width(&self) -> u32 {
        self.resolution.x
    }

    /// 图像高度（像素）。
    #[inline]
    pub fn height(&self) -> u32 {
        self.resolution.y
    }

    /// 像素总数，等于 `width * height`。
    #[inline]
    pub fn pixel_count(&self) -> usize {
        self.pixels.len()
    }

    /// 所有像素的只读切片。行主序。
    #[inline]
    pub fn pixels(&self) -> &[Rgb] {
        &self.pixels
    }

    /// 所有像素的可变切片。行主序。
    ///
    /// 返回的是 `&mut [Rgb]`，长度固定——调用方不能通过切片改变
    /// `pixel_count`，不变量保持。
    #[inline]
    pub fn pixels_mut(&mut self) -> &mut [Rgb] {
        &mut self.pixels
    }

    /// 第 `y` 行的只读切片（宽度为 `width()`）。
    ///
    /// # Panics
    ///
    /// `y >= height()` 时 panic。
    #[inline]
    pub fn row(&self, y: u32) -> &[Rgb] {
        assert!(y < self.resolution.y, "row index {y} out of bounds (height = {})", self.resolution.y);
        let w = self.resolution.x as usize;
        let start = (y as usize) * w;
        &self.pixels[start..start + w]
    }

    /// 第 `y` 行的可变切片。
    ///
    /// # Panics
    ///
    /// `y >= height()` 时 panic。
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [Rgb] {
        assert!(y < self.resolution.y, "row index {y} out of bounds (height = {})", self.resolution.y);
        let w = self.resolution.x as usize;
        let start = (y as usize) * w;
        &mut self.pixels[start..start + w]
    }

    /// 计算 `(x, y)` 的线性索引。行主序：`y * width + x`。
    ///
    /// # Panics
    ///
    /// `x >= width()` 或 `y >= height()` 时 panic。
    ///
    /// 调用方已在热路径外校验越界时，可以用 `x + y * width` 直接算，
    /// 避免重复检查。
    #[inline]
    pub fn index(&self, x: u32, y: u32) -> usize {
        assert!(
            x < self.resolution.x && y < self.resolution.y,
            "pixel ({x}, {y}) out of bounds (resolution = {}x{})",
            self.resolution.x,
            self.resolution.y,
        );
        (y as usize) * (self.resolution.x as usize) + (x as usize)
    }

    /// 读取 `(x, y)` 的像素值。`Rgb` 是 `Copy`，返回副本。
    ///
    /// # Panics
    ///
    /// `(x, y)` 越界时 panic。需要非 panic 版本时用 `pixels()[index]` 自己处理。
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> Rgb {
        self.pixels[self.index(x, y)]
    }

    /// 写入 `(x, y)` 的像素值。
    ///
    /// # Panics
    ///
    /// `(x, y)` 越界时 panic。
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, value: Rgb) {
        let idx = self.index(x, y);
        self.pixels[idx] = value;
    }

    /// 用 `f` 填充所有像素。
    ///
    /// 避免调用方手动遍历 `pixels_mut()` + `set()`；对简单填充足够，
    /// 复杂变换调用方直接 `pixels_mut().iter_mut()`。
    pub fn fill<F>(&mut self, mut f: F)
    where
        F: FnMut(u32, u32) -> Rgb,
    {
        let w = self.resolution.x;
        for y in 0..self.resolution.y {
            let start = (y as usize) * (w as usize);
            for x in 0..w {
                self.pixels[start + (x as usize)] = f(x, y);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// ImageMetadata
// ═══════════════════════════════════════════════════════════════

/// 图像元数据。
///
/// 与 [`Image`] 解耦的元数据容器。`Image` 只管像素；写出 EXR / PNG
/// 时的元数据（曝光、相机参数等）由实现层组合 `Image + ImageMetadata`
/// 使用。
///
/// # 扩展性
///
/// 标记 `#[non_exhaustive]`：外部 crate 不能直接构造，必须通过
/// [`Default::default`] 或实现层提供的构造器。这允许未来加字段
/// 而不破坏 API。
///
/// # 当前字段
///
/// 只包含 `exposure_time`——最小、最通用的元数据。其余字段（相机
/// 变换、渲染器版本、色彩空间等）等 I/O 层和后期处理实际需要时再加。
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ImageMetadata {
    /// 曝光时间（秒）。用于后期处理的曝光补偿。
    ///
    /// 渲染器不做 tone mapping 时，原始像素是“曝光时间 × 辐射亮度”；
    /// 后期处理按需调整。这个字段只是记录，不参与任何计算——`Image`
    /// 存储的像素已经是最终值（未 sRGB 编码的线性值）。
    pub exposure_time: f32,
}

impl Default for ImageMetadata {
    fn default() -> Self {
        ImageMetadata { exposure_time: 0.0 }
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use crate::Rgb;

use super::*;

    // ─────────── 构造 ───────────

    #[test]
    fn new_initializes_to_black() {
        let img = Image::new(UVec2::new(3, 2));
        assert_eq!(img.pixel_count(), 6);
        for p in img.pixels() {
            assert_eq!(*p, Rgb::BLACK);
        }
    }

    #[test]
    fn new_filled_initializes_to_color() {
        let color = Rgb::new(0.5, 0.25, 0.75);
        let img = Image::new_filled(UVec2::new(3, 2), color);
        for p in img.pixels() {
            assert_eq!(*p, color);
        }
    }

    #[test]
    fn zero_size_image_is_allowed() {
        let img = Image::new(UVec2::ZERO);
        assert_eq!(img.pixel_count(), 0);
        assert_eq!(img.width(), 0);
        assert_eq!(img.height(), 0);
        assert!(img.pixels().is_empty());
    }

    #[test]
    fn resolution_width_height() {
        let img = Image::new(UVec2::new(640, 480));
        assert_eq!(img.resolution(), UVec2::new(640, 480));
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);
        assert_eq!(img.pixel_count(), 640 * 480);
    }

    // ─────────── 索引 ───────────

    #[test]
    fn index_row_major() {
        let img = Image::new(UVec2::new(4, 3));
        // (0,0) → 0, (3,0) → 3, (0,1) → 4, (3,2) → 11
        assert_eq!(img.index(0, 0), 0);
        assert_eq!(img.index(3, 0), 3);
        assert_eq!(img.index(0, 1), 4);
        assert_eq!(img.index(3, 2), 11);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_x_out_of_bounds_panics() {
        let img = Image::new(UVec2::new(4, 3));
        img.index(4, 0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_y_out_of_bounds_panics() {
        let img = Image::new(UVec2::new(4, 3));
        img.index(0, 3);
    }

    // ─────────── get / set ───────────

    #[test]
    fn get_set_roundtrip() {
        let mut img = Image::new(UVec2::new(4, 3));
        let c = Rgb::new(0.1, 0.2, 0.3);
        img.set(2, 1, c);
        assert_eq!(img.get(2, 1), c);
        // 相邻像素不受影响
        assert_eq!(img.get(1, 1), Rgb::BLACK);
        assert_eq!(img.get(3, 1), Rgb::BLACK);
        assert_eq!(img.get(2, 0), Rgb::BLACK);
        assert_eq!(img.get(2, 2), Rgb::BLACK);
    }

    #[test]
    fn get_returns_copy() {
        let mut img = Image::new(UVec2::new(2, 2));
        img.set(0, 0, Rgb::WHITE);
        let v = img.get(0, 0);
        // get 返回副本，修改不影响图像
        let mut v2 = v;
        v2.r = 0.0;
        assert_eq!(img.get(0, 0).r, 1.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_out_of_bounds_panics() {
        let mut img = Image::new(UVec2::new(4, 3));
        img.set(4, 0, Rgb::WHITE);
    }

    // ─────────── 行访问 ───────────

    #[test]
    fn row_returns_correct_slice() {
        let mut img = Image::new(UVec2::new(4, 3));
        img.set(0, 1, Rgb::new(1.0, 0.0, 0.0));
        img.set(3, 1, Rgb::new(0.0, 1.0, 0.0));
        let row = img.row(1);
        assert_eq!(row.len(), 4);
        assert_eq!(row[0], Rgb::new(1.0, 0.0, 0.0));
        assert_eq!(row[1], Rgb::BLACK);
        assert_eq!(row[3], Rgb::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn row_mut_allows_write() {
        let mut img = Image::new(UVec2::new(4, 3));
        for p in img.row_mut(1) {
            *p = Rgb::WHITE;
        }
        for x in 0..4 {
            assert_eq!(img.get(x, 1), Rgb::WHITE);
        }
        // 相邻行不受影响
        for x in 0..4 {
            assert_eq!(img.get(x, 0), Rgb::BLACK);
            assert_eq!(img.get(x, 2), Rgb::BLACK);
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn row_out_of_bounds_panics() {
        let img = Image::new(UVec2::new(4, 3));
        img.row(3);
    }

    // ─────────── pixels / pixels_mut ───────────

    #[test]
    fn pixels_returns_full_slice() {
        let img = Image::new(UVec2::new(4, 3));
        assert_eq!(img.pixels().len(), 12);
    }

    #[test]
    fn pixels_mut_allows_write() {
        let mut img = Image::new(UVec2::new(4, 3));
        let white = Rgb::WHITE;
        for p in img.pixels_mut() {
            *p = white;
        }
        for p in img.pixels() {
            assert_eq!(*p, Rgb::WHITE);
        }
    }

    #[test]
    fn pixels_mut_cannot_change_length() {
        // &mut [Rgb] 是定长切片，调用方不能 push / resize
        // 这个测试验证类型正确性（编译通过即成立）
        let mut img = Image::new(UVec2::new(2, 2));
        let slice: &mut [Rgb] = img.pixels_mut();
        assert_eq!(slice.len(), 4);
    }

    // ─────────── fill ───────────

    #[test]
    fn fill_with_function() {
        let mut img = Image::new(UVec2::new(4, 3));
        img.fill(|x, y| Rgb::new(x as f32, y as f32, 0.0));
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(img.get(x, y), Rgb::new(x as f32, y as f32, 0.0));
            }
        }
    }

    #[test]
    fn fill_with_constant() {
        let mut img = Image::new(UVec2::new(4, 3));
        let c = Rgb::new(0.5, 0.5, 0.5);
        img.fill(|_, _| c);
        for p in img.pixels() {
            assert_eq!(*p, c);
        }
    }

    // ─────────── derive 行为 ───────────

    #[test]
    fn clone_produces_independent_image() {
        let mut img = Image::new(UVec2::new(2, 2));
        img.set(0, 0, Rgb::WHITE);
        let cloned = img.clone();
        img.set(0, 0, Rgb::new(0.5, 0.5, 0.5));
        // 原图像改了，克隆不受影响
        assert_eq!(cloned.get(0, 0), Rgb::WHITE);
        assert_eq!(img.get(0, 0), Rgb::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn partial_eq_compares_content() {
        let a = Image::new_filled(UVec2::new(2, 2), Rgb::WHITE);
        let b = Image::new_filled(UVec2::new(2, 2), Rgb::WHITE);
        assert_eq!(a, b);

        let c = Image::new(UVec2::new(2, 2)); // 黑色
        assert_ne!(a, c);
    }

    #[test]
    fn partial_eq_distinguishes_resolution() {
        let a = Image::new(UVec2::new(2, 3));
        let b = Image::new(UVec2::new(3, 2));
        assert_ne!(a, b);
    }

    #[test]
    fn send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Image>();
        assert_send_sync::<ImageMetadata>();
    }

    // ─────────── ImageMetadata ───────────

    #[test]
    fn metadata_default() {
        let m = ImageMetadata::default();
        assert_eq!(m.exposure_time, 0.0);
    }

    #[test]
    fn metadata_clone_and_eq() {
        let m = ImageMetadata { exposure_time: 1.0 / 30.0 };
        let n = m.clone();
        assert_eq!(m, n);
    }
}
