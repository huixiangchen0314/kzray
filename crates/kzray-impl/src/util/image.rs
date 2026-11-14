//! 图像输出。
//!
//! # 抽象边界
//!
//! 本模块是**唯一**接触 `image` crate 的地方。对外只暴露：
//!
//! ```text
//! write_image(image: &Image, path: impl AsRef<Path>) -> io::Result<()>
//! ```
//!
//! `image` crate 的类型（`ImageBuffer` / `DynamicImage` / `ImageFormat` /
//! `ImageError`）、错误、API **不出现**在本模块之外。
//!
//! # 为什么封装
//!
//! - **外部依赖可能替换**：今天用 `image` crate，明天可能换 `exr` 处理
//!   HDR、`png` 处理特定格式。封装后只改这一个文件，所有调用方不动。
//! - **`image` crate 的 API 会变**：0.24 → 0.25 有 API 调整。封装后
//!   升级只需改此模块。
//! - **调用方不该知道"我们用什么库"**：`render` / example 只知道
//!   "把 `Image` 写到路径"，不知道底层实现。
//!
//! # 数据格式
//!
//! kzray 的 `Image` 是**线性 RGB**（`kzray_data::color::Rgb`，`f32`）。
//! 写出前用 `Rgb::to_srgb_u8` 做 sRGB 编码（IEC 61966-2-1），得到
//! `[u8; 3]`。`image` crate 存的是 sRGB 编码后的 8 位值，两者语义一致。
//!
//! **不做** tone mapping / 曝光 / 白平衡——超亮值（> 1）被
//! `to_srgb_u8` clamp 到 255。场景亮度需要事先调整。
//!
//! # 支持的格式
//!
//! 按文件扩展名选择：`.png` / `.jpg` / `.jpeg` / `.ppm` / `.pgm`。
//! 未知扩展名返回 `InvalidInput` 错误。格式由 `image` crate 的
//! `ImageFormat::from_path` 推断。

use std::io;
use std::path::Path;

use kzray_data::Image;

/// 把线性 RGB 图像写到指定路径。
///
/// 按扩展名选择格式（PNG / JPEG / PPM / PGM）。内部做 sRGB 编码，
/// 不做 tone mapping。
///
/// # 错误
///
/// - 路径扩展名缺失或未知 → `InvalidInput`；
/// - 文件创建失败 → `PermissionDenied` / `NotFound` 等；
/// - 编码失败 → `Other`（包装 `image` crate 的错误消息）。
///
/// # 示例
///
/// ```text
/// use kzray_impl::render::write_image;
///
/// let image = film.resolve();
/// write_image(&image, "output.png")?;
/// ```
pub fn write_image(image: &Image, path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();

    // 扩展名 → ImageFormat。from_path 是 image crate 的推断函数。
    // 这里封装了 image crate 的 API，调用方不需要知道 ImageFormat。
    let format = image::ImageFormat::from_path(path)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

    let width = image.width();
    let height = image.height();

    // 构造 image crate 的 8 位 RGB 缓冲
    let mut buf = image::ImageBuffer::<image::Rgb<u8>, _>::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let rgb = image.get(x, y).to_srgb_u8();
            buf.put_pixel(x, y, image::Rgb(rgb));
        }
    }

    buf.save_with_format(path, format)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}
