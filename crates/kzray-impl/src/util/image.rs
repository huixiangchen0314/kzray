//! PPM（P3 ASCII）图像输出。
//!
//! 最简单的图像格式：纯文本，无压缩，无依赖。用于阶段 4 的调试输出。
//! 阶段 6 换 PNG / EXR。
//!
//! # sRGB 编码
//!
//! `Image` 存线性 RGB，写出前用 `Rgb::to_srgb_u8` 做 sRGB 编码。这是
//! 标准做法——线性值转 sRGB 显示空间。
//!
//! # 不做 tone mapping
//!
//! 线性值超亮（> 1）会被 `to_srgb_u8` clamp 到 255。场景亮度需要事先
//! 调整好（光源强度、曝光），或加 tone mapping。阶段 4 先不做 tone
//! mapping，场景参数手动调。

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use kzray_data::Image;

/// 写 PPM（P3）图像。
///
/// 输出格式：
///
/// ```text
/// P3
/// <width> <height>
/// 255
/// <r> <g> <b> <r> <g> <b> ...
/// ```
///
/// 每像素 RGB 三个 `[0, 255]` 整数，行主序，每行末尾换行。
pub fn write_ppm(image: &Image, path: impl AsRef<Path>) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let width = image.width();
    let height = image.height();

    writeln!(w, "P3")?;
    writeln!(w, "{} {}", width, height)?;
    writeln!(w, "255")?;

    for y in 0..height {
        for x in 0..width {
            let rgb = image.get(x, y);
            let encoded = rgb.to_srgb_u8();
            write!(w, "{} {} {} ", encoded[0], encoded[1], encoded[2])?;
        }
        writeln!(w)?;
    }

    w.flush()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use glam::UVec2;
    use kzray_data::Rgb;

    #[test]
    fn write_ppm_produces_valid_header() {
        let image = Image::new_filled(UVec2::new(2, 2), Rgb::BLACK);
        let tmp = std::env::temp_dir().join("kzray_test_ppm_header.ppm");
        write_ppm(&image, &tmp).unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines[0], "P3");
        assert_eq!(lines[1], "2 2");
        assert_eq!(lines[2], "255");

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn write_ppm_white_pixel() {
        let image = Image::new_filled(UVec2::new(1, 1), Rgb::WHITE);
        let tmp = std::env::temp_dir().join("kzray_test_ppm_white.ppm");
        write_ppm(&image, &tmp).unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        // 应该包含 "255 255 255"
        assert!(content.contains("255 255 255"), "content: {content}");

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn write_ppm_black_pixel() {
        let image = Image::new_filled(UVec2::new(1, 1), Rgb::BLACK);
        let tmp = std::env::temp_dir().join("kzray_test_ppm_black.ppm");
        write_ppm(&image, &tmp).unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("0 0 0"), "content: {content}");

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn write_ppm_gray_pixel() {
        // 0.5 线性 → sRGB ≈ 188
        let image = Image::new_filled(UVec2::new(1, 1), Rgb::splat(0.5));
        let tmp = std::env::temp_dir().join("kzray_test_ppm_gray.ppm");
        write_ppm(&image, &tmp).unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("188 188 188"), "content: {content}");

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn write_ppm_pixel_count_correct() {
        let image = Image::new_filled(UVec2::new(3, 2), Rgb::WHITE);
        let tmp = std::env::temp_dir().join("kzray_test_ppm_count.ppm");
        write_ppm(&image, &tmp).unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        // 头 3 行 + 2 行像素
        assert_eq!(lines.len(), 5);

        // 每行像素有 3 个 "255 255 255"
        for line in &lines[3..5] {
            let count = line.matches("255 255 255").count();
            assert_eq!(count, 3);
        }

        std::fs::remove_file(&tmp).ok();
    }
}
