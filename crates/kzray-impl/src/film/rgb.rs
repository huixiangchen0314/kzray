//! RGB 胶片。
//!
//! 累积样本到像素缓冲，按 filter 权重分配，resolve 时归一化为 `Image`。
//!
//! # 数据流
//!
//! ```text
//! add_sample(pixel, value, wavelengths, weight)
//!   → sensor.process(value, wavelengths) -> Rgb
//!   → filter.weight(offset) 分配到邻域像素
//!   → rgb_sum += rgb × filter_w × weight
//!   → weight_sum += filter_w × weight
//!
//! resolve()
//!   → rgb = rgb_sum / weight_sum
//!   → Image
//! ```
//!
//! # Filter 的语义
//!
//! `filter.weight(offset)` 返回 filter 在偏移处的权重。`add_sample` 把
//! `rgb × filter_w × weight` 累积到邻域像素，`filter_w × weight` 累积到
//! 权重和。resolve 时 `rgb_sum / weight_sum` 是 filter 加权平均，与
//! `weight` 的具体含义（`1/spp` 或 MIS 权重）无关。
//!
//! # 并发
//!
//! `add_sample` 用 `&self`，内部 `Mutex<Vec<PixelData>>`。多线程渲染时
//! 所有线程竞争同一把锁——阶段 1 验证契约够用，阶段 6 优化为分块。
//!
//! # 光谱转换
//!
//! `sensor: Arc<dyn PixelSensor<S>>` 把光谱值 `S` 转成 `Rgb`：
//!
//! - `RgbSensor`：`S = SpectrumRgb`，直通；
//! - `CieSensor`：`S = Spectrum<N>`，CIE 积分。

use std::sync::{Arc, Mutex};

use glam::{UVec2, Vec2};

use kzray_abs::film::Film;
use kzray_math::filter::Filter;
use kzray_abs::sensor::PixelSensor;
use kzray_abs::spectrum::SpectralAlgebra;

use kzray_data::Rgb;
use kzray_data::Image;

// ═══════════════════════════════════════════════════════════════
// PixelData
// ═══════════════════════════════════════════════════════════════

/// 单像素的累积数据。
///
/// `rgb_sum` 是加权累积的 RGB，`weight_sum` 是累积的权重。resolve 时
/// `rgb_sum / weight_sum` 得到 filter 加权平均。
#[derive(Clone, Copy, Default)]
struct PixelData {
    rgb_sum: Rgb,
    weight_sum: f32,
}

// ═══════════════════════════════════════════════════════════════
// RGBFilm
// ═══════════════════════════════════════════════════════════════

/// RGB 胶片。
///
/// 累积样本到像素缓冲，按 filter 权重分配到邻域像素。resolve 时
/// 归一化为 `Image`。
///
/// # 泛型参数
///
/// `S: SpectralAlgebra`：样本的光谱类型。`sensor` 负责把 `S` 转成
/// `Rgb`。`RGBFilm<SpectrumRgb>` 用于 RGB 渲染，`RGBFilm<Spectrum<N>>`
/// 用于光谱渲染。
///
/// # 内部可变性
///
/// `pixels: Mutex<Vec<PixelData>>`：粗粒度锁。阶段 1 验证契约，阶段 6
/// 优化为分块（每 tile 一把锁）。
pub struct RGBFilm<S: SpectralAlgebra> {
    resolution: UVec2,
    filter: Arc<dyn Filter>,
    sensor: Arc<dyn PixelSensor<S>>,
    pixels: Mutex<Vec<PixelData>>,
}

impl<S: SpectralAlgebra> RGBFilm<S> {
    /// 构造 RGB 胶片。
    ///
    /// # Panics
    ///
    /// `resolution` 任一维为 0 时 panic。零尺寸胶片无意义。
    pub fn new(
        resolution: UVec2,
        filter: Arc<dyn Filter>,
        sensor: Arc<dyn PixelSensor<S>>,
    ) -> Self {
        assert!(
            resolution.x > 0 && resolution.y > 0,
            "film resolution must be non-zero"
        );
        let count = (resolution.x as usize) * (resolution.y as usize);
        RGBFilm {
            resolution,
            filter,
            sensor,
            pixels: Mutex::new(vec![PixelData::default(); count]),
        }
    }
}

impl<S: SpectralAlgebra> Film<S> for RGBFilm<S> {
    type Output = Image;

    fn add_sample(
        &self,
        pixel: Vec2,
        value: S,
        wavelengths: &S::WavelengthContext,
        weight: f32,
    ) {
        // 光谱 → RGB（低频操作，每样本一次）
        let rgb = self.sensor.process(&value, wavelengths);

        // filter 覆盖的像素范围
        let radius = self.filter.radius();
        let x_min = ((pixel.x - radius.x).floor() as i32).max(0);
        let x_max = ((pixel.x + radius.x).ceil() as i32).min(self.resolution.x as i32 - 1);
        let y_min = ((pixel.y - radius.y).floor() as i32).max(0);
        let y_max = ((pixel.y + radius.y).ceil() as i32).min(self.resolution.y as i32 - 1);

        if x_min > x_max || y_min > y_max {
            // 样本完全在胶片外，忽略
            return;
        }

        let width = self.resolution.x as usize;
        let mut pixels = self.pixels.lock().unwrap();

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let pixel_center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let offset = pixel - pixel_center;
                let filter_w = self.filter.weight(offset);
                if filter_w <= 0.0 {
                    continue;
                }
                let w = filter_w * weight;
                let idx = (y as usize) * width + (x as usize);
                let data = &mut pixels[idx];
                data.rgb_sum.r += rgb.r * w;
                data.rgb_sum.g += rgb.g * w;
                data.rgb_sum.b += rgb.b * w;
                data.weight_sum += w;
            }
        }
    }

    fn resolve(&self) -> Image {
        let pixels = self.pixels.lock().unwrap();
        let mut image = Image::new(self.resolution);

        for y in 0..self.resolution.y {
            for x in 0..self.resolution.x {
                let idx = (y as usize) * (self.resolution.x as usize) + (x as usize);
                let data = pixels[idx];
                let rgb = if data.weight_sum > 0.0 {
                    let inv = 1.0 / data.weight_sum;
                    Rgb::new(
                        data.rgb_sum.r * inv,
                        data.rgb_sum.g * inv,
                        data.rgb_sum.b * inv,
                    )
                } else {
                    Rgb::BLACK
                };
                image.set(x, y, rgb);
            }
        }
        image
    }

    fn resolution(&self) -> UVec2 {
        self.resolution
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use kzray_math::filter::{BoxFilter, TriangleFilter};
    use kzray_abs::sensor::RgbSensor;
    use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb};

    fn make_film(res: UVec2) -> RGBFilm<SpectrumRgb> {
        RGBFilm::new(
            res,
            Arc::new(BoxFilter::new(Vec2::splat(0.5))),
            Arc::new(RgbSensor),
        )
    }

    fn make_film_triangle(res: UVec2) -> RGBFilm<SpectrumRgb> {
        RGBFilm::new(
            res,
            Arc::new(TriangleFilter::new(Vec2::splat(1.0))),
            Arc::new(RgbSensor),
        )
    }

    // ─────────── 构造 ───────────

    #[test]
    #[should_panic(expected = "film resolution must be non-zero")]
    fn reject_zero_resolution() {
        let _ = RGBFilm::<SpectrumRgb>::new(
            UVec2::ZERO,
            Arc::new(BoxFilter::new(Vec2::splat(0.5))),
            Arc::new(RgbSensor),
        );
    }

    #[test]
    fn resolution_query() {
        let film = make_film(UVec2::new(8, 6));
        assert_eq!(film.resolution(), UVec2::new(8, 6));
    }

    // ─────────── 空胶片 ───────────

    #[test]
    fn new_creates_black_image() {
        let film = make_film(UVec2::new(4, 4));
        let image = film.resolve();
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(image.get(x, y), Rgb::BLACK);
            }
        }
    }

    // ─────────── 单样本 ───────────

    #[test]
    fn single_sample_at_pixel_center() {
        let film = make_film(UVec2::new(4, 4));
        film.add_sample(
            Vec2::new(2.5, 2.5),
            SpectrumRgb::new(0.5, 0.6, 0.7),
            &NoWavelengths,
            1.0,
        );
        let image = film.resolve();
        let p = image.get(2, 2);
        assert!((p.r - 0.5).abs() < 1e-5);
        assert!((p.g - 0.6).abs() < 1e-5);
        assert!((p.b - 0.7).abs() < 1e-5);
    }

    #[test]
    fn box_filter_no_bleeding() {
        // BoxFilter radius=0.5 时，只有最近的像素被写入
        let film = make_film(UVec2::new(4, 4));
        film.add_sample(Vec2::new(2.5, 2.5), SpectrumRgb::ONE, &NoWavelengths, 1.0);
        let image = film.resolve();
        assert_eq!(image.get(2, 2), Rgb::WHITE);
        assert_eq!(image.get(1, 2), Rgb::BLACK);
        assert_eq!(image.get(3, 2), Rgb::BLACK);
        assert_eq!(image.get(2, 1), Rgb::BLACK);
        assert_eq!(image.get(2, 3), Rgb::BLACK);
    }

    // ─────────── 多样本 ───────────

    #[test]
    fn multi_sample_average() {
        let film = make_film(UVec2::new(4, 4));
        // 两次样本：白 + 黑，1/spp 权重
        film.add_sample(Vec2::new(2.5, 2.5), SpectrumRgb::ONE, &NoWavelengths, 0.5);
        film.add_sample(Vec2::new(2.5, 2.5), SpectrumRgb::ZERO, &NoWavelengths, 0.5);
        let image = film.resolve();
        let p = image.get(2, 2);
        assert!((p.r - 0.5).abs() < 1e-5);
        assert!((p.g - 0.5).abs() < 1e-5);
        assert!((p.b - 0.5).abs() < 1e-5);
    }

    #[test]
    fn weight_scales_uniformly() {
        // weight=0.5 时，rgb_sum 和 weight_sum 都减半，归一化后仍是原值
        let film = make_film(UVec2::new(4, 4));
        film.add_sample(Vec2::new(2.5, 2.5), SpectrumRgb::ONE, &NoWavelengths, 0.5);
        let image = film.resolve();
        assert_eq!(image.get(2, 2), Rgb::WHITE);
    }

    // ─────────── Filter 空间分配 ───────────

    #[test]
    fn triangle_filter_mixes_neighbors() {
        // 在像素 1 和 2 之间抖动两个样本，验证 filter 权重混合
        let film = make_film_triangle(UVec2::new(4, 4));
        // 样本 1：在 (2.0, 2.5)，白色，权重 0.5
        // 样本 2：在 (2.5, 2.5)，黑色，权重 0.5
        film.add_sample(Vec2::new(2.0, 2.5), SpectrumRgb::ONE, &NoWavelengths, 0.5);
        film.add_sample(Vec2::new(2.5, 2.5), SpectrumRgb::ZERO, &NoWavelengths, 0.5);

        let image = film.resolve();

        // pixel (1, 2) 中心 (1.5, 2.5)
        //   样本 1 offset = (0.5, 0)，TriangleFilter 权重 = (1-0.5)(1-0) = 0.5
        //   样本 2 offset = (-1.0, 0)，权重 = 0（边界）
        //   rgb_sum = 1.0 × 0.5 × 0.5 = 0.25，weight_sum = 0.5 × 0.5 = 0.25
        //   结果 = 0.25 / 0.25 = 1.0
        let p1 = image.get(1, 2);
        assert!((p1.r - 1.0).abs() < 1e-5, "p1 = {:?}", p1);

        // pixel (2, 2) 中心 (2.5, 2.5)
        //   样本 1 offset = (-0.5, 0)，权重 = 0.5
        //   样本 2 offset = (0, 0)，权重 = 1.0
        //   rgb_sum = 1.0 × 0.5 × 0.5 + 0 × 1.0 × 0.5 = 0.25
        //   weight_sum = 0.5 × 0.5 + 1.0 × 0.5 = 0.75
        //   结果 = 0.25 / 0.75 = 0.333
        let p2 = image.get(2, 2);
        assert!((p2.r - 1.0 / 3.0).abs() < 1e-3, "p2 = {:?}", p2);
    }

    #[test]
    fn sample_on_pixel_boundary_affects_both() {
        // BoxFilter radius=0.5，样本在像素边界 (1.0, 1.0) 上
        // 附近像素的 offset 都在 filter 覆盖范围内
        let film = make_film(UVec2::new(4, 4));
        film.add_sample(Vec2::new(1.0, 1.0), SpectrumRgb::ONE, &NoWavelengths, 1.0);
        let image = film.resolve();
        // pixel (0, 0) 中心 (0.5, 0.5)：offset = (0.5, 0.5)，BoxFilter 边界包含
        // pixel (1, 0) 中心 (1.5, 0.5)：offset = (-0.5, 0.5)，边界包含
        // pixel (0, 1) 中心 (0.5, 1.5)：offset = (0.5, -0.5)，边界包含
        // pixel (1, 1) 中心 (1.5, 1.5)：offset = (-0.5, -0.5)，边界包含
        assert_eq!(image.get(0, 0), Rgb::WHITE);
        assert_eq!(image.get(1, 0), Rgb::WHITE);
        assert_eq!(image.get(0, 1), Rgb::WHITE);
        assert_eq!(image.get(1, 1), Rgb::WHITE);
    }

    // ─────────── 越界保护 ───────────

    #[test]
    fn sample_far_outside_ignored() {
        let film = make_film(UVec2::new(4, 4));
        // 完全在胶片外
        film.add_sample(Vec2::new(-100.0, -100.0), SpectrumRgb::ONE, &NoWavelengths, 1.0);
        film.add_sample(Vec2::new(200.0, 200.0), SpectrumRgb::ONE, &NoWavelengths, 1.0);
        let image = film.resolve();
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(image.get(x, y), Rgb::BLACK);
            }
        }
    }

    #[test]
    fn sample_near_edge_clipped() {
        let film = make_film(UVec2::new(4, 4));
        // 靠近边缘，filter 半径可能超出胶片
        film.add_sample(Vec2::new(0.5, 0.5), SpectrumRgb::ONE, &NoWavelengths, 1.0);
        let image = film.resolve();
        assert_eq!(image.get(0, 0), Rgb::WHITE);
    }

    // ─────────── Object-safety / 线程安全 ───────────

    #[test]
    fn film_is_object_safe() {
        let _: Box<dyn Film<SpectrumRgb, Output = Image>> = Box::new(make_film(UVec2::new(4, 4)));
    }

    #[test]
    fn film_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RGBFilm<SpectrumRgb>>();
    }

    #[test]
    fn parallel_add_sample() {
        use std::sync::Arc;
        use std::thread;

        let film = Arc::new(make_film(UVec2::new(4, 4)));
        let mut handles = vec![];

        for _ in 0..4 {
            let film = Arc::clone(&film);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    film.add_sample(
                        Vec2::new(2.5, 2.5),
                        SpectrumRgb::new(i as f32 / 100.0, 0.0, 0.0),
                        &NoWavelengths,
                        0.01,
                    );
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let image = film.resolve();
        let p = image.get(2, 2);
        // 400 个样本，平均值 ≈ 0.495
        assert!(
            (p.r - 0.495).abs() < 0.01,
            "expected ≈0.495, got {}",
            p.r
        );
    }

    // ─────────── 光谱转换 ───────────

    #[test]
    fn sensor_transforms_value() {
        // RgbSensor 是直通，S = SpectrumRgb 直接转成数据层 Rgb
        let film = make_film(UVec2::new(2, 2));
        film.add_sample(
            Vec2::new(1.5, 1.5),
            SpectrumRgb::new(0.1, 0.2, 0.3),
            &NoWavelengths,
            1.0,
        );
        let image = film.resolve();
        let p = image.get(1, 1);
        assert!((p.r - 0.1).abs() < 1e-5);
        assert!((p.g - 0.2).abs() < 1e-5);
        assert!((p.b - 0.3).abs() < 1e-5);
    }
}
