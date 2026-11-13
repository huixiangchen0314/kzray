//! 针孔相机。
//!
//! 透视投影相机，支持薄透镜景深。相机局部空间：+Z 前方，+Y 上，+X 右。
//!
//! # 投影模型
//!
//! 屏幕窗口在相机局部空间的 `z = 1` 平面上，范围：
//!
//! - x: `[-tan(fov_y/2) × aspect, +tan(fov_y/2) × aspect]`
//! - y: `[-tan(fov_y/2), +tan(fov_y/2)]`
//!
//! 像素样本 `(u, v) ∈ [0,1]²` 映射到屏幕窗口，方向 `w = normalize(p_screen)`
//! 指向屏幕点。
//!
//! # 景深
//!
//! `aperture == 0` 时无景深，光线从相机原点出发。`aperture > 0` 时：
//!
//! - `lens_sample` 采样透镜圆盘上的点 `p_lens`；
//! - 光线从 `p_lens` 出发，指向焦点平面 `z = focus_distance` 上的点
//!   `p_focus = p_screen × focus_distance`；
//! - 焦点平面上的物体清晰，其他位置模糊。
//!
//! # PDF
//!
//! `pdf_wo(ray)` 返回方向的立体角测度 PDF。透视相机的方向分布是连续的
//! （像素采样映射到方向空间），不是 delta。公式：
//!
//! ```text
//! pdf_wo = 1 / (A × w.z³)
//! ```
//!
//! 其中 `A = 4 × tan²(fov_y/2) × aspect` 是屏幕窗口在 z=1 平面上的面积，
//! `w.z` 是归一化方向的 z 分量。边缘方向 `w.z` 更小，pdf 更大——透视
//! 投影在边缘压缩了方向范围，单位立体角的概率更高。
//!
//! **注意**：契约层 `Camera::pdf_wo` 的文档说"针孔相机返回 0（delta）"
//! 是**错误的**。针孔相机的方向不是 delta，是连续的。本实现返回正确
//! 的有限 PDF。契约层文档需要修正。

use glam::{Affine3A, UVec2, Vec2, Vec3};

use kzray_abs::camera::{Camera, CameraSample};
use kzray_abs::ray::Ray;
use kzray_abs::spectrum::SpectralAlgebra;

// ═══════════════════════════════════════════════════════════════
// PinholeCamera
// ═══════════════════════════════════════════════════════════════

/// 透视投影相机（针孔 + 薄透镜）。
///
/// # 构造
///
/// 参数较多，用 [`PinholeCameraConfig`] 聚合。构造时计算 `world_to_camera`
/// 和 `tan_half_fov`，后续采样无三角函数。
///
/// # 泛型参数
///
/// `S: SpectralAlgebra` 用于 `CameraSample::weight` 的类型。针孔相机的
/// `W_e` 恒为 `S::one()`（不随方向变化）；色散相机会按波长改变，当前
/// 不支持。
pub struct PinholeCamera<S: SpectralAlgebra> {
    camera_to_world: Affine3A,
    world_to_camera: Affine3A,
    film_size: UVec2,
    tan_half_fov: f32,
    aspect: f32,
    /// 屏幕窗口在 z=1 平面上的面积。`pdf_wo` 的分母。
    screen_area: f32,
    /// 光圈直径（世界单位）。0 = 无景深。
    aperture: f32,
    /// 焦平面距离（相机局部 z）。
    focus_distance: f32,
    /// 快门开启时刻。`open == close` 表示瞬间快门，无运动模糊。
    shutter_open: f32,
    shutter_close: f32,
    /// 相机权重 `W_e`。针孔相机为 `S::one()`。
    weight: S,
}

/// 针孔相机构造配置。
///
/// 所有时间单位是秒，空间单位是世界单位。
#[derive(Clone, Copy, Debug)]
pub struct PinholeCameraConfig {
    /// camera-to-world 变换。局部空间 +Z 前方，+Y 上，+X 右。
    pub camera_to_world: Affine3A,
    /// 胶片分辨率。
    pub film_size: UVec2,
    /// 垂直视场角（弧度）。必须在 `(0, π)` 内。
    pub fov_y: f32,
    /// 光圈直径（世界单位）。0 = 无景深。
    pub aperture: f32,
    /// 焦平面距离（相机局部 z）。无景深时忽略。
    pub focus_distance: f32,
    /// 快门开启时刻（秒）。
    pub shutter_open: f32,
    /// 快门关闭时刻（秒）。`open == close` 表示瞬间快门。
    pub shutter_close: f32,
}

impl<S: SpectralAlgebra> PinholeCamera<S> {
    /// 构造针孔相机。
    ///
    /// # Panics
    ///
    /// - `fov_y <= 0` 或 `fov_y >= π`；
    /// - `aperture < 0`；
    /// - `aperture > 0` 时 `focus_distance <= 0`；
    /// - `film_size` 任一维为 0；
    /// - `shutter_open > shutter_close`；
    /// - `camera_to_world` 不可逆。
    pub fn new(config: PinholeCameraConfig) -> Self {
        assert!(
            config.fov_y > 0.0 && config.fov_y < std::f32::consts::PI,
            "fov_y must be in (0, π), got {}",
            config.fov_y
        );
        assert!(config.aperture >= 0.0, "aperture must be non-negative");
        if config.aperture > 0.0 {
            assert!(
                config.focus_distance > 0.0,
                "focus_distance must be positive when aperture > 0"
            );
        }
        assert!(
            config.film_size.x > 0 && config.film_size.y > 0,
            "film_size must be non-zero"
        );
        assert!(
            config.shutter_open <= config.shutter_close,
            "shutter_open must be <= shutter_close"
        );

        let tan_half_fov = (config.fov_y * 0.5).tan();
        let aspect = config.film_size.x as f32 / config.film_size.y as f32;
        let screen_area = 4.0 * tan_half_fov * tan_half_fov * aspect;
        let world_to_camera = config.camera_to_world.inverse();

        PinholeCamera {
            camera_to_world: config.camera_to_world,
            world_to_camera,
            film_size: config.film_size,
            tan_half_fov,
            aspect,
            screen_area,
            aperture: config.aperture,
            focus_distance: config.focus_distance,
            shutter_open: config.shutter_open,
            shutter_close: config.shutter_close,
            weight: S::one(),
        }
    }

    /// 屏幕窗口面积（z=1 平面上）。
    pub fn screen_area(&self) -> f32 {
        self.screen_area
    }

    /// 是否有景深。
    pub fn has_depth_of_field(&self) -> bool {
        self.aperture > 0.0
    }
}

impl<S: SpectralAlgebra> Camera<S> for PinholeCamera<S> {
    fn gen_ray(
        &self,
        pixel_sample: Vec2,
        lens_sample: Vec2,
        time: f32,
        _wavelengths: &S::WavelengthContext,
    ) -> Option<CameraSample<S>> {
        // 像素样本 → 屏幕窗口上的点（z=1 平面）
        let p_screen = Vec3::new(
            (2.0 * pixel_sample.x - 1.0) * self.tan_half_fov * self.aspect,
            (2.0 * pixel_sample.y - 1.0) * self.tan_half_fov,
            1.0,
        );

        let (origin_local, dir_local) = if self.aperture > 0.0 {
            // 薄透镜：从透镜点出发，指向焦平面上的 p_screen × focus_distance
            let p_lens = sample_uniform_disk_concentric(lens_sample) * (self.aperture * 0.5);
            let p_focus = p_screen * self.focus_distance;
            let origin = Vec3::new(p_lens.x, p_lens.y, 0.0);
            let dir = (p_focus - origin).normalize();
            (origin, dir)
        } else {
            // 针孔：从相机原点出发，方向 = normalize(p_screen)
            (Vec3::ZERO, p_screen.normalize())
        };

        // 变换到世界空间
        let origin_world = self.camera_to_world.transform_point3(origin_local);
        let dir_world = self.camera_to_world.transform_vector3(dir_local).normalize();

        Some(CameraSample {
            ray: Ray::new(origin_world, dir_world, time),
            film_position: pixel_sample * Vec2::new(self.film_size.x as f32, self.film_size.y as f32),
            pdf: self.pdf_wo(&Ray::new(origin_world, dir_world, time)),
            weight: self.weight,
        })
    }

    fn pdf_wo(&self, ray: &Ray) -> f32 {
        // 方向变换到相机局部空间
        let dir_local = self.world_to_camera.transform_vector3(ray.dir);
        if dir_local.z <= 0.0 {
            return 0.0;
        }
        let dir_local = dir_local.normalize();

        // 投影到 z=1 平面，得到屏幕坐标
        let p_screen_x = dir_local.x / dir_local.z;
        let p_screen_y = dir_local.y / dir_local.z;

        // 归一化到像素坐标
        let u = (p_screen_x / (self.tan_half_fov * self.aspect) + 1.0) * 0.5;
        let v = (p_screen_y / self.tan_half_fov + 1.0) * 0.5;

        // 视锥外（或像素边界外）返回 0
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return 0.0;
        }

        // pdf = 1 / (A × w.z³)
        // A 是屏幕窗口面积，w.z 是归一化方向在局部空间的 z 分量
        1.0 / (self.screen_area * dir_local.z * dir_local.z * dir_local.z)
    }

    fn sample_time(&self, u: f32) -> f32 {
        if self.shutter_open == self.shutter_close {
            // 瞬间快门，无运动模糊
            0.0
        } else {
            // 归一化到快门区间 [0, 1]
            u.clamp(0.0, 1.0)
        }
    }

    fn film_size(&self) -> UVec2 {
        self.film_size
    }
}

// ═══════════════════════════════════════════════════════════════
// 均匀圆盘采样
// ═══════════════════════════════════════════════════════════════

/// 单位圆盘上的均匀采样，concentric 映射（Shirley & Chiu 1997）。
///
/// 输入 `u ∈ [0,1)²`，输出在单位圆盘内均匀分布的点。concentric 映射
/// 比极坐标映射的角向分布更均匀，方差更低。
fn sample_uniform_disk_concentric(u: Vec2) -> Vec2 {
    // 映射到 [-1, 1]²
    let u_offset = 2.0 * u - Vec2::ONE;

    if u_offset == Vec2::ZERO {
        return Vec2::ZERO;
    }

    let (r, theta) = if u_offset.x.abs() > u_offset.y.abs() {
        (
            u_offset.x,
            std::f32::consts::FRAC_PI_4 * (u_offset.y / u_offset.x),
        )
    } else {
        (
            u_offset.y,
            std::f32::consts::FRAC_PI_2
                - std::f32::consts::FRAC_PI_4 * (u_offset.x / u_offset.y),
        )
    };

    r * Vec2::new(theta.cos(), theta.sin())
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb};

    const FILM_W: u32 = 800;
    const FILM_H: u32 = 600;

    fn make_camera() -> PinholeCamera<SpectrumRgb> {
        PinholeCamera::new(PinholeCameraConfig {
            camera_to_world: Affine3A::IDENTITY,
            film_size: UVec2::new(FILM_W, FILM_H),
            fov_y: std::f32::consts::FRAC_PI_4, // 45°
            aperture: 0.0,
            focus_distance: 1.0,
            shutter_open: 0.0,
            shutter_close: 0.0,
        })
    }

    fn make_camera_with_dof() -> PinholeCamera<SpectrumRgb> {
        PinholeCamera::new(PinholeCameraConfig {
            camera_to_world: Affine3A::IDENTITY,
            film_size: UVec2::new(FILM_W, FILM_H),
            fov_y: std::f32::consts::FRAC_PI_4,
            aperture: 0.1,
            focus_distance: 2.0,
            shutter_open: 0.0,
            shutter_close: 0.0,
        })
    }

    // ─────────── 构造 ───────────

    #[test]
    #[should_panic(expected = "fov_y must be")]
    fn reject_invalid_fov() {
        PinholeCamera::<SpectrumRgb>::new(PinholeCameraConfig {
            camera_to_world: Affine3A::IDENTITY,
            film_size: UVec2::new(100, 100),
            fov_y: 0.0,
            aperture: 0.0,
            focus_distance: 1.0,
            shutter_open: 0.0,
            shutter_close: 0.0,
        });
    }

    #[test]
    #[should_panic(expected = "focus_distance must be positive")]
    fn reject_zero_focus_distance_with_aperture() {
        PinholeCamera::<SpectrumRgb>::new(PinholeCameraConfig {
            camera_to_world: Affine3A::IDENTITY,
            film_size: UVec2::new(100, 100),
            fov_y: std::f32::consts::FRAC_PI_4,
            aperture: 0.1,
            focus_distance: 0.0,
            shutter_open: 0.0,
            shutter_close: 0.0,
        });
    }

    #[test]
    #[should_panic(expected = "film_size must be non-zero")]
    fn reject_zero_film_size() {
        PinholeCamera::<SpectrumRgb>::new(PinholeCameraConfig {
            camera_to_world: Affine3A::IDENTITY,
            film_size: UVec2::ZERO,
            fov_y: std::f32::consts::FRAC_PI_4,
            aperture: 0.0,
            focus_distance: 1.0,
            shutter_open: 0.0,
            shutter_close: 0.0,
        });
    }

    // ─────────── gen_ray ───────────

    #[test]
    fn center_pixel_looks_forward() {
        let camera = make_camera();
        let sample = camera
            .gen_ray(Vec2::splat(0.5), Vec2::ZERO, 0.0, &NoWavelengths)
            .unwrap();
        // 中心像素朝 +z
        assert!((sample.ray.dir - Vec3::Z).length() < 1e-5);
        // 起点在原点
        assert!(sample.ray.origin.length() < 1e-5);
    }

    #[test]
    fn corner_pixel_looks_diagonal() {
        let camera = make_camera();
        let sample = camera
            .gen_ray(Vec2::ZERO, Vec2::ZERO, 0.0, &NoWavelengths)
            .unwrap();
        // 左上角：x 负，y 负，z 正
        assert!(sample.ray.dir.x < 0.0);
        assert!(sample.ray.dir.y < 0.0);
        assert!(sample.ray.dir.z > 0.0);
    }

    #[test]
    fn pixel_range_maps_to_screen_window() {
        let camera = make_camera();
        // 右上角（u=1, v=1）方向应该有最大的 x 和 y
        let top_right = camera
            .gen_ray(Vec2::ONE, Vec2::ZERO, 0.0, &NoWavelengths)
            .unwrap();
        assert!(top_right.ray.dir.x > 0.0);
        assert!(top_right.ray.dir.y > 0.0);

        // 左下角方向 x 和 y 都负
        let bottom_left = camera
            .gen_ray(Vec2::ZERO, Vec2::ZERO, 0.0, &NoWavelengths)
            .unwrap();
        assert!(bottom_left.ray.dir.x < 0.0);
        assert!(bottom_left.ray.dir.y < 0.0);
    }

    #[test]
    fn ray_time_filled() {
        let camera = make_camera();
        let sample = camera
            .gen_ray(Vec2::splat(0.5), Vec2::ZERO, 0.7, &NoWavelengths)
            .unwrap();
        assert!((sample.ray.time - 0.7).abs() < 1e-6);
    }

    #[test]
    fn film_position_scaled_to_resolution() {
        let camera = make_camera();
        let sample = camera
            .gen_ray(Vec2::new(0.5, 0.5), Vec2::ZERO, 0.0, &NoWavelengths)
            .unwrap();
        assert!((sample.film_position.x - 400.0).abs() < 1e-3);
        assert!((sample.film_position.y - 300.0).abs() < 1e-3);
    }

    #[test]
    fn weight_is_one() {
        let camera = make_camera();
        let sample = camera
            .gen_ray(Vec2::splat(0.5), Vec2::ZERO, 0.0, &NoWavelengths)
            .unwrap();
        assert_eq!(sample.weight, SpectrumRgb::ONE);
    }

    // ─────────── 景深 ───────────

    #[test]
    fn no_dof_origin_at_camera() {
        let camera = make_camera();
        assert!(!camera.has_depth_of_field());
        let sample = camera
            .gen_ray(Vec2::splat(0.5), Vec2::new(0.3, 0.7), 0.0, &NoWavelengths)
            .unwrap();
        // 无景深时 lens_sample 被忽略，origin 在原点
        assert!(sample.ray.origin.length() < 1e-5);
    }

    #[test]
    fn dof_origin_on_lens_plane() {
        let camera = make_camera_with_dof();
        assert!(camera.has_depth_of_field());
        let sample = camera
            .gen_ray(Vec2::splat(0.5), Vec2::new(0.3, 0.7), 0.0, &NoWavelengths)
            .unwrap();
        // 有景深时 origin 在 z=0 平面，x/y 在透镜范围内
        assert!(sample.ray.origin.z.abs() < 1e-5);
        assert!(sample.ray.origin.length() <= 0.05 + 1e-5); // aperture/2
    }

    #[test]
    fn dof_different_lens_samples_different_origins() {
        let camera = make_camera_with_dof();
        let a = camera
            .gen_ray(Vec2::splat(0.5), Vec2::new(0.1, 0.1), 0.0, &NoWavelengths)
            .unwrap();
        let b = camera
            .gen_ray(Vec2::splat(0.5), Vec2::new(0.9, 0.9), 0.0, &NoWavelengths)
            .unwrap();
        // 不同透镜样本，起点不同
        assert!((a.ray.origin - b.ray.origin).length() > 1e-4);
    }

    #[test]
    fn dof_focus_plane_clear() {
        // 焦点平面上的点应该由所有透镜样本汇聚
        let camera = make_camera_with_dof();
        let focus_z = 2.0; // 与 focus_distance 一致
        // 两个不同透镜样本的射线，在焦点平面上应该相交（近似）
        let a = camera
            .gen_ray(Vec2::splat(0.5), Vec2::new(0.1, 0.5), 0.0, &NoWavelengths)
            .unwrap();
        let b = camera
            .gen_ray(Vec2::splat(0.5), Vec2::new(0.9, 0.5), 0.0, &NoWavelengths)
            .unwrap();
        // 求与 z=focus_z 平面的交点
        let ta = (focus_z - a.ray.origin.z) / a.ray.dir.z;
        let tb = (focus_z - b.ray.origin.z) / b.ray.dir.z;
        let pa = a.ray.origin + a.ray.dir * ta;
        let pb = b.ray.origin + b.ray.dir * tb;
        // 焦点平面上应该汇聚到同一点
        assert!((pa - pb).length() < 1e-3, "focus points: {pa:?} vs {pb:?}");
    }

    // ─────────── pdf_wo ───────────

    #[test]
    fn pdf_positive_inside_frustum() {
        let camera = make_camera();
        let sample = camera
            .gen_ray(Vec2::splat(0.5), Vec2::ZERO, 0.0, &NoWavelengths)
            .unwrap();
        assert!(sample.pdf > 0.0);
    }

    #[test]
    fn pdf_matches_sample() {
        let camera = make_camera();
        for u in [0.1, 0.3, 0.5, 0.7, 0.9] {
            for v in [0.1, 0.5, 0.9] {
                let sample = camera
                    .gen_ray(Vec2::new(u, v), Vec2::ZERO, 0.0, &NoWavelengths)
                    .unwrap();
                let expected = camera.pdf_wo(&sample.ray);
                assert!(
                    (sample.pdf - expected).abs() < 1e-5,
                    "pdf mismatch at ({u}, {v}): {} vs {}",
                    sample.pdf,
                    expected
                );
            }
        }
    }

    #[test]
    fn pdf_zero_behind_camera() {
        let camera = make_camera();
        let ray = Ray::new(Vec3::ZERO, Vec3::NEG_Z, 0.0);
        assert_eq!(camera.pdf_wo(&ray), 0.0);
    }

    #[test]
    fn pdf_zero_outside_frustum() {
        let camera = make_camera();
        // 方向与 -z 成 89 度，几乎垂直于视轴，远在视锥外
        let ray = Ray::new(Vec3::ZERO, Vec3::new(100.0, 0.0, 1.0).normalize(), 0.0);
        assert_eq!(camera.pdf_wo(&ray), 0.0);
    }

    #[test]
    fn pdf_edge_larger_than_center() {
        // 透视投影：边缘方向 pdf 更大（单位立体角的概率更高）
        let camera = make_camera();
        let center = camera
            .gen_ray(Vec2::splat(0.5), Vec2::ZERO, 0.0, &NoWavelengths)
            .unwrap();
        let edge = camera
            .gen_ray(Vec2::new(0.05, 0.5), Vec2::ZERO, 0.0, &NoWavelengths)
            .unwrap();
        assert!(
            edge.pdf > center.pdf,
            "edge pdf {} should be > center pdf {}",
            edge.pdf,
            center.pdf
        );
    }

    // ─────────── sample_time ───────────

    #[test]
    fn sample_time_instant_shutter_returns_zero() {
        let camera = make_camera();
        assert_eq!(camera.sample_time(0.0), 0.0);
        assert_eq!(camera.sample_time(0.5), 0.0);
        assert_eq!(camera.sample_time(1.0), 0.0);
    }

    #[test]
    fn sample_time_interval_returns_u() {
        let camera = PinholeCamera::<SpectrumRgb>::new(PinholeCameraConfig {
            camera_to_world: Affine3A::IDENTITY,
            film_size: UVec2::new(100, 100),
            fov_y: std::f32::consts::FRAC_PI_4,
            aperture: 0.0,
            focus_distance: 1.0,
            shutter_open: 0.0,
            shutter_close: 0.033,
        });
        assert!((camera.sample_time(0.3) - 0.3).abs() < 1e-6);
        assert!((camera.sample_time(0.7) - 0.7).abs() < 1e-6);
    }

    // ─────────── 相机变换 ───────────

    #[test]
    fn camera_translation_applied() {
        let camera = PinholeCamera::<SpectrumRgb>::new(PinholeCameraConfig {
            camera_to_world: Affine3A::from_translation(Vec3::new(10.0, 20.0, 30.0)),
            film_size: UVec2::new(100, 100),
            fov_y: std::f32::consts::FRAC_PI_4,
            aperture: 0.0,
            focus_distance: 1.0,
            shutter_open: 0.0,
            shutter_close: 0.0,
        });
        let sample = camera
            .gen_ray(Vec2::splat(0.5), Vec2::ZERO, 0.0, &NoWavelengths)
            .unwrap();
        assert!((sample.ray.origin - Vec3::new(10.0, 20.0, 30.0)).length() < 1e-5);
    }

    // ─────────── 圆盘采样 ───────────

    #[test]
    fn disk_sample_inside_unit_circle() {
        for i in 0..100 {
            for j in 0..100 {
                let u = Vec2::new(i as f32 / 100.0, j as f32 / 100.0);
                let p = sample_uniform_disk_concentric(u);
                assert!(p.length() <= 1.0 + 1e-5, "point {p:?} outside unit disk");
            }
        }
    }

    #[test]
    fn disk_sample_center_maps_to_center() {
        let p = sample_uniform_disk_concentric(Vec2::splat(0.5));
        assert!(p.length() < 1e-5);
    }

    #[test]
    fn disk_sample_corner_maps_to_edge() {
        // u=(0,0) → u_offset=(-1,-1) → |x|=|y|，走 else 分支
        // 应该映射到圆周上
        let p = sample_uniform_disk_concentric(Vec2::ZERO);
        assert!((p.length() - 1.0).abs() < 1e-5);
    }

    // ─────────── object-safety ───────────

    #[test]
    fn camera_is_object_safe() {
        let _: Box<dyn Camera<SpectrumRgb>> = Box::new(make_camera());
    }

    #[test]
    fn camera_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PinholeCamera<SpectrumRgb>>();
    }
}
