//! 端到端示例：球体 + 平面 + 点光源。
//!
//! 运行：
//!
//! ```bash
//! cargo run -p kzray-impl --example sphere_scene --release
//! ```
//!
//! 输出 `sphere_scene.ppm` 到当前目录。
//!
//! # 场景
//!
//! - 球体：原点，半径 1，红色 Lambert（albedo 0.8, 0.2, 0.2）；
//! - 平面：y = -1，灰色 Lambert（albedo 0.5, 0.5, 0.5）；
//! - 点光源：(5, 5, 5)，强度 100；
//! - 相机：(0, 2, 5)，看向原点，45° FOV；
//! - 分辨率 800×600，64 spp。

use std::sync::Arc;
use std::time::Instant;

use glam::{UVec2, Vec2, Vec3};

use kzray_abs::material::Material;
use kzray_abs::sensor::RgbSensor;
use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
use kzray_data::common::LightId;
use kzray_math::filter::BoxFilter;

use kzray_impl::camera::pinhole::{PinholeCamera, PinholeCameraConfig};
use kzray_impl::film::rgb::RGBFilm;
use kzray_impl::geometry::{Plane, Sphere};
use kzray_impl::integrator::PathIntegrator;
use kzray_impl::light::sampler::UniformLightSampler;
use kzray_impl::light::{AnyInfiniteLight, AnyLight, PointLight};
use kzray_impl::material::LambertBSDFBuilder;
use kzray_impl::render::{render, RenderConfig};
use kzray_impl::sampler::PCGSampler;
use kzray_impl::scene::{SimpleScene, SimpleSceneBuilder};
use kzray_impl::util::{look_at, write_image};


fn main() -> Result<(), Box<dyn std::error::Error>> {

    rayon::ThreadPoolBuilder::new().num_threads(12).build_global().unwrap();

    // 类型别名简化
    type Lt = AnyLight<SpectrumRgb>;
    type IL = AnyInfiniteLight<SpectrumRgb>;
    type LS = UniformLightSampler;
    type Sc = SimpleScene<SpectrumRgb, LS, Lt, IL>;

    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 600;

    // ─── 相机 ───
    let camera = PinholeCamera::<SpectrumRgb>::new(PinholeCameraConfig {
        camera_to_world: look_at(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y),
        film_size: UVec2::new(WIDTH, HEIGHT),
        fov_y: 45.0_f32.to_radians(),
        aperture: 0.0,
        focus_distance: 1.0,
        shutter_open: 0.0,
        shutter_close: 0.0,
    });

    // ─── 场景 ───
    let mut builder = SimpleSceneBuilder::<SpectrumRgb, LS, Lt, IL>::new();

    let red_material = Material::bsdf(Arc::new(LambertBSDFBuilder::new(SpectrumRgb::new(
        0.8, 0.2, 0.2,
    ))));
    builder.add_sphere(Sphere::new(Vec3::ZERO, 1.0), red_material);

    let gray_material = Material::bsdf(Arc::new(LambertBSDFBuilder::new(SpectrumRgb::new(
        0.5, 0.5, 0.5,
    ))));
    builder.add_plane(Plane::horizontal(-1.0), gray_material);

    builder.add_light(AnyLight::Point(PointLight::new(
        Vec3::new(5.0, 5.0, 5.0),
        SpectrumRgb::new(100.0, 100.0, 100.0),
    )));

    let scene: Sc = builder
        .with_light_sampler(UniformLightSampler::new(vec![LightId::from(0)]))
        .build();

    // ─── 胶片 ───
    let film = RGBFilm::<SpectrumRgb>::new(
        UVec2::new(WIDTH, HEIGHT),
        Arc::new(BoxFilter::new(Vec2::splat(0.5))),
        Arc::new(RgbSensor),
    );

    // ─── 积分器 ───
    let integrator = PathIntegrator::new(5);

    // ─── 渲染 ───
    let config = RenderConfig::new(64, 42);
    let wavelengths = NoWavelengths;
    let base_sampler = PCGSampler::new(config.seed);

    println!(
        "Rendering {}x{} @ {} spp on {} threads...",
        WIDTH,
        HEIGHT,
        config.samples_per_pixel,
        rayon::current_num_threads()
    );
    let start = Instant::now();

    let image = render(
        &camera,
        &scene,
        &integrator,
        &film,
        base_sampler, // 按值传入，从 &mut sampler 改为 base_sampler
        config,
        &wavelengths,
    );

    let elapsed = start.elapsed();
    println!("Render complete in {:.2?}", elapsed);

    // ─── 输出 ───
    let output_path = "sphere_scene.png";
    write_image(&image, output_path);
    println!("Wrote {}", output_path);

    Ok(())
}
