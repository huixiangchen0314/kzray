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

use kzray_abs::AlbedoChannel;
use kzray_abs::material::Material;
use kzray_abs::sensor::RgbSensor;
use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
use kzray_data::TextureId;
use kzray_data::common::LightId;
use kzray_impl::texture::image::Wrap;
use kzray_impl::ImageTexture;
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

/// 构造 N×N 棋盘图像，每 `cell` 像素一格，交替 `a` / `b`。
fn checker_image(
    size: u32,
    cell: u32,
    a: SpectrumRgb,
    b: SpectrumRgb,
) -> ImageTexture<AlbedoChannel<SpectrumRgb>> {
    let mut data = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        for x in 0..size {
            let cx = x / cell;
            let cy = y / cell;
            data.push(if (cx + cy) & 1 == 0 { a } else { b });
        }
    }
    ImageTexture::from_values_with_wrap(size, size, data, Wrap::Repeat)
}

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
    let checker_tex0 = TextureId::from(0);
    let checker_tex1 = TextureId::from(1);
    // 先注册 albedo 纹理，再构造材质
    builder.library_mut().add_albedo(
        checker_tex0,
        Box::new(checker_image(
            64, 8,
            SpectrumRgb::new(0.8, 0.2, 0.2),
            SpectrumRgb::new(0.9, 0.9, 0.9),
        )),
    );
    builder.library_mut().add_albedo(
        checker_tex1,
                Box::new(checker_image(
            64, 8,
            SpectrumRgb::new(0.2, 0.4, 0.8),
            SpectrumRgb::new(0.9, 0.9, 0.9),
        ).with_wrap(Wrap::Clamp)),
    );

    let checker_mat0 = Material::bsdf(Arc::new(
        LambertBSDFBuilder::new(checker_tex0)));
    let checker_mat1 = Material::bsdf(Arc::new(
        LambertBSDFBuilder::new(checker_tex1)));
    
    builder.add_plane(Plane::horizontal(-1.0), checker_mat0);
    builder.add_sphere(Sphere::new(Vec3::ZERO, 1.0), checker_mat1);
    
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
