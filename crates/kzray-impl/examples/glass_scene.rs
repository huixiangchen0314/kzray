//! 端到端示例：玻璃球。
//!
//! 运行：
//!
//! ```bash
//! cargo run -p kzray-impl --example glass_scene --release
//! ```
//!
//! 输出 `glass_scene.png`。
//!
//! # 场景
//!
//! - 地面：y = -1，黑白棋盘 Lambert；
//! - 玻璃球：原点，半径 0.8，IOR 1.5，无色透明；
//! - 环境光：淡蓝灰常量无限远光源；
//! - 点光源：(5, 5, 5)，强度 100；
//! - 相机：(0, 2, 5) 看向原点，45° FOV；
//! - 分辨率 800×600，512 spp（玻璃球噪声大，需要足够样本）。
//!
//! # 预期
//!
//! - 玻璃球上可见地面棋盘的**倒立像**（单次折射的球面透镜效应）；
//! - 球边缘有**全内反射**产生的环（TIR 反射）；
//! - 球内部可见多次内反射的复杂光线路径（焦散）；
//! - 与 `mirror_scene` 对比：镜面球反射地面为"正立像"，玻璃球为"倒立像"。

use std::sync::Arc;
use std::time::Instant;

use glam::{UVec2, Vec2, Vec3};

use kzray_abs::material::Material;
use kzray_abs::sensor::RgbSensor;
use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
use kzray_abs::AlbedoChannel;
use kzray_data::common::{InstanceId, LightId};
use kzray_data::TextureId;
use kzray_math::filter::BoxFilter;

use kzray_impl::camera::pinhole::{PinholeCamera, PinholeCameraConfig};
use kzray_impl::film::rgb::RGBFilm;
use kzray_impl::geometry::{Plane, Sphere};
use kzray_impl::integrator::PathIntegrator;
use kzray_impl::light::sampler::UniformLightSampler;
use kzray_impl::light::{
    AnyAreaLight, AnyDeltaLight, AnyInfiniteLight, ConstInfiniteLight, PointLight,
};
use kzray_impl::material::{DielectricBSDFBuilder, LambertBSDFBuilder};
use kzray_impl::render::{render, RenderConfig};
use kzray_impl::sampler::PCGSampler;
use kzray_impl::scene::{SimpleScene, SimpleSceneBuilder};
use kzray_impl::texture::image::Wrap;
use kzray_impl::texture::{ConstantTexture, ImageTexture};
use kzray_impl::util::{look_at, write_image};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

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
    rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .build_global()
        .unwrap();

    type DL = AnyDeltaLight<SpectrumRgb>;
    type IL = AnyInfiniteLight<SpectrumRgb>;
    type AL = AnyAreaLight<SpectrumRgb>;
    type LS = UniformLightSampler;
    type Sc = SimpleScene<SpectrumRgb, LS, DL, IL, AL>;
    type Builder = SimpleSceneBuilder<SpectrumRgb, LS, DL, IL, AL>;

    let camera = PinholeCamera::<SpectrumRgb>::new(PinholeCameraConfig {
        camera_to_world: look_at(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y),
        film_size: UVec2::new(WIDTH, HEIGHT),
        fov_y: 45.0_f32.to_radians(),
        aperture: 0.0,
        focus_distance: 1.0,
        shutter_open: 0.0,
        shutter_close: 0.0,
    });

    let mut builder = Builder::new();

    // 纹理
    let checker_id = TextureId::from(0);
    let white_tint_id = TextureId::from(1);
    let checker_id1 = TextureId::from(2);
    builder.library_mut().add_albedo(
        checker_id,
        Box::new(checker_image(
            64,
            8,
            SpectrumRgb::new(0.8, 0.8, 0.8),
            SpectrumRgb::new(0.15, 0.15, 0.15),
        )),
    );
    builder.library_mut().add_albedo(
        checker_id1,
        Box::new(checker_image(
            64,
            8,
            SpectrumRgb::new(0.8, 0.8, 0.8),
            SpectrumRgb::new(0.15, 0.15, 0.15),
        )),
    );

    builder.library_mut().add_albedo(
        white_tint_id,
        Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
            SpectrumRgb::ONE,
        )),
    );

    let checker_mat = Material::bsdf(Arc::new(LambertBSDFBuilder::new(checker_id)));
    let checker_mat1 = Material::bsdf(Arc::new(LambertBSDFBuilder::new(checker_id1)));
    let glass_mat = Material::bsdf(Arc::new(
        DielectricBSDFBuilder::new(1.5, white_tint_id)));

    builder.add_plane(InstanceId::from(0), Plane::horizontal(-1.0), checker_mat);
    builder.add_sphere(InstanceId::from(1), Sphere::new(Vec3::ZERO, 0.8), glass_mat);
    builder.add_plane(
        InstanceId::from(10),
        Plane::horizontal(3.0),
        checker_mat1, // 或白色 Lambert
    );

    builder.add_delta_light(
        LightId::from(0),
        AnyDeltaLight::Point(PointLight::new(
            Vec3::new(5.0, 5.0, 5.0),
            SpectrumRgb::new(100.0, 100.0, 100.0),
        )),
    );

    builder.add_infinite_light(AnyInfiniteLight::Constant(ConstInfiniteLight::new(
        SpectrumRgb::new(0.15, 0.2, 0.3),
    )));

    let scene: Sc = builder
        .with_light_sampler(UniformLightSampler::new(vec![LightId::from(0)]))
        .build();

    let film = RGBFilm::<SpectrumRgb>::new(
        UVec2::new(WIDTH, HEIGHT),
        Arc::new(BoxFilter::new(Vec2::splat(0.5))),
        Arc::new(RgbSensor),
    );

    // 玻璃需要更深的反弹次数（多次内反射）
    let integrator = PathIntegrator::new(12);

    // 玻璃球噪声大——512 spp 是较低标准
    let config = RenderConfig::new(512, 42);
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
        base_sampler,
        config,
        &wavelengths,
    );

    let elapsed = start.elapsed();
    println!("Render complete in {:.2?}", elapsed);

    let output_path = "glass_scene.png";
    write_image(&image, output_path);
    println!("Wrote {}", output_path);

    Ok(())
}
