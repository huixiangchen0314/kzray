//! 端到端示例：粗糙玻璃球阵（磨砂度递增）。
//!
//! 输出 `rough_glass_scene.png`。
//!
//! # 场景
//!
//! 4 个玻璃球，从左到右 roughness 递增：0.02 / 0.1 / 0.25 / 0.5。
//! 棋盘地面 + 点光源 + 淡蓝灰环境光。IOR = 1.5。
//!
//! # 预期
//!
//! - 球 1（roughness 0.02）：近完美玻璃——清晰的倒立地面像 + TIR 边缘环；
//! - 球 2（0.1）：像略糊，边缘环仍在；
//! - 球 3（0.25）：明显磨砂，像模糊不清；
//! - 球 4（0.5）：强磨砂，只看到球形状的透光感。
//! - TODO GGX 多次散射丢失, 走  Kulla-Conty 补偿

use std::sync::Arc;
use std::time::Instant;

use glam::{UVec2, Vec2, Vec3};

use kzray_abs::AlbedoChannel;
use kzray_abs::material::Material;
use kzray_abs::sensor::RgbSensor;
use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
use kzray_data::TextureId;
use kzray_data::common::{InstanceId, LightId};
use kzray_math::filter::BoxFilter;

use kzray_impl::camera::pinhole::{PinholeCamera, PinholeCameraConfig};
use kzray_impl::film::rgb::RGBFilm;
use kzray_impl::geometry::{Plane, Sphere};
use kzray_impl::integrator::PathIntegrator;
use kzray_impl::light::sampler::UniformLightSampler;
use kzray_impl::light::{
    AnyAreaLight, AnyDeltaLight, AnyInfiniteLight, ConstInfiniteLight, PointLight,
};
use kzray_impl::material::{LambertBSDFBuilder, RoughDielectricBSDFBuilder};
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
        camera_to_world: look_at(
            Vec3::new(0.0, 2.0, 8.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::Y,
        ),
        film_size: UVec2::new(WIDTH, HEIGHT),
        fov_y: 55.0_f32.to_radians(),
        aperture: 0.0,
        focus_distance: 1.0,
        shutter_open: 0.0,
        shutter_close: 0.0,
    });

    let mut builder = Builder::new();

    let checker_id = TextureId::from(0);
    let white_tint_id = TextureId::from(1);
    let r002 = TextureId::from(2);
    let r01 = TextureId::from(3);
    let r025 = TextureId::from(4);
    let r05 = TextureId::from(5);

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
        white_tint_id,
        Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
            SpectrumRgb::ONE,
        )),
    );
    for (id, r) in [
        (r002, 0.02_f32),
        (r01, 0.1),
        (r025, 0.25),
        (r05, 0.5),
    ] {
        builder.library_mut().add_roughness(
            id,
            Box::new(ConstantTexture::<kzray_abs::RoughnessChannel>::new(r)),
        );
    }

    let checker_mat = Material::bsdf(Arc::new(LambertBSDFBuilder::new(checker_id)));
    builder.add_plane(InstanceId::from(0), Plane::horizontal(-1.0), checker_mat);

    let radius = 0.7;
    let y = -1.0 + radius;
    for (i, rough_id) in [r002, r01, r025, r05].into_iter().enumerate() {
        let x = -2.1 + 1.4 * i as f32;
        let mat = Material::bsdf(Arc::new(RoughDielectricBSDFBuilder::new(
            1.5,
            rough_id,
            white_tint_id,
        )));
        builder.add_sphere(
            InstanceId::from(1 + i as u64),
            Sphere::new(Vec3::new(x, y, 0.0), radius),
            mat,
        );
    }

    builder.add_delta_light(
        LightId::from(0),
        AnyDeltaLight::Point(PointLight::new(
            Vec3::new(5.0, 5.0, 5.0),
            SpectrumRgb::new(100.0, 100.0, 100.0),
        )),
    );
    builder.add_infinite_light(AnyInfiniteLight::Constant(
        ConstInfiniteLight::new(SpectrumRgb::new(0.15, 0.2, 0.3)),
    ));

    let scene: Sc = builder
        .with_light_sampler(UniformLightSampler::new(vec![LightId::from(0)]))
        .build();

    let film = RGBFilm::<SpectrumRgb>::new(
        UVec2::new(WIDTH, HEIGHT),
        Arc::new(BoxFilter::new(Vec2::splat(0.5))),
        Arc::new(RgbSensor),
    );

    let integrator = PathIntegrator::new(12);
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

    println!("Render complete in {:.2?}", start.elapsed());
    write_image(&image, "rough_glass_scene.png");
    println!("Wrote rough_glass_scene.png");

    Ok(())
}
