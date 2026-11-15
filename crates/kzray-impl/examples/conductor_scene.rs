//! 端到端示例：GGX 导体（金属）球阵。
//!
//! 运行：
//!
//! ```bash
//! cargo run -p kzray-impl --example conductor_scene --release
//! ```
//!
//! 输出 `conductor_scene.png`。
//!
//! # 场景
//!
//! - 地面：y = -1，黑白棋盘 Lambert；
//! - 4 个金属球，从左到右粗糙度递增：
//!   - 银镜（roughness 0.05）；
//!   - 银磨砂（roughness 0.3）；
//!   - 金光滑（roughness 0.1，F0 金色）；
//!   - 金粗糙（roughness 0.5，F0 金色）；
//! - 环境光：淡蓝灰常量无限远光源；
//! - 点光源：(5, 5, 5)，强度 100；
//! - 相机：(0, 2, 8) 看向 (0, 0, 0)，55° FOV；
//! - 分辨率 800×600，256 spp。
//!
//! # 预期
//!
//! - 光滑银球：清晰的镜面反射——地面棋盘、点光源亮点、环境光色；
//! - 磨砂银球：模糊反射，光源亮点扩散成光斑；
//! - 光滑金球：金色镜面反射（F0 染色）；
//! - 粗糙金球：模糊金色反射。

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
use kzray_impl::material::{ConductorBSDFBuilder, LambertBSDFBuilder};
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

    // ─── 纹理 ───
    //
    // 0: 棋盘
    // 1..4: 四种金属的 F0（银 ×2 相同，金 ×2 相同）
    // 5..8: 四种粗糙度（constant）
    let checker_id = TextureId::from(0);
    let silver_f0_id = TextureId::from(1);
    let gold_f0_id = TextureId::from(2);
    let r005_id = TextureId::from(3);
    let r01_id = TextureId::from(4);
    let r03_id = TextureId::from(5);
    let r05_id = TextureId::from(6);

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
        silver_f0_id,
        Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
            SpectrumRgb::new(0.95, 0.93, 0.88), // 银色 F0
        )),
    );
    builder.library_mut().add_albedo(
        gold_f0_id,
        Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
            SpectrumRgb::new(1.0, 0.766, 0.336), // 金色 F0（PBRT 标准值）
        )),
    );
    builder.library_mut().add_roughness(
        r005_id,
        Box::new(ConstantTexture::<kzray_abs::RoughnessChannel>::new(0.05)),
    );
    builder.library_mut().add_roughness(
        r01_id,
        Box::new(ConstantTexture::<kzray_abs::RoughnessChannel>::new(0.1)),
    );
    builder.library_mut().add_roughness(
        r03_id,
        Box::new(ConstantTexture::<kzray_abs::RoughnessChannel>::new(0.3)),
    );
    builder.library_mut().add_roughness(
        r05_id,
        Box::new(ConstantTexture::<kzray_abs::RoughnessChannel>::new(0.5)),
    );

    // ─── 材质 ───
    let checker_mat = Material::bsdf(Arc::new(LambertBSDFBuilder::new(checker_id)));
    let silver_smooth = Material::bsdf(Arc::new(ConductorBSDFBuilder::new(
        silver_f0_id,
        r005_id,
    )));
    let silver_rough = Material::bsdf(Arc::new(ConductorBSDFBuilder::new(
        silver_f0_id,
        r03_id,
    )));
    let gold_smooth = Material::bsdf(Arc::new(ConductorBSDFBuilder::new(
        gold_f0_id,
        r01_id,
    )));
    let gold_rough = Material::bsdf(Arc::new(ConductorBSDFBuilder::new(
        gold_f0_id,
        r05_id,
    )));

    // ─── 几何 ───
    builder.add_plane(InstanceId::from(0), Plane::horizontal(-1.0), checker_mat);

    // 4 个球，x 从 -2.1 到 2.1
    let radius = 0.7;
    let y = -1.0 + radius;
    builder.add_sphere(InstanceId::from(1), Sphere::new(Vec3::new(-2.1, y, 0.0), radius), silver_smooth);
    builder.add_sphere(InstanceId::from(2), Sphere::new(Vec3::new(-0.7, y, 0.0), radius), silver_rough);
    builder.add_sphere(InstanceId::from(3), Sphere::new(Vec3::new(0.7, y, 0.0), radius), gold_smooth);
    builder.add_sphere(InstanceId::from(4), Sphere::new(Vec3::new(2.1, y, 0.0), radius), gold_rough);

    // ─── 光源 ───
    builder.add_delta_light(
        LightId::from(0),
        AnyDeltaLight::Point(PointLight::new(
            Vec3::new(5.0, 5.0, 5.0),
            SpectrumRgb::new(100.0, 100.0, 100.0),
        )),
    );

    // 环境光：金属球上半反射淡蓝灰
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

    let integrator = PathIntegrator::new(6);

    let config = RenderConfig::new(256, 42);
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

    let output_path = "conductor_scene.png";
    write_image(&image, output_path);
    println!("Wrote {}", output_path);

    Ok(())
}
