//! 端到端示例：镜面球 + 棋盘地面 + 点光源。
//!
//! 运行：
//!
//! ```bash
//! cargo run -p kzray-impl --example mirror_scene --release
//! ```
//!
//! 输出 `mirror_scene.png`。
//!
//! # 场景
//!
//! - 地面：y = -1，黑白棋盘 Lambert（`Wrap::Repeat`）；
//! - 镜面球：原点，半径 1，银色镜面（albedo 0.95）；
//! - 点光源：(5, 5, 5)，强度 100；
//! - 相机：(0, 2, 5) 看向原点，45° FOV；
//! - 分辨率 800×600，128 spp。
//!
//! # 预期
//!
//! 镜面球反射地面棋盘（变形）和点光源（高光亮点）。镜面是 delta
//! BxDF——`PathIntegrator` 的 `specular_bounce = true`，下一跳直接累加
//! 反射方向的贡献，跳过 NEE。
//!
//! # 与 `sphere_scene` 的对比
//!
//! | | `sphere_scene` | `mirror_scene` |
//! |---|---|---|
//! | 球材质 | Lambert（棋盘纹理） | Mirror（银色） |
//! | 球面外观 | 明暗过渡，纹理可见 | 反射环境，变形棋盘 + 光源亮点 |
//! | BSDF 类型 | `Diffuse` | `Mirror`（delta） |
//! | NEE | 参与 MIS | 跳过（`specular_bounce = true`） |

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
use kzray_impl::light::{AnyAreaLight, AnyDeltaLight, AnyInfiniteLight, ConstInfiniteLight, PointLight};
use kzray_impl::material::{LambertBSDFBuilder, MirrorBSDFBuilder};
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

    let floor_id = InstanceId::from(0);
    let mirror_ball_id = InstanceId::from(1);
    let point_light_id = LightId::from(0);

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

    // 纹理：棋盘地面 + 银色 albedo（镜面球反射率）
    let checker_id = TextureId::from(0);
    let silver_id = TextureId::from(1);

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
        silver_id,
        Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
            SpectrumRgb::new(0.95, 0.95, 0.95),
        )),
    );

    let checker_mat = Material::bsdf(Arc::new(LambertBSDFBuilder::new(checker_id)));
    let mirror_mat = Material::bsdf(Arc::new(MirrorBSDFBuilder::new(silver_id)));

    builder.add_plane(floor_id, Plane::horizontal(-1.0), checker_mat);
    builder.add_sphere(mirror_ball_id, Sphere::new(Vec3::ZERO, 1.0), mirror_mat);

    builder.add_delta_light(
        point_light_id,
        AnyDeltaLight::Point(PointLight::new(
            Vec3::new(5.0, 5.0, 5.0),
            SpectrumRgb::new(100.0, 100.0, 100.0),
        )),
    );

    // ─── 环境光：让镜面球上半部分反射"天空" ───
    //
    // 常量无限远光源：所有逃逸方向返回相同辐射亮度。镜面球上半部
    // 反射方向朝上逃逸，累加这个环境光——呈现天空色。
    builder.add_infinite_light(AnyInfiniteLight::Constant(
        ConstInfiniteLight::new(SpectrumRgb::new(0.15, 0.2, 0.3)), // 淡蓝灰
    ));

    let scene: Sc = builder
        .with_light_sampler(UniformLightSampler::new(vec![point_light_id]))
        .build();

    let film = RGBFilm::<SpectrumRgb>::new(
        UVec2::new(WIDTH, HEIGHT),
        Arc::new(BoxFilter::new(Vec2::splat(0.5))),
        Arc::new(RgbSensor),
    );

    let integrator = PathIntegrator::new(5);

    let config = RenderConfig::new(128, 42);
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

    let output_path = "mirror_scene.png";
    write_image(&image, output_path);
    println!("Wrote {}", output_path);

    Ok(())
}
