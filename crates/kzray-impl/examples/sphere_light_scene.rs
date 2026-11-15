//! 端到端示例：球面积光。
//!
//! 运行：
//!
//! ```bash
//! cargo run -p kzray-impl --example sphere_light_scene --release
//! ```
//!
//! 输出 `sphere_light_scene.png`。
//!
//! # 面积光双注册（example 自己协调）
//!
//! 面积光需要**两处注册**，共享同一 `LightId`：
//!
//! - `builder.add_area_light(id, AnyAreaLight::ShapeArea(...))` →
//!   `area_lights`，供 NEE 采样和 emission 求值；
//! - `builder.add_primitive(Primitive::ShapeAreaLight(...), material)` →
//!   `primitives`，供求交。`accel::ShapeAreaLight` 内部持 `light_ref`
//!   指向同一 `LightId`。
//!
//! **builder 不协调两处**——`add_area_light` 和 `add_primitive` 是独立
//! 的底层方法，id 由 example 自己指定和关联。面积光几何进加速器后，
//! 光线可以命中它——发光球在图像里**可见**。
//!
//! # id 分配
//!
//! `InstanceId` / `LightId` 由 example 硬编码（小场景手动分配，
//! 分布式场景由全局分配器分配）。builder 只做冲突检测。

use std::sync::Arc;
use std::time::Instant;

use glam::{Affine3A, UVec2, Vec2, Vec3};

use kzray_abs::{AlbedoChannel, AreaLightBackface};
use kzray_abs::material::Material;
use kzray_abs::sensor::RgbSensor;
use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
use kzray_data::TextureId;
use kzray_data::common::{InstanceId, LightId};
use kzray_math::filter::BoxFilter;

use kzray_impl::accel::accelerator::Primitive;
use kzray_impl::accel::shape::ShapeInstance;
use kzray_impl::accel::shape_area_light::{
     ShapeAreaLight as AccelShapeAreaLight,
};
use kzray_impl::camera::pinhole::{PinholeCamera, PinholeCameraConfig};
use kzray_impl::film::rgb::RGBFilm;
use kzray_impl::geometry::{AnyShapeA, Plane, Sphere};
use kzray_impl::integrator::PathIntegrator;
use kzray_impl::light::sampler::UniformLightSampler;
use kzray_impl::light::{
    AnyAreaLight, AnyDeltaLight, AnyInfiniteLight, ShapeAreaLight,
};
use kzray_impl::material::LambertBSDFBuilder;
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

    // ─── id 硬编码 ───
    let floor_id = InstanceId::from(0);
    let ball_id = InstanceId::from(1);
    let area_light_instance_id = InstanceId::from(2); // 面积光几何也占一个实例
    let area_light_id = LightId::from(0);

    // ─── 相机 ───
    let camera = PinholeCamera::<SpectrumRgb>::new(PinholeCameraConfig {
        camera_to_world: look_at(
            Vec3::new(0.0, 1.5, 5.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
        ),
        film_size: UVec2::new(WIDTH, HEIGHT),
        fov_y: 50.0_f32.to_radians(),
        aperture: 0.0,
        focus_distance: 1.0,
        shutter_open: 0.0,
        shutter_close: 0.0,
    });

    // ─── 场景 ───
    let mut builder = Builder::new();

    let checker_id = TextureId::from(0);
    let blue_id = TextureId::from(1);
    let zero_id = TextureId::from(2);

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
        blue_id,
        Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
            SpectrumRgb::new(0.2, 0.4, 0.8),
        )),
    );
    builder.library_mut().add_albedo(
        zero_id,
        Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
            SpectrumRgb::ZERO,
        )),
    );

    let checker_mat = Material::bsdf(Arc::new(LambertBSDFBuilder::new(checker_id)));
    let blue_mat = Material::bsdf(Arc::new(LambertBSDFBuilder::new(blue_id)));

    // 地面 + 蓝球
    builder.add_plane(floor_id, Plane::horizontal(-1.0), checker_mat);
    builder.add_sphere(ball_id, Sphere::new(Vec3::ZERO, 0.7), blue_mat);

    // ─── 球面积光：example 协调两处注册 ───
    //
    // 面积光几何：中心 (0, 2, 0)，半径 0.4。辐射亮度 30，双面。
    let light_shape = AnyShapeA::Sphere(
        Sphere::new(Vec3::new(0.0, 2.0, 0.0), 0.4).into(),
    );
    let light_transform = Affine3A::IDENTITY;
    let light_radiance = SpectrumRgb::new(8.0, 2.0, 0.5);

    // ── 1. light 视图（NEE + emission）──
    let light_light = ShapeAreaLight::new(light_shape, light_transform, light_radiance)
        .two_sided(true);
    builder.add_area_light(area_light_id, AnyAreaLight::ShapeArea(light_light));

    // ── 2. 几何视图（求交）—— 共享同一 LightId ──
    let light_instance = ShapeInstance::new(area_light_instance_id, light_shape, light_transform);
    let accel_light = AccelShapeAreaLight {
        instance: light_instance,
        light_id: area_light_id,
        backface: AreaLightBackface::TwoSided,
    };
    let light_material = Material::bsdf(Arc::new(LambertBSDFBuilder::new(zero_id)));
    builder.add_primitive(Primitive::ShapeAreaLight(accel_light), light_material);

    // ─── 构建场景 ───
    let scene: Sc = builder
        .with_light_sampler(UniformLightSampler::new(vec![area_light_id]))
        .build();

    // ─── 胶片 ───
    let film = RGBFilm::<SpectrumRgb>::new(
        UVec2::new(WIDTH, HEIGHT),
        Arc::new(BoxFilter::new(Vec2::splat(0.5))),
        Arc::new(RgbSensor),
    );

    // ─── 积分器 ───
    let integrator = PathIntegrator::new(8);

    // ─── 渲染 ───
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

    let output_path = "sphere_light_scene.png";
    write_image(&image, output_path);
    println!("Wrote {}", output_path);

    Ok(())
}
