//! kzray-impl/examples/cornell_box_scene.rs
//! 端到端示例：Cornell Box 场景。
//!
//! 输出 `cornell_box.png`。
//!
//! # TODO
//!
//! - **Quad 图元**：当前没有 `Quad`，所有四边形（5 墙 + 1 面积光）拆成
//!   两个 `TriangleA`。`Quad` 落地后墙和面积光改为单次注册。
//! - **简化不做**：当前不做任何简化（如统一 `AnyLight`），重构与优化
//!   不同时做。`Scene` 的 `DL` / `IL` / `AL` 三个泛型独立，接口稳定后
//!   再评估是否需要便利抽象。

use std::sync::Arc;
use std::time::Instant;

use glam::{Affine3A, UVec2, Vec2, Vec3, Vec3A};

use kzray_abs::AlbedoChannel;
use kzray_abs::material::Material;
use kzray_abs::sensor::RgbSensor;
use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
use kzray_data::TextureId;
use kzray_data::common::LightId;
use kzray_math::filter::BoxFilter;

use kzray_impl::camera::pinhole::{PinholeCamera, PinholeCameraConfig};
use kzray_impl::film::rgb::RGBFilm;
use kzray_impl::geometry::{AnyShapeA, Sphere, TriangleA};
use kzray_impl::integrator::PathIntegrator;
use kzray_impl::light::sampler::UniformLightSampler;
use kzray_impl::light::{AnyAreaLight, AnyDeltaLight, AnyInfiniteLight, ShapeAreaLight};
use kzray_impl::material::LambertBSDFBuilder;
use kzray_impl::render::{render, RenderConfig};
use kzray_impl::sampler::PCGSampler;
use kzray_impl::scene::{SimpleScene, SimpleSceneBuilder};
use kzray_impl::texture::ConstantTexture;
use kzray_impl::util::{look_at, write_image};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// 四边形拆成两个三角形。
///
/// 顶点顺序 `v0 → v1 → v2 → v3` 沿逆时针（从外部看）绕行。
///
/// **TODO**：`Quad` 图元落地后返回单个 `AnyShapeA::Quad(...)`。
fn quad(v0: Vec3A, v1: Vec3A, v2: Vec3A, v3: Vec3A) -> [AnyShapeA; 2] {
    [
        AnyShapeA::Triangle(TriangleA::new(v0, v1, v2)),
        AnyShapeA::Triangle(TriangleA::new(v0, v2, v3)),
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .build_global()
        .unwrap();

    // ─── 类型别名：三个光源泛型独立 ───
    //
    // `DL` / `IL` / `AL` 分别对应 `Scene` 的三个光源参数。示例不用 delta
    // 光源，但类型仍需指定——`AnyDeltaLight` 可为空。
    type DL = AnyDeltaLight<SpectrumRgb>;
    type IL = AnyInfiniteLight<SpectrumRgb>;
    type AL = AnyAreaLight<SpectrumRgb>;
    type LS = UniformLightSampler;
    type Sc = SimpleScene<SpectrumRgb, LS, DL, IL, AL>;
    type Builder = SimpleSceneBuilder<SpectrumRgb, LS, DL, IL, AL>;

    // ─── 相机 ───
    let camera = PinholeCamera::<SpectrumRgb>::new(PinholeCameraConfig {
        camera_to_world: look_at(Vec3::new(0.0, 0.0, 3.5), Vec3::ZERO, Vec3::Y),
        film_size: UVec2::new(WIDTH, HEIGHT),
        fov_y: 45.0_f32.to_radians(),
        aperture: 0.0,
        focus_distance: 1.0,
        shutter_open: 0.0,
        shutter_close: 0.0,
    });

    // ─── 场景构建器 ───
    let mut builder = Builder::new();

    // ─── 材质纹理：白、红、绿 ───
    let white_id = TextureId::from(0);
    let red_id = TextureId::from(1);
    let green_id = TextureId::from(2);
    builder.library_mut().add_albedo(
        white_id,
        Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
            SpectrumRgb::new(0.8, 0.8, 0.8),
        )),
    );
    builder.library_mut().add_albedo(
        red_id,
        Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
            SpectrumRgb::new(0.8, 0.2, 0.2),
        )),
    );
    builder.library_mut().add_albedo(
        green_id,
        Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
            SpectrumRgb::new(0.2, 0.6, 0.2),
        )),
    );

    let white_mat = || Material::bsdf(Arc::new(LambertBSDFBuilder::new(white_id)));
    let red_mat = || Material::bsdf(Arc::new(LambertBSDFBuilder::new(red_id)));
    let green_mat = || Material::bsdf(Arc::new(LambertBSDFBuilder::new(green_id)));

    // ─── 房间：x, y, z ∈ [-1, 1]，前墙 z=1 留空 ───

    // 地板 y = -1
    // TODO: Quad 图元落地后改为单次 add_shape。
    for tri in quad(
        Vec3A::new(-1.0, -1.0, -1.0),
        Vec3A::new(-1.0, -1.0, 1.0),
        Vec3A::new(1.0, -1.0, 1.0),
        Vec3A::new(1.0, -1.0, -1.0),
    ) {
        builder.add_shape(tri, white_mat());
    }

    // 天花板 y = 1
    // TODO: Quad 图元落地后改为单次 add_shape。
    for tri in quad(
        Vec3A::new(-1.0, 1.0, -1.0),
        Vec3A::new(1.0, 1.0, -1.0),
        Vec3A::new(1.0, 1.0, 1.0),
        Vec3A::new(-1.0, 1.0, 1.0),
    ) {
        builder.add_shape(tri, white_mat());
    }

    // 左墙 x = -1，红
    // TODO: Quad 图元落地后改为单次 add_shape。
    for tri in quad(
        Vec3A::new(-1.0, -1.0, -1.0),
        Vec3A::new(-1.0, 1.0, -1.0),
        Vec3A::new(-1.0, 1.0, 1.0),
        Vec3A::new(-1.0, -1.0, 1.0),
    ) {
        builder.add_shape(tri, red_mat());
    }

    // 右墙 x = 1，绿
    // TODO: Quad 图元落地后改为单次 add_shape。
    for tri in quad(
        Vec3A::new(1.0, -1.0, -1.0),
        Vec3A::new(1.0, -1.0, 1.0),
        Vec3A::new(1.0, 1.0, 1.0),
        Vec3A::new(1.0, 1.0, -1.0),
    ) {
        builder.add_shape(tri, green_mat());
    }

    // 后墙 z = -1，白
    // TODO: Quad 图元落地后改为单次 add_shape。
    for tri in quad(
        Vec3A::new(-1.0, -1.0, -1.0),
        Vec3A::new(1.0, -1.0, -1.0),
        Vec3A::new(1.0, 1.0, -1.0),
        Vec3A::new(-1.0, 1.0, -1.0),
    ) {
        builder.add_shape(tri, white_mat());
    }

    // ─── 球：中心 (0, -0.5, 0)，半径 0.5，白 ───
    builder.add_sphere(Sphere::new(Vec3::new(0.0, -0.5, 0.0), 0.5), white_mat());

    // ─── 面积光：顶部中央小 quad，双面 ───
    //
    // 面积光**不进 primitives**——不参与 intersect，只在 NEE 时被采样。
    // 命中 `Hit::AreaLight` 时 PathIntegrator 查 `scene.area_light(id)`
    // 求 emission。
    //
    // 两个三角形拆成两个面积光——每个独立 `LightId`。Uniform 采样器
    // 均匀选一个三角形再采样。
    //
    // TODO: Quad 图元落地后改为单个 `AnyAreaLight::ShapeArea(...)`，
    // 一次 `add_area_light` 拿一个 id。
    let light_radiance = SpectrumRgb::new(15.0, 15.0, 15.0);
    let mut light_ids: Vec<LightId> = Vec::new();
    for tri in quad(
        Vec3A::new(-0.3, 0.99, -0.3),
        Vec3A::new(0.3, 0.99, -0.3),
        Vec3A::new(0.3, 0.99, 0.3),
        Vec3A::new(-0.3, 0.99, 0.3),
    ) {
        let light = ShapeAreaLight::new(tri, Affine3A::IDENTITY, light_radiance).two_sided(true);
        let id = builder.add_area_light(AnyAreaLight::ShapeArea(light));
        light_ids.push(id);
    }

    // ─── 构建场景 ───
    let scene: Sc = builder
        .with_light_sampler(UniformLightSampler::new(light_ids))
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
        base_sampler,
        config,
        &wavelengths,
    );

    let elapsed = start.elapsed();
    println!("Render complete in {:.2?}", elapsed);

    let output_path = "cornell_box.png";
    write_image(&image, output_path);
    println!("Wrote {}", output_path);

    Ok(())
}
