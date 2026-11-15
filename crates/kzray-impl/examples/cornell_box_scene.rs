//! kzray-impl/examples/cornell_box_scene.rs
//! 端到端示例：Cornell Box 场景。
//!
//! 运行：
//!
//! ```bash
//! cargo run -p kzray-impl --example cornell_box_scene --release
//! ```
//!
//! 输出 `cornell_box.png` 到当前目录。
//!
//! # 场景
//!
//! - 房间：`x, y, z ∈ [-1, 1]` 的正方体，含地板 / 天花板 / 左右墙 / 后墙
//!   （前墙 `z = 1` 留空作为相机口）；
//! - 左墙：红色 Lambert；右墙：绿色 Lambert；
//! - 地板 / 天花板 / 后墙：白色 Lambert；
//! - 球：中心 `(0, -0.5, 0)`，半径 `0.5`，白色 Lambert；
//! - 面积光：顶部中央（`y = 0.99`）的 quad，双面发光，`radiance = 15`；
//! - 相机：`(0, 0, 3.5)` 看向原点，45° FOV；
//! - 分辨率 800×600，64 spp。
//!
//! # 预期行为
//!
//! 面积光**不进 `primitives`**——它不参与 `intersect`，只在 NEE 时被
//! 采样。因此**面光源本身在图像里不可见**（顶部会显示为天花板，不是
//! 白色发光面）。这是当前阶段的预期行为——面光源只贡献间接光照。要
//! 让面光源本身可见需要"自发光材质"（阶段 4 或之后）。
//!
//! # TODO: 解析形状不够，没有Quad图元，墙面不是Cornell模型的标准
//!
//! 当前没有 `Quad` 图元，所有四边形（5 个墙 + 1 个面积光）都拆成两个
//! `TriangleA`。每处 `for tri in quad(...)` 出现的地方即是待替换点。
//! `Quad` 图元落地后：
//!
//! - `quad(...)` 返回 `AnyShapeA::Quad(...)` 单个值；
//! - 墙循环改为 `builder.add_shape(quad(...), mat)`；
//! - 面积光循环改为一次 `builder.add_light(AnyLight::ShapeArea(...))`；
//! - `light_ids` 从 `(0..2)` 变为单个 `LightId`；
//! - `Quad` 的面积光采样天然覆盖整个 quad，无两三角形边界不连续。

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
use kzray_impl::light::{AnyInfiniteLight, AnyLight, ShapeAreaLight};
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
/// 顶点顺序 `v0 → v1 → v2 → v3` 沿逆时针（从外部看）绕行，两个三角形
/// `(v0, v1, v2)` 和 `(v0, v2, v3)` 的法线都朝外。
///
/// **TODO**：待实现 `Quad` 图元后，改为返回单个 `AnyShapeA::Quad(...)`。
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

    type Lt = AnyLight<SpectrumRgb>;
    type IL = AnyInfiniteLight<SpectrumRgb>;
    type LS = UniformLightSampler;
    type Sc = SimpleScene<SpectrumRgb, LS, Lt, IL>;

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

    // ─── 场景 ───
    let mut builder = SimpleSceneBuilder::<SpectrumRgb, LS, Lt, IL>::new();

    // 注册 albedo 纹理：白、红、绿（Cornell Box 的经典三色）
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

    // `Material` 无 `Clone`（内包 `Arc<dyn BSDFBuilder>`），用闭包每次
    // 重新构造——`LambertBSDFBuilder` 是 `Copy` 的，代价极小。
    let white_mat = || Material::bsdf(Arc::new(LambertBSDFBuilder::new(white_id)));
    let red_mat = || Material::bsdf(Arc::new(LambertBSDFBuilder::new(red_id)));
    let green_mat = || Material::bsdf(Arc::new(LambertBSDFBuilder::new(green_id)));

    // ─── 房间：x, y, z ∈ [-1, 1] ───
    //
    // 前墙（z = 1）留空作为相机口。顶点顺序沿逆时针（从外部看）绕行，
    // 法线朝内。

    // 地板 y = -1，法线 +Y
    // TODO: Quad 图元落地后改为单次 add_shape。
    for tri in quad(
        Vec3A::new(-1.0, -1.0, -1.0),
        Vec3A::new(-1.0, -1.0, 1.0),
        Vec3A::new(1.0, -1.0, 1.0),
        Vec3A::new(1.0, -1.0, -1.0),
    ) {
        builder.add_shape(tri, white_mat());
    }

    // 天花板 y = 1，法线 -Y
    // TODO: Quad 图元落地后改为单次 add_shape。
    for tri in quad(
        Vec3A::new(-1.0, 1.0, -1.0),
        Vec3A::new(1.0, 1.0, -1.0),
        Vec3A::new(1.0, 1.0, 1.0),
        Vec3A::new(-1.0, 1.0, 1.0),
    ) {
        builder.add_shape(tri, white_mat());
    }

    // 左墙 x = -1，法线 +X，红
    // TODO: Quad 图元落地后改为单次 add_shape。
    for tri in quad(
        Vec3A::new(-1.0, -1.0, -1.0),
        Vec3A::new(-1.0, 1.0, -1.0),
        Vec3A::new(-1.0, 1.0, 1.0),
        Vec3A::new(-1.0, -1.0, 1.0),
    ) {
        builder.add_shape(tri, red_mat());
    }

    // 右墙 x = 1，法线 -X，绿
    // TODO: Quad 图元落地后改为单次 add_shape。
    for tri in quad(
        Vec3A::new(1.0, -1.0, -1.0),
        Vec3A::new(1.0, -1.0, 1.0),
        Vec3A::new(1.0, 1.0, 1.0),
        Vec3A::new(1.0, 1.0, -1.0),
    ) {
        builder.add_shape(tri, green_mat());
    }

    // 后墙 z = -1，法线 +Z，白
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

    // ─── 面积光：顶部中央的小 quad，双面 ───
    //
    // y = 0.99，略低于天花板避免共面。边长 0.6，面积 0.36。
    //
    // 面积光**不进 primitives**——它不参与 intersect，只在 NEE 时被采样。
    //
    // TODO: Quad 图元落地后改为单个 ShapeAreaLight（一次 add_light）。
    let light_radiance = SpectrumRgb::new(15.0, 15.0, 15.0);
    for tri in quad(
        Vec3A::new(-0.3, 0.99, -0.3),
        Vec3A::new(0.3, 0.99, -0.3),
        Vec3A::new(0.3, 0.99, 0.3),
        Vec3A::new(-0.3, 0.99, 0.3),
    ) {
        let light = ShapeAreaLight::new(tri, Affine3A::IDENTITY, light_radiance).two_sided(true);
        builder.add_light(AnyLight::ShapeArea(light));
    }

    // ─── 光源采样器 ───
    //
    // 面积光拆成两个三角形，每个都是独立光源。Uniform 采样器均匀选中
    // 一个三角形，再在该三角形内均匀采样。两三角形面积相等时，等价于
    // 整个 quad 上均匀采样。
    //
    // TODO: Quad 图元落地后只需一个 LightId（面积光不再拆分）。
    let light_ids: Vec<LightId> = (0..2).map(|i| LightId::from(i as u64)).collect();
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

    // ─── 输出 ───
    let output_path = "cornell_box.png";
    write_image(&image, output_path);
    println!("Wrote {}", output_path);

    Ok(())
}
