//! 端到端示例：HDR 环境光下的金属球阵。
//!
//! 运行：
//!
//! ```bash
//! cargo run -p kzray-impl --example hdr_scene --release
//! ```
//!
//! 输出 `hdr_scene.png`。
//!
//! # HDR 加载
//!
//! **用户侧负责加载 HDR 文件并转成 `Rgb`**——`HdrImage` 不依赖 `image`
//! crate，也不定义 `Vec3 → S` 的转换规则。本示例的 `load_hdr_as_rgb`
//! 是用户代码的模板。
//!
//! 如果没有 HDR 文件，示例会自动生成一个程序化 HDR（蓝色天空 + 橙色
//! 太阳），无需外部资源也能跑。
//!
//! # 场景
//!
//! - 无地面（只有 HDR 环境 + 点光源 + 4 个金属球）
//! - 4 个金属球：银光滑 / 银粗糙 / 金光滑 / 金粗糙
//! - 点光源：(5, 5, 5)，强度 5（弱补光，不抢 HDR）
//! - HDR 环境光：文件加载优先，否则程序化替代
//! - 相机：拉远 + 低角度，观察 HDR 环境反射
//! - 分辨率 1920×1080，256 spp

use std::sync::Arc;
use std::time::Instant;

use glam::{UVec2, Vec2, Vec3};

use kzray_abs::AlbedoChannel;
use kzray_abs::material::Material;
use kzray_abs::sensor::RgbSensor;
use kzray_abs::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
use kzray_data::TextureId;
use kzray_data::common::{InstanceId, LightId};
use kzray_impl::AnyAreaLight;
use kzray_math::filter::BoxFilter;

use kzray_impl::camera::pinhole::{PinholeCamera, PinholeCameraConfig};
use kzray_impl::film::rgb::RGBFilm;
use kzray_impl::geometry::Sphere;
use kzray_impl::integrator::PathIntegrator;
use kzray_impl::light::sampler::UniformLightSampler;
use kzray_impl::light::{
    AnyDeltaLight, AnyInfiniteLight, HdrImage, HdrInfiniteLight, PointLight,
};
use kzray_impl::material::{ConductorBSDFBuilder};
use kzray_impl::render::{render, RenderConfig};
use kzray_impl::sampler::PCGSampler;
use kzray_impl::scene::{SimpleScene, SimpleSceneBuilder};
use kzray_impl::texture::ConstantTexture;
use kzray_impl::util::{look_at, write_image};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

// ═══════════════════════════════════════════════════════════════
// HDR 加载（用户侧代码模板）
// ═══════════════════════════════════════════════════════════════

/// 从 `.hdr` / `.exr` 文件加载并转成 `HdrImage<Rgb>`。
///
/// 这是**用户代码**——`HdrImage` 不依赖 `image` crate，用户按需引入
/// 并负责 `Vec3 → Rgb` 的转换。换成 `Spectrum<N>` 时只需改 map 的
/// 闭包做 CIE 积分。
fn load_hdr_as_rgb(path: &str) -> Result<HdrImage<SpectrumRgb>, Box<dyn std::error::Error>> {
    let img = image::ImageReader::open(path)?.decode()?;
    let rgb = img.to_rgb32f();
    let (w, h) = (rgb.width(), rgb.height());
    let pixels: Vec<SpectrumRgb> = rgb
        .pixels()
        .map(|p| SpectrumRgb::new(p[0], p[1], p[2]))
        .collect();
    Ok(HdrImage::from_pixels(w, h, pixels))
}

/// 程序化 HDR：落日天空 + 太阳。无外部文件时的替代。
fn procedural_hdr(width: u32, height: u32) -> HdrImage<SpectrumRgb> {
    let mut pixels = Vec::with_capacity((width * height) as usize);

    let sun_dir = Vec3::new(0.35, 0.55, 0.75).normalize();

    let lerp = |a: SpectrumRgb, b: SpectrumRgb, t: f32| -> SpectrumRgb {
        let t = t.clamp(0.0, 1.0);
        SpectrumRgb::new(
            a.0[0] + (b.0[0] - a.0[0]) * t,
            a.0[1] + (b.0[1] - a.0[1]) * t,
            a.0[2] + (b.0[2] - a.0[2]) * t,
        )
    };

    let sky_horizon = SpectrumRgb::new(0.95, 0.60, 0.35);
    let sky_mid     = SpectrumRgb::new(0.30, 0.50, 0.95);
    let sky_zenith  = SpectrumRgb::new(0.04, 0.10, 0.40);
    let ground      = SpectrumRgb::new(0.02, 0.02, 0.03);

    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        let theta = v * std::f32::consts::PI;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let phi = (u - 0.5) * 2.0 * std::f32::consts::PI;
            let dir = Vec3::new(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin());

            let elev = cos_theta.max(0.0);
            let sky = if elev < 0.3 {
                lerp(sky_horizon, sky_mid, elev / 0.3)
            } else {
                lerp(sky_mid, sky_zenith, (elev - 0.3) / 0.7)
            };

            let sun_cos = dir.dot(sun_dir).max(0.0);

            let core_t = ((sun_cos - 0.9990) / 0.0009).clamp(0.0, 1.0);
            let core = core_t * core_t * 80.0;

            let halo_t = ((sun_cos - 0.96) / 0.04).clamp(0.0, 1.0);
            let halo = halo_t * halo_t * halo_t * 2.5;

            let sun = SpectrumRgb::new(
                core + halo,
                core * 0.80 + halo * 0.55,
                core * 0.45 + halo * 0.20,
            );

            let base = if cos_theta > 0.0 { sky } else { ground };
            pixels.push(SpectrumRgb::new(
                base.0[0] + sun.0[0],
                base.0[1] + sun.0[1],
                base.0[2] + sun.0[2],
            ));
        }
    }

    HdrImage::from_pixels(width, height, pixels)
}

// ═══════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════

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

    // ─── 相机：拉远 + 低角度 ───
    let camera = PinholeCamera::<SpectrumRgb>::new(PinholeCameraConfig {
        camera_to_world: look_at(
            Vec3::new(0.0, 0.8, 14.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::Y,
        ),
        film_size: UVec2::new(WIDTH, HEIGHT),
        fov_y: 42.0_f32.to_radians(),
        aperture: 0.0,
        focus_distance: 1.0,
        shutter_open: 0.0,
        shutter_close: 0.0,
    });

    let mut builder = Builder::new();

    // ─── HDR 加载（用户侧）───
    let hdr_path = "studio.hdr";
    let hdr_image = match load_hdr_as_rgb(hdr_path) {
        Ok(img) => {
            println!("Loaded HDR: {}", hdr_path);
            img
        }
        Err(_) => {
            println!("HDR file not found, using procedural HDR");
            procedural_hdr(1024, 512)
        }
    };

    // ─── 纹理 ───
    let silver_f0_id = TextureId::from(1);
    let gold_f0_id   = TextureId::from(2);
    let r005_id      = TextureId::from(3);
    let r01_id       = TextureId::from(4);
    let r03_id       = TextureId::from(5);
    let r05_id       = TextureId::from(6);

    builder.library_mut().add_albedo(
        silver_f0_id,
        Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
            SpectrumRgb::new(0.95, 0.93, 0.88),
        )),
    );
    builder.library_mut().add_albedo(
        gold_f0_id,
        Box::new(ConstantTexture::<AlbedoChannel<SpectrumRgb>>::new(
            SpectrumRgb::new(1.0, 0.766, 0.336),
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

    // ─── 4 个金属球（无地面）───
    let radius = 0.7;
    let y = radius;
    for (i, (f0_id, r_id)) in [
        (silver_f0_id, r005_id),
        (silver_f0_id, r03_id),
        (gold_f0_id,   r01_id),
        (gold_f0_id,   r05_id),
    ]
    .into_iter()
    .enumerate()
    {
        let x = -2.1 + 1.4 * i as f32;
        let mat = Material::bsdf(Arc::new(ConductorBSDFBuilder::new(f0_id, r_id)));
        builder.add_sphere(
            InstanceId::from(1 + i as u64),
            Sphere::new(Vec3::new(x, y, 0.0), radius),
            mat,
        );
    }

    // ─── 点光源：弱补光 ───
    builder.add_delta_light(
        LightId::from(0),
        AnyDeltaLight::Point(PointLight::new(
            Vec3::new(5.0, 5.0, 5.0),
            SpectrumRgb::new(100.0, 80.0, 90.0),
        )),
    );

    // ─── HDR 环境光 ───
    builder.add_infinite_light(AnyInfiniteLight::Hdr(HdrInfiniteLight::new(
        hdr_image,
        1.0,
    )));

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

    println!("Render complete in {:.2?}", start.elapsed());
    write_image(&image, "hdr_scene.png");
    println!("Wrote hdr_scene.png");

    Ok(())
}
