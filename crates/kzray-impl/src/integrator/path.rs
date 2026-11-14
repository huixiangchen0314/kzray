//! 路径追踪积分器。
//!
//! 经典路径追踪：每次反弹做 NEE（光源采样）+ 自发光累加 + BSDF 采样。
//!
//! # 简化
//!
//! - **无 MIS**：当前只有 delta 光源（点光源），NEE 的 MIS 权重恒为 1。
//! - **无 RR**：用 `max_depth` 截断。RR 留作阶段 6。
//! - **无面积光自发光**：当前无 `AreaLight` 实现，命中光源图元时不自发光。
//!
//! # 自发光规则
//!
//! `specular_bounce` 标志控制无限远光源的自发光累加：
//!
//! - 初始 `true`（第一次反弹，直接从相机看到环境光，必须累加）；
//! - 每次 BSDF 采样后，`specular_bounce = bs.flags.is_pure_delta()`；
//! - 光线逃逸到无限远时，`specular_bounce == true` 才累加环境光。
//!
//! # 维度追踪
//!
//! ```text
//! 每次反弹：
//!   LightSampler::sample(ctx, sampler, dim)      消耗 c1
//!   sampler.next_2d(dim + c1)                    消耗 2
//!   BSDF::sample_f(sampler, dim + c1 + 2)        消耗 c2
//!   dim += c1 + 2 + c2
//! ```

use glam::Vec2;

use kzray_abs::bsdf::TransportMode;
use kzray_abs::integrator::Integrator;
use kzray_abs::light::{InfiniteLight, LiGeometry, Light, LightSampleContext, LightSampler};
use kzray_abs::ray::Ray;
use kzray_abs::sampler::Sampler;
use kzray_abs::scene::Scene;
use kzray_abs::spectrum::SpectralAlgebra;

// ═══════════════════════════════════════════════════════════════
// PathIntegrator
// ═══════════════════════════════════════════════════════════════

/// 路径追踪积分器。
///
/// 无泛型参数——`S` / `LS` 在 `impl Integrator<S, LS>` 上。`PathIntegrator`
/// 本身只持有配置（`max_depth`），不依赖光谱类型或采样器类型。
pub struct PathIntegrator {
    max_depth: u32,
}

impl PathIntegrator {
    /// 构造路径追踪积分器。
    ///
    /// `max_depth` 是最大反弹次数。典型值 5-10。
    pub fn new(max_depth: u32) -> Self {
        PathIntegrator { max_depth }
    }

    /// 最大反弹次数。
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }
}

impl<S, LS, SC, L, IL> Integrator<S, LS, SC, L, IL> for PathIntegrator
where
    S: SpectralAlgebra,
    LS: LightSampler,
    SC: Scene<S, LS, L, IL>,
    L: Light<S>,
    IL: InfiniteLight<S>,
{
    fn li<SM: Sampler>(
        &self,
        scene: &SC,
        ray: &Ray,
        sampler: &mut SM,
        start_dim: u32,
        wavelengths: &S::WavelengthContext,
    ) -> S {
        let mut radiance = S::zero();
        let mut throughput = S::one();
        let mut current_ray = *ray;
        let mut specular_bounce = true;
        let mut bounces = 0u32;
        let mut dim = start_dim;

        loop {
            
            // ─── 求交 ───
            let hit = match scene.intersect(&current_ray, 0.0..f32::INFINITY) {
                Some(h) => h,
                None => {
                    if specular_bounce {
                        if let Some(inf) = scene.infinite_light() {
                            let le = inf.emitted_radiance(&current_ray, wavelengths);
                            radiance = radiance.add(&throughput.mul(&le));
                        }
                    }
                    break;
                }
            };
            // ─── 材质 → BSDF ───
            let material = match scene.material(&hit) {
                Some(m) => m,
                None => break,
            };

            let wo = -current_ray.dir.normalize();
            let bsdf = match material.build_bsdf(&hit, wo, sampler, wavelengths) {
                Some(b) => b,
                None => break,
            };

            // ─── NEE（非 delta BSDF 才需要）───
            if !bsdf.is_delta() {
                let ctx = LightSampleContext {
                    position: hit.position,
                    normal: hit.shading_normal,
                };

                if let Some(light_sampler) = scene.light_sampler() {
                    if let Some(sel) = light_sampler.sample(&ctx, sampler, dim) {
                        dim += sel.consumed;

                        if let Some(light) = scene.light(sel.light_id) {
                            let (dim1, dim2) = sampler.next_2d(dim);
                            dim += 2;

                            if let Some(li) =
                                light.sample_li(&ctx, Vec2::new(dim1, dim2), wavelengths)
                            {
                                
                                // 阴影射线
                                let occluded = match li.geometry {
                                    LiGeometry::Surface { position, .. } => {
                                        let to_light = position - hit.position;
                                        let distance = to_light.length();
                                        if distance <= 1e-6 {
                                            true
                                        } else {
                                            let shadow_origin = hit.offset_origin(1.0, 1e-3);
                                            let shadow_dir = to_light / distance;
                                            let shadow_ray = Ray::new(
                                                shadow_origin,
                                                shadow_dir,
                                                current_ray.time,
                                            );
                                            scene.intersect_p(&shadow_ray, 1e-3..distance - 1e-3)
                                        }
                                    }
                                    LiGeometry::Direction => false,
                                };

                                if !occluded {
                                    let f = bsdf.eval(li.wi, TransportMode::Radiance);
                                    let cos_theta = ctx.normal.dot(li.wi).max(0.0);

                                     // 关键修复：delta 光源的 li.pdf 无有限值（填 0.0
                                    // 作为占位），不能直接乘进 light_pdf。delta 光源的
                                    // 方向由几何唯一确定，采样 PDF 就是"选择该光源的
                                    // 概率" sel.light_pdf。
                                    let is_delta = light.light_type().is_delta();
                                    let light_pdf = if is_delta {
                                        sel.light_pdf
                                    } else {
                                        sel.light_pdf * li.pdf
                                    };

                                    if light_pdf > 0.0 {
                                        let mis_weight = if is_delta {
                                            1.0
                                        } else {
                                            let bsdf_pdf =
                                                bsdf.pdf(li.wi, TransportMode::Radiance);
                                            power_heuristic(light_pdf, bsdf_pdf)
                                        };

                                        let contribution = f.mul(&li.incident).scale(
                                            S::Scalar::from(cos_theta * mis_weight / light_pdf),
                                        );
                                        radiance = radiance.add(&throughput.mul(&contribution));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ─── BSDF 采样 ───
            let bs = match bsdf.sample_f(sampler, dim, TransportMode::Radiance) {
                Some(b) => b,
                None => break,
            };
            dim += bs.consumed;

            throughput = throughput.mul(&bs.weight);
            specular_bounce = bs.flags.is_pure_delta();
            bounces += 1;

            if bounces >= self.max_depth {
                break;
            }

            // ─── 更新光线 ───
            let new_origin = hit.offset_origin(1.0, 1e-3);
            current_ray = Ray::new(new_origin, bs.wi, current_ray.time);
        }

        radiance
    }
}

// ═══════════════════════════════════════════════════════════════
// Power Heuristic
// ═══════════════════════════════════════════════════════════════

/// Power heuristic（β=2）：`pdf_a² / (pdf_a² + pdf_b²)`。
#[inline]
fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    a2 / (a2 + b2)
}
