//! 路径追踪积分器。
//!
//! # 简化
//!
//! - **无 MIS**：当前只有 delta 光源（点光源），NEE 的 MIS 权重恒为 1。
//! - **RR**：`bounces >= RR_START_BOUNCE` 后按 `throughput.max_abs()`
//!   概率存活。见下方常量说明。
//! - **无面积光自发光**：当前无 `AreaLight` 实现，命中光源图元时不自发光。
//!
//! # RR（俄罗斯轮盘赌）
//!
//! 反弹到 `RR_START_BOUNCE` 次后，以 `throughput.max_abs()` 的概率
//! 继续，否则终止；若继续，`throughput /= p` 保持无偏。
//!
//! 前 `RR_START_BOUNCE` 次不 RR——低反弹场景（球 + 平面）的路径在 3
//! 次内结束的概率高，RR 过早会引入噪声而不减时间。
//!
//! `RR_MIN_PROB` 兜底：存活概率低于此值直接终止。严格无偏会得到
//! 极少数 throughput 爆炸的路径（`throughput / 0.001`），方差不减反增；
//! 设阈值是有偏但方差可控的实践做法（PBRT / Cycles 同）。

use glam::Vec2;

use kzray_abs::bsdf::TransportMode;
use kzray_abs::integrator::Integrator;
use kzray_abs::light::{InfiniteLight, LiGeometry, Light, LightSampleContext, LightSampler};
use kzray_abs::ray::Ray;
use kzray_abs::sampler::Sampler;
use kzray_abs::scene::Scene;
use kzray_abs::spectrum::SpectralAlgebra;

// ═══════════════════════════════════════════════════════════════
// RR 配置
// ═══════════════════════════════════════════════════════════════

/// 从第几次反弹开始 RR。
///
/// 前几次反弹路径几乎必然继续，RR 只会引入噪声。3 是低反弹场景的
/// 经验值——球 + 平面场景 90% 的路径在 3 次内结束。
const RR_START_BOUNCE: u32 = 3;

/// 存活概率下限。低于此值直接终止，避免 throughput 爆炸。
const RR_MIN_PROB: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════
// PathIntegrator
// ═══════════════════════════════════════════════════════════════

pub struct PathIntegrator {
    max_depth: u32,
}

impl PathIntegrator {
    pub fn new(max_depth: u32) -> Self {
        PathIntegrator { max_depth }
    }

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
            let bsdf = match material.build_bsdf(
                &hit,
                wo,
                sampler,
                wavelengths,
                scene.library(),
            ) {
                Some(b) => b,
                None => break,
            };

            // ─── NEE（非 delta BSDF 才需要）───
            if !bsdf.is_delta() {
                let ctx = LightSampleContext {
                    position: hit.position,
                    normal: hit.shading_normal,
                    library: scene.library(), 
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
                                        radiance =
                                            radiance.add(&throughput.mul(&contribution));
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

            // ─── RR ───
            //
            // `throughput.max_abs()` 是 `f32`（ScalarAlgebra 保证），
            // 直接与阈值比较。`min(1.0)` 防御 albedo > 1 的场景
            // （物理上不现实但材质系统不禁止）。
            if bounces >= RR_START_BOUNCE {
                let p = throughput.max_abs().min(1.0);
                if p < RR_MIN_PROB {
                    break;
                }
                // 消耗一维采样。注意 `next_f32` 之后无论是否 break，
                // 该维度都已消耗——`dim += 1` 在 break 判断之前。
                let r = sampler.next_f32(dim);
                dim += 1;
                if r >= p {
                    break;
                }
                // 存活：补偿概率损失，保持无偏。
                throughput = throughput.scale((1.0 / p).into());
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

#[inline]
fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    a2 / (a2 + b2)
}
