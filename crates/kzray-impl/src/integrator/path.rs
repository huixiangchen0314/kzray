// kzray-impl/src/integrator/path.rs

use glam::{Vec2, Vec3};

use kzray_abs::bsdf::TransportMode;
use kzray_abs::integrator::Integrator;
use kzray_abs::light::{
    AreaLight, InfiniteLight, LiGeometry, Light, LightSampleContext, LightSampler,
};
use kzray_abs::ray::Ray;
use kzray_abs::sampler::Sampler;
use kzray_abs::scene::Scene;
use kzray_abs::spectrum::SpectralAlgebra;

const RR_START_BOUNCE: u32 = 3;
const RR_MIN_PROB: f32 = 0.05;

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

impl<S, LS, SC, DL, IL, AL> Integrator<S, LS, SC, DL, IL, AL> for PathIntegrator
where
    S: SpectralAlgebra,
    LS: LightSampler,
    SC: Scene<S, LS, DL, IL, AL>,
    DL: Light<S>,
    IL: InfiniteLight<S>,
    AL: AreaLight<S>,
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
        let mut prev_bsdf: Option<(Vec3, Vec3, f32)> = None;

        loop {
            // ─── 求交 ───
            let hit = match scene.intersect(&current_ray, 0.0..f32::INFINITY) {
                Some(h) => h,
                None => {
                    // 逃逸光线：只有 specular_bounce 时才累加无限远光源。
                    // 无限远光源是 `&[IL]`——遍历求和（环境光 + 天空光叠加）。
                    if specular_bounce {
                        let mut le = S::zero();
                        for inf in scene.infinite_lights() {
                            le = le.add(&inf.emitted_radiance(&current_ray, wavelengths));
                        }
                        radiance = radiance.add(&throughput.mul(&le));
                    }
                    break;
                }
            };
            let surface = *hit.surface().expect(
                "PathIntegrator 当前只处理表面命中；Curve/Volume 落地后应在此 match 分流",
            );
            let wo = surface.wo(&current_ray);

            // ─── 材质 → BSDF ───
            let material = match scene.material(&surface) {
                Some(m) => m,
                None => break,
            };
            let bsdf =
                match material.build_bsdf(&surface, wo, sampler, wavelengths, scene.library()) {
                    Some(b) => b,
                    None => break,
                };

            // ─── 面积光 emission ───
            if let Some(light_id) = hit.light_id() {
                let area_light = scene
                    .area_light(light_id)
                    .expect("hit.light_id() 返回的 id 必然在 area_lights 中");

                let le = area_light.emitted_radiance(
                    surface.position,
                    surface.shading_normal,
                    surface.uv,
                    wo,
                    wavelengths,
                );

                let mis_weight = match prev_bsdf {
                    // primary ray 或上次 BSDF 采样是 delta：NEE 采不到，
                    // BSDF 采样是唯一命中方式，全量累加。
                    None => 1.0,
                    Some((prev_pos, prev_norm, bsdf_pdf)) => {
                        // BSDF 采样和 NEE 都可能命中，MIS 加权。
                        // `light_pdf` 用 `area_light.pdf_li`——面积光的
                        // NEE 采样密度（与 `sample_li` 返回的 `pdf` 一致）。
                        let ctx = LightSampleContext {
                            position: prev_pos,
                            normal: prev_norm,
                            library: scene.library(),
                        };
                        let wi_to_light = current_ray.dir.normalize();
                        let light_pdf = scene
                            .light_sampler()
                            .map(|ls| {
                                ls.pdf(&ctx, light_id) * area_light.pdf_li(&ctx, wi_to_light)
                            })
                            .unwrap_or(0.0);
                        power_heuristic(bsdf_pdf, light_pdf)
                    }
                };

                radiance =
                    radiance.add(&throughput.mul(&le).scale(S::Scalar::from(mis_weight)));
                break;
            }

            // ─── NEE（非 delta BSDF 才需要）───
            if !bsdf.is_delta() {
                let ctx = LightSampleContext {
                    position: surface.position,
                    normal: surface.shading_normal,
                    library: scene.library(),
                };

                if let Some(light_sampler) = scene.light_sampler() {
                    if let Some(sel) = light_sampler.sample(&ctx, sampler, dim) {
                        dim += sel.consumed;

                        let (dim1, dim2) = sampler.next_2d(dim);
                        dim += 2;
                        let sample_2d = Vec2::new(dim1, dim2);

                        // ── Delta 光源 NEE ──
                        //
                        // `mis_weight = 1.0`——delta 光源方向由几何唯一确定，
                        // BSDF 采样命中同一方向的概率为零（连续 pdf 对点/
                        // 方向采不到），不需要 MIS。
                        if let Some(light) = scene.delta_light(sel.light_id) {
                            if let Some(li) = light.sample_li(&ctx, sample_2d, wavelengths) {
                                let occluded = match li.geometry {
                                    LiGeometry::Surface { position, distance } => {
                                        if distance <= 1e-6 {
                                            true
                                        } else {
                                            let shadow_origin =
                                                surface.offset_origin(1.0, 1e-3);
                                            let shadow_dir =
                                                (position - surface.position) / distance;
                                            let shadow_ray = Ray::new(
                                                shadow_origin,
                                                shadow_dir,
                                                current_ray.time,
                                            );
                                            scene.intersect_p(
                                                &shadow_ray,
                                                1e-3..distance - 1e-3,
                                            )
                                        }
                                    }
                                    LiGeometry::Direction => false,
                                };

                                if !occluded {
                                    let f = bsdf.eval(li.wi, TransportMode::Radiance);
                                    let cos_theta = ctx.normal.dot(li.wi).max(0.0);
                                    // delta 光源：只用选择概率，`li.pdf` 无意义
                                    let light_pdf = sel.light_pdf;
                                    if light_pdf > 0.0 {
                                        let contribution = f.mul(&li.incident).scale(
                                            S::Scalar::from(cos_theta / light_pdf),
                                        );
                                        radiance =
                                            radiance.add(&throughput.mul(&contribution));
                                    }
                                }
                            }
                        }
                        // ── 面积光 NEE ──
                        //
                        // `mis_weight` 用 `power_heuristic(light_pdf, bsdf_pdf)`——
                        // BSDF 采样也可能命中同一面积光，两条路径按 pdf 加权。
                        else if let Some(area_light) = scene.area_light(sel.light_id) {
                            if let Some(li) = area_light.sample_li(&ctx, sample_2d, wavelengths) {
                                let occluded = match li.geometry {
                                    LiGeometry::Surface { position, distance } => {
                                        if distance <= 1e-6 {
                                            true
                                        } else {
                                            let shadow_origin =
                                                surface.offset_origin(1.0, 1e-3);
                                            let shadow_dir =
                                                (position - surface.position) / distance;
                                            let shadow_ray = Ray::new(
                                                shadow_origin,
                                                shadow_dir,
                                                current_ray.time,
                                            );
                                            scene.intersect_p(
                                                &shadow_ray,
                                                1e-3..distance - 1e-3,
                                            )
                                        }
                                    }
                                    LiGeometry::Direction => false,
                                };

                                if !occluded {
                                    let f = bsdf.eval(li.wi, TransportMode::Radiance);
                                    let cos_theta = ctx.normal.dot(li.wi).max(0.0);
                                    // 面积光：选择概率 × 光源内部采样 pdf
                                    let light_pdf = sel.light_pdf * li.pdf;
                                    if light_pdf > 0.0 {
                                        let bsdf_pdf =
                                            bsdf.pdf(li.wi, TransportMode::Radiance);
                                        let mis_weight =
                                            power_heuristic(light_pdf, bsdf_pdf);
                                        let contribution = f.mul(&li.incident).scale(
                                            S::Scalar::from(
                                                cos_theta * mis_weight / light_pdf,
                                            ),
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
            if bounces >= RR_START_BOUNCE {
                let p = throughput.max_abs().min(1.0);
                if p < RR_MIN_PROB {
                    break;
                }
                let r = sampler.next_f32(dim);
                dim += 1;
                if r >= p {
                    break;
                }
                throughput = throughput.scale((1.0 / p).into());
            }

            // ─── 保存 prev_bsdf（供下次面积光 MIS）───
            prev_bsdf = if specular_bounce {
                None
            } else {
                Some((
                    surface.position,
                    surface.shading_normal,
                    bsdf.pdf(bs.wi, TransportMode::Radiance),
                ))
            };

            // ─── 更新光线 ───
            let new_origin = surface.offset_origin(1.0, 1e-3);
            current_ray = Ray::new(new_origin, bs.wi, current_ray.time);
        }

        radiance
    }
}

#[inline]
fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    let denom = a2 + b2;
    if denom <= 0.0 {
        return 1.0;
    }
    a2 / denom
}
