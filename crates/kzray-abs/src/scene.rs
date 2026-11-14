//! 场景只读视图。
//!
//! # 全泛型化（`Material` 除外）
//!
//! `Scene<S, LS, L, IL>` 携带 4 个泛型参数：
//!
//! - `S`：光谱类型；
//! - `LS: LightSampler`：光源采样器；
//! - `L: Light<S>`：光源；
//! - `IL: InfiniteLight<S>`：无限远光源。
//!
//! `material` 返回 `&Material<S>`——`Material` 是契约层的具体 enum 类型，
//! 不是泛型参数。材质种类封闭（传输方程结构），由 enum 表达；光源种类
//! 开放（场景内容），由泛型 `L` 表达。
//!
//! 所有查询方法返回具体类型引用，零虚分派。`Scene` 本身 object-safe
//! （方法都不是泛型），但通过泛型 `SC: Scene<...>` 传播。

use std::ops::Range;

use kzray_data::common::LightId;
use kzray_data::geometry::Aabb;

use crate::light::{InfiniteLight, Light, LightSampler};
use crate::material::Material;
use crate::ray::{Ray, SurfaceHit};
use crate::spectrum::SpectralAlgebra;

pub trait Scene<S, LS, L, IL>: Send + Sync
where
    S: SpectralAlgebra,
    LS: LightSampler,
    L: Light<S>,
    IL: InfiniteLight<S>,
{
    fn bounds(&self) -> Aabb;

    fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<SurfaceHit>;

    fn intersect_p(&self, ray: &Ray, t_range: Range<f32>) -> bool;

    /// 查询运行时材质。返回 `&Material<S>`（具体 enum）。
    fn material(&self, hit: &SurfaceHit) -> Option<&Material<S>>;

    fn light(&self, id: LightId) -> Option<&L>;

    fn light_sampler(&self) -> Option<&LS>;

    fn infinite_light(&self) -> Option<&IL>;
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::light::{
        InfiniteLight, LeSample, LiSample, Light, LightSampleContext, LightSelection,
        LightType,
    };
    use crate::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
    use glam::{UVec2, Vec2, Vec3};

    struct TestSampler;
    impl crate::sampler::Sampler for TestSampler {
        fn start_pixel_sample(&mut self, _: UVec2, _: u32) {}
        fn next_f32(&mut self, _: u32) -> f32 {
            0.5
        }
    }

    struct EmptyLightSampler;
    impl LightSampler for EmptyLightSampler {
        fn sample<SM: crate::sampler::Sampler>(
            &self,
            _ctx: &LightSampleContext,
            _sampler: &mut SM,
            _dim: u32,
        ) -> Option<LightSelection> {
            None
        }
        fn pdf(&self, _ctx: &LightSampleContext, _id: LightId) -> f32 {
            0.0
        }
        fn max_sample_dimensions(&self) -> u32 {
            1
        }
    }

    struct EmptyLight;
    impl Light<SpectrumRgb> for EmptyLight {
        fn light_type(&self) -> LightType {
            LightType::DeltaPosition
        }
        fn sample_li(
            &self,
            _ctx: &LightSampleContext,
            _s: Vec2,
            _wl: &NoWavelengths,
        ) -> Option<LiSample<SpectrumRgb>> {
            None
        }
        fn pdf_li(&self, _ctx: &LightSampleContext, _wi: Vec3) -> f32 {
            0.0
        }
        fn phi(&self, _wl: &NoWavelengths) -> SpectrumRgb {
            SpectrumRgb::ZERO
        }
    }

    struct EmptyInfiniteLight;
    impl InfiniteLight<SpectrumRgb> for EmptyInfiniteLight {
        fn emitted_radiance(&self, _: &Ray, _: &NoWavelengths) -> SpectrumRgb {
            SpectrumRgb::ZERO
        }
        fn sample_le(&self, _: Vec2, _: &NoWavelengths) -> Option<LeSample<SpectrumRgb>> {
            None
        }
        fn pdf_le(&self, _: Vec3, _: &NoWavelengths) -> f32 {
            0.0
        }
    }
    impl Light<SpectrumRgb> for EmptyInfiniteLight {
        fn light_type(&self) -> LightType {
            LightType::Infinite
        }
        fn sample_li(
            &self,
            _: &LightSampleContext,
            _: Vec2,
            _: &NoWavelengths,
        ) -> Option<LiSample<SpectrumRgb>> {
            None
        }
        fn pdf_li(&self, _: &LightSampleContext, _: Vec3) -> f32 {
            0.0
        }
        fn phi(&self, _: &NoWavelengths) -> SpectrumRgb {
            SpectrumRgb::ZERO
        }
    }

    struct EmptyScene;
    impl Scene<SpectrumRgb, EmptyLightSampler, EmptyLight, EmptyInfiniteLight> for EmptyScene {
        fn bounds(&self) -> Aabb {
            Aabb::empty()
        }
        fn intersect(&self, _: &Ray, _: Range<f32>) -> Option<SurfaceHit> {
            None
        }
        fn intersect_p(&self, _: &Ray, _: Range<f32>) -> bool {
            false
        }
        fn material(&self, _: &SurfaceHit) -> Option<&Material<SpectrumRgb>> {
            None
        }
        fn light(&self, _: LightId) -> Option<&EmptyLight> {
            None
        }
        fn light_sampler(&self) -> Option<&EmptyLightSampler> {
            None
        }
        fn infinite_light(&self) -> Option<&EmptyInfiniteLight> {
            None
        }
    }

    #[test]
    fn empty_scene_bounds_is_empty() {
        let scene = EmptyScene;
        assert!(scene.bounds().is_empty());
    }

    #[test]
    fn empty_scene_never_hits() {
        let scene = EmptyScene;
        let ray = Ray::new(Vec3::ZERO, Vec3::X, 0.0);
        assert!(scene.intersect(&ray, 0.0..f32::INFINITY).is_none());
        assert!(!scene.intersect_p(&ray, 0.0..f32::INFINITY));
    }

    #[test]
    fn empty_scene_has_no_lights() {
        let scene = EmptyScene;
        assert!(scene.light(LightId::from(0)).is_none());
        assert!(scene.light_sampler().is_none());
        assert!(scene.infinite_light().is_none());
    }

    #[test]
    fn scene_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EmptyScene>();
    }

    #[test]
    fn _use_test_sampler() {
        let _ = TestSampler;
    }
}
