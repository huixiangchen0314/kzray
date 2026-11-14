//! 光源枚举。
//!
//! `AnyLight<S>` / `AnyInfiniteLight<S>` 封装所有具体光源类型。`Scene` 的
//! `L: Light<S>` 泛型参数实例化为 `AnyLight<S>` 时，`sample_li` 走 enum
//! `match`（静态分派），零虚分派。
//!
//! # 为什么光源用 enum，材质用 enum（契约层）
//!
//! 光源种类是场景内容（开放），但实现层当前只有 PointLight 一种。用
//! enum 表达"实现层支持的光源集合"，加新光源时扩展 enum。与 `Material`
//! 用 enum 是同一哲学——**实现层的种类封闭，用 enum**。
//!
//! 如果未来需要运行时插件光源（用户自定义），enum 不够用，退回
//! `Arc<dyn Light<S>>`。那是极端场景，当前不支持。

use glam::Vec2;

use kzray_abs::light::{
    InfiniteLight, LeSample, LiSample, Light, LightSampleContext, LightType,
};
use kzray_abs::ray::Ray;
use kzray_abs::spectrum::SpectralAlgebra;

use super::PointLight;

// ═══════════════════════════════════════════════════════════════
// AnyLight
// ═══════════════════════════════════════════════════════════════

/// 光源枚举。当前支持点光源，未来扩展 Spot / Area / Sun。
pub enum AnyLight<S: SpectralAlgebra> {
    Point(PointLight<S>),
    // 未来：Spot(SpotLight<S>),
    // 未来：Area(AreaLight<S>),
    // 未来：Sun(SunLight<S>),
}

impl<S: SpectralAlgebra> Light<S> for AnyLight<S> {
    fn light_type(&self) -> LightType {
        match self {
            AnyLight::Point(l) => l.light_type(),
        }
    }

    fn sample_li(
        &self,
        ctx: &LightSampleContext,
        sample: Vec2,
        wavelengths: &S::WavelengthContext,
    ) -> Option<LiSample<S>> {
        match self {
            AnyLight::Point(l) => l.sample_li(ctx, sample, wavelengths),
        }
    }

    fn pdf_li(&self, ctx: &LightSampleContext, wi: glam::Vec3) -> f32 {
        match self {
            AnyLight::Point(l) => l.pdf_li(ctx, wi),
        }
    }

    fn phi(&self, wavelengths: &S::WavelengthContext) -> S {
        match self {
            AnyLight::Point(l) => l.phi(wavelengths),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// AnyInfiniteLight
// ═══════════════════════════════════════════════════════════════

/// 无限远光源枚举。当前支持常数环境光，未来扩展 HDR 环境图。
pub enum AnyInfiniteLight<S: SpectralAlgebra> {
    Constant(super::ConstInfiniteLight<S>),
    // 未来：Hdr(HdrInfiniteLight<S>),
}

impl<S: SpectralAlgebra> Light<S> for AnyInfiniteLight<S> {
    fn light_type(&self) -> LightType {
        LightType::Infinite
    }

    fn sample_li(
        &self,
        _ctx: &LightSampleContext,
        _sample: Vec2,
        _wl: &S::WavelengthContext,
    ) -> Option<LiSample<S>> {
        None // 无限远光源不参与 NEE 采样（环境光通过逃逸光线累加）
    }

    fn pdf_li(&self, _ctx: &LightSampleContext, _wi: glam::Vec3) -> f32 {
        0.0
    }

    fn phi(&self, _wl: &S::WavelengthContext) -> S {
        S::zero()
    }
}

impl<S: SpectralAlgebra> InfiniteLight<S> for AnyInfiniteLight<S> {
    fn emitted_radiance(&self, ray: &Ray, wl: &S::WavelengthContext) -> S {
        match self {
            AnyInfiniteLight::Constant(l) => l.emitted_radiance(ray, wl),
        }
    }

    fn sample_le(&self, sample: Vec2, wl: &S::WavelengthContext) -> Option<LeSample<S>> {
        match self {
            AnyInfiniteLight::Constant(l) => l.sample_le(sample, wl),
        }
    }

    fn pdf_le(&self, direction: glam::Vec3, wl: &S::WavelengthContext) -> f32 {
        match self {
            AnyInfiniteLight::Constant(l) => l.pdf_le(direction, wl),
        }
    }
}
