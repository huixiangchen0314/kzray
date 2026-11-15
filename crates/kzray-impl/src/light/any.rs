//! kzray-impl/src/light/any.rs
//!
//! 三个独立的光源容器 enum，对应 [`kzray_abs::scene::Scene`] 的三个光源
//! 泛型参数：
//!
//! - [`AnyDeltaLight`]：delta 光源（点光源、方向光、spot）；
//! - [`AnyAreaLight`]：面积光（解析形状、未来 mesh）；
//! - [`AnyInfiniteLight`]：无限远光源（环境光、天空光）。
//!
//! # 没有统一的 `AnyLight`
//!
//! 三者物理性质不同（NEE pdf 测度、MIS 参与、是否有发射表面），
//! 强行统一成一个顶层 enum 会让类型系统丢失区分，诱导调用方写
//! `match AnyLight` 时忽略物理差异。**不存在 `AnyLight`**——`Scene`
//! 的 `DL` / `IL` / `AL` 三个泛型独立，各自对应用户选择的具体类型。
//!
//! # 用户扩展
//!
//! 用户想加自定义光源时，实现 `Light<S>` / `AreaLight<S>` /
//! `InfiniteLight<S>` 后作为 `DL` / `AL` / `IL` 传入 `SimpleScene`。
//! 三个 enum 只是"标准实现集合"，不是强制抽象。

use glam::{Vec2, Vec3};

use kzray_abs::light::{
    AreaLight, InfiniteLight, LeSample, LiSample, Light, LightSampleContext, LightType,
};
use kzray_abs::ray::Ray;
use kzray_abs::spectrum::SpectralAlgebra;

use super::{ConstInfiniteLight, PointLight, ShapeAreaLight};

// ═══════════════════════════════════════════════════════════════
// AnyDeltaLight
// ═══════════════════════════════════════════════════════════════

/// Delta 光源容器。位置或方向成 delta 分布——NEE 的 `pdf` 是离散
/// 概率，不参与 MIS（`mis_weight = 1.0`）。
///
/// 当前只有点光源，未来扩展方向光、spot。
pub enum AnyDeltaLight<S: SpectralAlgebra> {
    Point(PointLight<S>),
    // 未来：Directional(DirectionalLight<S>),
    // 未来：Spot(SpotLight<S>),
}

impl<S: SpectralAlgebra> Light<S> for AnyDeltaLight<S> {
    #[inline]
    fn light_type(&self) -> LightType {
        match self {
            Self::Point(l) => l.light_type(),
        }
    }

    #[inline]
    fn sample_li(
        &self,
        ctx: &LightSampleContext<'_, S>,
        sample: Vec2,
        wavelengths: &S::WavelengthContext,
    ) -> Option<LiSample<S>> {
        match self {
            Self::Point(l) => l.sample_li(ctx, sample, wavelengths),
        }
    }

    #[inline]
    fn pdf_li(&self, ctx: &LightSampleContext<'_, S>, wi: Vec3) -> f32 {
        match self {
            Self::Point(l) => l.pdf_li(ctx, wi),
        }
    }

    #[inline]
    fn phi(&self, wavelengths: &S::WavelengthContext) -> S {
        match self {
            Self::Point(l) => l.phi(wavelengths),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// AnyAreaLight
// ═══════════════════════════════════════════════════════════════

/// 面积光容器。NEE 的 `pdf` 是连续测度，参与 MIS。
///
/// 面积光同时提供 NEE 采样（`Light`）和 emission 求值（`AreaLight`）——
/// 两个能力都属于面积光，`AnyAreaLight` 双实现。与
/// [`AnyDeltaLight`] 独立，不共享父 enum。
pub enum AnyAreaLight<S: SpectralAlgebra> {
    ShapeArea(ShapeAreaLight<S>),
    // 未来：MeshArea(MeshAreaLight<S>),
}

impl<S: SpectralAlgebra> Light<S> for AnyAreaLight<S> {
    #[inline]
    fn light_type(&self) -> LightType {
        match self {
            Self::ShapeArea(l) => l.light_type(),
        }
    }

    #[inline]
    fn sample_li(
        &self,
        ctx: &LightSampleContext<'_, S>,
        sample: Vec2,
        wavelengths: &S::WavelengthContext,
    ) -> Option<LiSample<S>> {
        match self {
            Self::ShapeArea(l) => l.sample_li(ctx, sample, wavelengths),
        }
    }

    #[inline]
    fn pdf_li(&self, ctx: &LightSampleContext<'_, S>, wi: Vec3) -> f32 {
        match self {
            Self::ShapeArea(l) => l.pdf_li(ctx, wi),
        }
    }

    #[inline]
    fn phi(&self, wavelengths: &S::WavelengthContext) -> S {
        match self {
            Self::ShapeArea(l) => l.phi(wavelengths),
        }
    }
}

impl<S: SpectralAlgebra> AreaLight<S> for AnyAreaLight<S> {
    #[inline]
    fn emitted_radiance(
        &self,
        point: Vec3,
        normal: Vec3,
        uv: Vec2,
        wo: Vec3,
        wavelengths: &S::WavelengthContext,
    ) -> S {
        match self {
            Self::ShapeArea(l) => l.emitted_radiance(point, normal, uv, wo, wavelengths),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// AnyInfiniteLight
// ═══════════════════════════════════════════════════════════════

/// 无限远光源容器。环境光 / 天空光。
///
/// `Scene::infinite_lights() -> &[IL]` 返回一个 `Vec<IL>`——无限远光源
/// 无 id，逃逸光线累加时遍历求和。通常场景只有一个环境光，接口允许
/// 多个（分层天空等）。
pub enum AnyInfiniteLight<S: SpectralAlgebra> {
    Constant(ConstInfiniteLight<S>),
    // 未来：Hdr(HdrInfiniteLight<S>),
}

impl<S: SpectralAlgebra> Light<S> for AnyInfiniteLight<S> {
    #[inline]
    fn light_type(&self) -> LightType {
        LightType::Infinite
    }

    /// 环境光不参与 NEE。`PathIntegrator` 的 NEE 采样器只登记 delta /
    /// 面积光 id——无限远光源无 id，逃逸时累加。
    #[inline]
    fn sample_li(
        &self,
        _ctx: &LightSampleContext<'_, S>,
        _sample: Vec2,
        _wavelengths: &S::WavelengthContext,
    ) -> Option<LiSample<S>> {
        None
    }

    #[inline]
    fn pdf_li(&self, _ctx: &LightSampleContext<'_, S>, _wi: Vec3) -> f32 {
        0.0
    }

    #[inline]
    fn phi(&self, _wavelengths: &S::WavelengthContext) -> S {
        S::zero()
    }
}

impl<S: SpectralAlgebra> InfiniteLight<S> for AnyInfiniteLight<S> {
    #[inline]
    fn emitted_radiance(&self, ray: &Ray, wavelengths: &S::WavelengthContext) -> S {
        match self {
            Self::Constant(l) => l.emitted_radiance(ray, wavelengths),
        }
    }

    #[inline]
    fn sample_le(
        &self,
        sample: Vec2,
        wavelengths: &S::WavelengthContext,
    ) -> Option<LeSample<S>> {
        match self {
            Self::Constant(l) => l.sample_le(sample, wavelengths),
        }
    }

    #[inline]
    fn pdf_le(&self, direction: Vec3, wavelengths: &S::WavelengthContext) -> f32 {
        match self {
            Self::Constant(l) => l.pdf_le(direction, wavelengths),
        }
    }
}
