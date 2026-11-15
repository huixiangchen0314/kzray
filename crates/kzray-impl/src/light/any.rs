//! kzray-impl/src/light/any.rs
//! 光源枚举。
//!
//! `AnyLight<S>` / `AnyInfiniteLight<S>` 封装所有具体光源类型。`Scene` 的
//! `L: Light<S>` 泛型参数实例化为 `AnyLight<S>` 时，`sample_li` 走 enum
//! `match`（静态分派），零虚分派。
//!
//! # 为什么光源用 enum，材质用 enum（契约层）
//!
//! 光源种类是场景内容（开放），但实现层当前只有 `PointLight` /
//! `ShapeAreaLight` 两种。用 enum 表达"实现层支持的光源集合"，加新光源
//! 时扩展 enum。与 `Material` 用 enum 是同一哲学——**实现层的种类封闭，
//! 用 enum**。
//!
//! 如果未来需要运行时插件光源（用户自定义），enum 不够用，退回
//! `Arc<dyn Light<S>>`。那是极端场景，当前不支持。

use glam::{Vec2, Vec3};

use kzray_abs::light::{
    InfiniteLight, LeSample, LiSample, Light, LightSampleContext, LightType,
};
use kzray_abs::ray::Ray;
use kzray_abs::spectrum::SpectralAlgebra;

use super::{ConstInfiniteLight, PointLight, ShapeAreaLight};

// ═══════════════════════════════════════════════════════════════
// AnyLight
// ═══════════════════════════════════════════════════════════════

/// 光源枚举。当前支持点光源 + 解析形状面积光，未来扩展 Spot / Sun / MeshArea。
///
/// # `S` 泛型
///
/// `AnyLight<S>` 的 `S` 通过 `impl Light<S>` 传播到 `sample_li` /
/// `pdf_li` / `phi`。`sample_li` / `pdf_li` 的 `ctx` 是
/// `LightSampleContext<'_, S>`——`S` 与 `Self` 的 `S` 一致，编译器从
/// `self` 推断。
pub enum AnyLight<S: SpectralAlgebra> {
    Point(PointLight<S>),
    ShapeArea(ShapeAreaLight<S>),
    // 未来：Spot(SpotLight<S>),
    // 未来：MeshArea(MeshAreaLight<S>),
    // 未来：Sun(SunLight<S>),
}

impl<S: SpectralAlgebra> Light<S> for AnyLight<S> {
    #[inline]
    fn light_type(&self) -> LightType {
        match self {
            Self::Point(l) => l.light_type(),
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
            Self::Point(l) => l.sample_li(ctx, sample, wavelengths),
            Self::ShapeArea(l) => l.sample_li(ctx, sample, wavelengths),
        }
    }

    #[inline]
    fn pdf_li(&self, ctx: &LightSampleContext<'_, S>, wi: Vec3) -> f32 {
        match self {
            Self::Point(l) => l.pdf_li(ctx, wi),
            Self::ShapeArea(l) => l.pdf_li(ctx, wi),
        }
    }

    #[inline]
    fn phi(&self, wavelengths: &S::WavelengthContext) -> S {
        match self {
            Self::Point(l) => l.phi(wavelengths),
            Self::ShapeArea(l) => l.phi(wavelengths),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// AnyInfiniteLight
// ═══════════════════════════════════════════════════════════════

/// 无限远光源枚举。当前支持常数环境光，未来扩展 HDR 环境图。
///
/// # `Light` 实现 vs `InfiniteLight` 实现
///
/// `InfiniteLight: Light` 是 supertrait 关系，`AnyInfiniteLight` 必须
/// 同时实现两者。但作为 `IL: InfiniteLight<S>` 泛型参数使用的实例**不会**
/// 走 `Light::sample_li` 的 NEE 路径——NEE 只对 `L: Light<S>` 里的光源
/// 采样，环境光通过逃逸光线累加（`emitted_radiance`）。
///
/// 因此 `Light::sample_li` 返回 `None`、`pdf_li` 返回 `0.0`——环境光
/// 不参与 NEE，这是当前 PT 设计（不采样环境光）的预期行为。
pub enum AnyInfiniteLight<S: SpectralAlgebra> {
    Constant(ConstInfiniteLight<S>),
    // 未来：Hdr(HdrInfiniteLight<S>),
}

impl<S: SpectralAlgebra> Light<S> for AnyInfiniteLight<S> {
    #[inline]
    fn light_type(&self) -> LightType {
        LightType::Infinite
    }

    /// 环境光不参与 NEE。返回 `None`，`PathIntegrator` 的 NEE 段
    /// 走不到这个分支（因为 `IL` 与 `L` 是独立泛型参数）。
    #[inline]
    fn sample_li(
        &self,
        _ctx: &LightSampleContext<'_, S>,
        _sample: Vec2,
        _wavelengths: &S::WavelengthContext,
    ) -> Option<LiSample<S>> {
        None
    }

    /// 环境光无连续方向采样 PDF。
    #[inline]
    fn pdf_li(&self, _ctx: &LightSampleContext<'_, S>, _wi: Vec3) -> f32 {
        0.0
    }

    /// 环境光的总功率不参与光源采样器的构造（`LightSampler` 只登记
    /// `L` 列表里的光源）。返回零是占位，不会被读取。
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
