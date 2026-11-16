//! HDR 环境光。等距柱状投影采样 + 强度缩放。
//!
//! # 当前阶段
//!
//! 只实现 `emitted_radiance`——逃逸光线、金属反射、玻璃折射看到 HDR
//! 环境。`sample_le` / `pdf_le`（环境光 NEE）留 TODO，后续阶段做方向
//! 重要性采样。
//!
//! # 曝光控制
//!
//! `scale` 乘到 `emitted_radiance` 上。HDR 文件值域差异大（室内
//! `[0, 1]`、户外日落 `[0, 10000]`），用户按场景调整 `scale` 到合适
//! 亮度。
//!
//! # 与 `ConstInfiniteLight` 的关系
//!
//! 两者都实现 `InfiniteLight<S>`，可互换。调试场景用 `Const`（可控、
//! 无需外部文件），真实场景用 `Hdr`。

use glam::{Vec2, Vec3};

use kzray_abs::light::{
    InfiniteLight, LeSample, LiSample, Light, LightSampleContext, LightType,
};
use kzray_abs::ray::Ray;
use kzray_abs::spectrum::SpectralAlgebra;

use super::hdr_image::HdrImage;
use super::hdr_mapping::direction_to_uv;

// ═══════════════════════════════════════════════════════════════
// HdrInfiniteLight
// ═══════════════════════════════════════════════════════════════

/// HDR 环境光。
pub struct HdrInfiniteLight<S: SpectralAlgebra> {
    image: HdrImage<S>,
    scale: f32,
}

impl<S: SpectralAlgebra> HdrInfiniteLight<S> {
    /// 从 HDR 图像和强度缩放构造。
    ///
    /// # 参数
    ///
    /// - `image`：用户加载并转成 `S` 的 HDR 像素（`HdrImage` 不负责
    ///   文件加载，见其模块文档）；
    /// - `scale`：强度缩放，乘到 `emitted_radiance` 上。建议 `1.0` 起步
    ///   按视觉调整——HDR 值域差异大，不同文件需要不同 `scale`。
    pub fn new(image: HdrImage<S>, scale: f32) -> Self {
        Self { image, scale }
    }

    pub fn image(&self) -> &HdrImage<S> {
        &self.image
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }
}

impl<S: SpectralAlgebra> Light<S> for HdrInfiniteLight<S> {
    #[inline]
    fn light_type(&self) -> LightType {
        LightType::Infinite
    }

    /// 环境光不参与 NEE。
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

    /// 总功率。当前返回零——`LightSampler` 不登记无限远光源，
    /// `phi` 不被读取。
    #[inline]
    fn phi(&self, _wavelengths: &S::WavelengthContext) -> S {
        S::zero()
    }
}

impl<S: SpectralAlgebra> InfiniteLight<S> for HdrInfiniteLight<S> {
    /// 沿逃逸光线的辐射亮度。方向映射到等距柱状 UV，采样 HDR 图像，
    /// 乘 `scale`。
    #[inline]
    fn emitted_radiance(&self, ray: &Ray, _wavelengths: &S::WavelengthContext) -> S {
        let dir = ray.dir.normalize();
        let uv = direction_to_uv(dir);
        self.image.sample(uv).scale(S::Scalar::from(self.scale))
    }

    /// TODO: 环境光 NEE 的方向重要性采样。当前返回 `None`——`PathIntegrator`
    /// 的 NEE 段不采样无限远光源（逃逸累加已覆盖直接可见的环境光）。
    #[inline]
    fn sample_le(
        &self,
        _sample: Vec2,
        _wavelengths: &S::WavelengthContext,
    ) -> Option<LeSample<S>> {
        None
    }

    #[inline]
    fn pdf_le(&self, _direction: Vec3, _wavelengths: &S::WavelengthContext) -> f32 {
        0.0
    }
}
