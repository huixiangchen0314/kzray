//! 无限远光源实现。

use glam::Vec2;

use kzray_abs::light::{
    InfiniteLight, LeSample, LiSample, Light, LightSampleContext, LightType,
};
use kzray_abs::ray::Ray;
use kzray_abs::spectrum::SpectralAlgebra;

/// 常数环境光（均匀颜色，无方向变化）。
pub struct ConstInfiniteLight<S: SpectralAlgebra> {
    radiance: S,
}

impl<S: SpectralAlgebra> ConstInfiniteLight<S> {
    pub fn new(radiance: S) -> Self {
        ConstInfiniteLight { radiance }
    }

    /// 环境光辐射亮度。
    pub fn radiance(&self) -> S {
        self.radiance
    }
}

// InfiniteLight: Light，super trait 要求 Light<S> 也实现
impl<S: SpectralAlgebra> Light<S> for ConstInfiniteLight<S> {
    fn light_type(&self) -> LightType {
        LightType::Infinite
    }

    fn sample_li(
        &self,
        _ctx: &LightSampleContext,
        _sample: Vec2,
        _wavelengths: &S::WavelengthContext,
    ) -> Option<LiSample<S>> {
        // 无限远光源不参与 NEE 采样（环境光通过逃逸光线累加）
        None
    }

    fn pdf_li(&self, _ctx: &LightSampleContext, _wi: glam::Vec3) -> f32 {
        0.0
    }

    fn phi(&self, _wavelengths: &S::WavelengthContext) -> S {
        // 环境光的总功率没有意义（无限面积），返回 zero 表示不参与光源选择
        S::zero()
    }
}

impl<S: SpectralAlgebra> InfiniteLight<S> for ConstInfiniteLight<S> {
    fn emitted_radiance(&self, _ray: &Ray, _wavelengths: &S::WavelengthContext) -> S {
        self.radiance
    }

    fn sample_le(&self, _sample: Vec2, _wavelengths: &S::WavelengthContext) -> Option<LeSample<S>> {
        // 均匀环境光的重要性采样需要球面均匀采样，当前不实现
        None
    }

    fn pdf_le(&self, _direction: glam::Vec3, _wavelengths: &S::WavelengthContext) -> f32 {
        0.0
    }
}
