//! 渲染配置：分辨率、采样、通道、胶片。

use serde::{Deserialize, Serialize};

use crate::common::meta::HasMeta;
use crate::common::{Meta, Param};

use super::integrator::IntegratorConfig;
use super::sampler::SamplerConfig;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    /// 每像素总采样数。
    pub samples: u32,
    pub sampler: SamplerConfig,
    pub integrator: IntegratorConfig,
    pub channels: Vec<RenderChannelDesc>,
    pub film: FilmConfig,
    /// 动态拓展参数（执行层特定设置，比如线程数、降噪器配置等）。
    /// **核心配置强类型，执行层扩展走 meta**。
    pub meta: Meta,
}

impl HasMeta for RenderConfig {
    fn meta(&self) -> &Meta { &self.meta }
    fn meta_mut(&mut self) -> &mut Meta { &mut self.meta }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct FilmConfig {
    pub filter: FilterType,
    /// 滤波宽度（像素）。
    pub filter_width: f32,
    /// 曝光（可能与帧/时间相关，所以用 `Param`）。
    pub exposure: Param,
    /// 颜色空间名（`"srgb"` / `"rgb"` / `"acescg"`）。开放字符串，
    /// 由执行层解释。
    pub color_space: String,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum FilterType {
    Box,
    Gaussian,
    BlackmanHarris,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct RenderChannelDesc {
    /// 通道名。**开放字符串**（`"beauty"` / `"diffuse"` / `"zdepth"` /
    /// 用户自定义 `"custom_layer_01"`）。
    pub name: String,
    pub data_type: ChannelDataType,
    pub sample_mode: ChannelSampleMode,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChannelDataType {
    /// FP32 线性RGB
    Rgb,
    /// 三维向量（法线、位置等）。
    Vector,
    /// 单通道标量（深度、ID、遮罩）。
    Scalar,
    /// i64 ID。
    Id,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChannelSampleMode {
    /// 每个采样独立写入后平均（间接光等）。
    PerSample,
    /// 每个像素只写一次（ID、深度等）。
    PerPixel,
}
