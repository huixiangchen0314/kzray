//! 渲染配置。

pub mod integrator;
pub mod render;
pub mod sampler;

pub use integrator::{IntegratorConfig, PathTracerConfig, PhotonMapperConfig};
pub use render::{
    ChannelDataType, ChannelSampleMode, FilmConfig, FilterType, RenderChannelDesc, RenderConfig,
};
pub use sampler::SamplerConfig;
