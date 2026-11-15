//! kzray-abs：渲染积分抽象层。

pub mod error;
pub mod ray;
pub mod sampler;
pub mod bsdf;
pub mod material;
pub mod camera;
pub mod light;
pub mod spectrum;
pub mod sensor;
pub mod film;
pub mod scene;
pub mod integrator;
pub mod quantity;
pub mod texture;
pub mod library;

// ---- 能量代数 ----
pub use spectrum::{
    ScalarAlgebra,
    SpectralAlgebra,
    Rgb,
    SampledWavelengths,
};

pub use spectrum::Spectrum;

pub use sensor::{RgbSensor, CieSensor};

// ---- 采样 ----
pub use sampler::Sampler;

// ---- 几何 ----
pub use ray::{Ray, SurfaceHit};

pub use film::{Film, AdaptiveFilm, FilmWithVisibleSurface};
pub use material::{Material, BSDFBuilder, BXDFBuilder};
pub use scene::Scene;
pub use integrator::Integrator;
pub use quantity::{
    Area, Delta, Measure, MisPdf, PathSpace, RandomWalk, Pdf, SolidAngle,
};

// 具体通道类型（每个领域一个）
pub use texture::{
    // 标量领域
    AlphaChannel,
    RoughnessChannel,
    MetallicChannel,
    IorChannel,
    HeightChannel,
    // 向量领域
    NormalChannel,
    // 光谱领域
    AlbedoChannel,
    EmissiveChannel,
    TransmissionChannel,
    ConductorF0Channel,
};

// 纹理接口 + 微分
pub use texture::{Differential, Texture};

// 常用纹理别名（trait object，带 Send + Sync）
pub use texture::{
    // 标量领域
    AlphaTexture,
    RoughnessTexture,
    MetallicTexture,
    IorTexture,
    HeightTexture,
    // 向量领域
    NormalTexture,
    // 光谱领域
    AlbedoTexture,
    EmissiveTexture,
    TransmissionTexture,
    ConductorF0Texture,
    // 泛型光谱兜底
    SpectralTexture,
};

pub use library::Library;
