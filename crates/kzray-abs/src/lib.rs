//! kzray-abs：渲染积分抽象层。
//!
//! 定义 MIS 无偏积分的数学核心、四个执行角色（积分器、引导器、积累器、拦截器）、
//! 路径状态、以及执行上下文 `Runtime` 的占位。
//!
//! **边界**：本 crate 只定义抽象契约，不含具体实现。PT / BDPT / 光子映射、
//! 具体积累器、具体采样器、具体引导器由上层 crate 提供，在本 crate 的
//! trait 上装配。
//!
//! **范畴**：抽象层的内容是 MIS 积分的数学概念（BSDF、pdf、采样、权重）
//! 和四个执行角色的接口。路径来源、调度策略、工程配置全部在构建层和
//! `Runtime` 里，不在抽象层。

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
