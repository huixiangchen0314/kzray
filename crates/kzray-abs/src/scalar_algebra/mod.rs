//! 标量代数系统。
//!
//! 核心抽象见 [`ScalarAlgebra`]。

mod algebra;
mod primitive;
mod rgb;
mod spectral;

pub use algebra::ScalarAlgebra;
pub use rgb::Rgb;
pub use spectral::Spectral;
