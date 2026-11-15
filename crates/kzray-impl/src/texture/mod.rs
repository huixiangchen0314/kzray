//! kzray-impl/src/texture/mod.rs
//!
//! 具体纹理实现。
//!
//! 抽象层见 `kzray_abs::texture`——`Texture<C>` trait、`Channel`、
//! `Differential`、`TextureId` 等。本模块实现具体的纹理数据源：
//!
//! - [`ConstantTexture`]：常量，无空间变化；
//! - [`ImageTexture`]：图像 + 双线性采样。
//!
//! # 当前实现范围
//!
//! - **只做双线性采样，无 MIP**——MIP 依赖相机微分（`Differential`），
//!   阶段 5 实现。
//! - **不做文件 I/O**——`ImageTexture` 从已解码的 `Vec<C::Value>` 构造。
//!   图像加载与逐通道解码（文件 RGBA → `C::Value`）在后续迭代补，
//!   届时按通道类型提供 `from_file` 或 `load_*` 辅助函数。
//! - **不做程序化纹理**——checker / noise / fbm 等在需要时再加。
//!
//! # 新增纹理实现
//!
//! 实现 `Texture<C>` 后直接 `Box::new(...)` 注册到 `Library::add_<通道>`
//! 即可。具体类型不必在本模块——`Texture<C>` 是 trait，任何 crate
//! 都可以提供实现（用户自定义程序化纹理等）。

pub mod constant;
pub mod image;

pub use constant::ConstantTexture;
pub use image::ImageTexture;
