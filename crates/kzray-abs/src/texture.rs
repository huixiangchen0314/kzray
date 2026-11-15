//! kzray-abs/src/texture.rs
//!
//! 纹理抽象层。
//!
//! # 抽象层的全部内容
//!
//! - [`Channel`] trait：语义通道 + 值类型 + 插值；
//! - 具体通道类型（[`AlphaChannel`] 等）：类型层面的语义标签，ZST；
//! - [`Differential`]：`eval` 的输入参数；
//! - [`Texture`] trait：唯一的纹理接口——`eval(uv, differential) -> C::Value`；
//! - [`TextureId`]：类型化引用。
//!
//! # 不在抽象层的内容
//!
//! 数据源（常量 / 图像 / 程序化）、MIP 构建、滤波算法、存储布局、
//! 图像加载与格式转换——全部是 `kzray-impl` 的实现细节，各自实现
//! [`Texture`] trait。抽象层只回答"纹理是什么"，不回答"纹理怎么做"。
//!
//! # 设计原则
//!
//! 纹理是"定义在**领域通道**上的空间采样函数"。通道不是值类型，
//! 是语义类别——每个通道对应系统里一个具体的、有领域的量。即使两个
//! 通道的值类型相同（`RoughnessChannel` 和 `MetallicChannel` 都是
//! `f32`），它们是不同的纹理类型，编译器拒绝互换。
//!
//! 不存在"通用标量通道"或"通用光谱通道"：`ScalarChannel` 会让
//! roughness 纹理误传给 metallic 槽位。每个领域一个通道，样板成本由
//! [`channel!`] / [`spectral_channel!`] 宏消除。

use std::marker::PhantomData;

use glam::{Vec2, Vec3};

use crate::SpectralAlgebra;

// ═══════════════════════════════════════════════════════════════
// Channel
// ═══════════════════════════════════════════════════════════════

/// 纹理通道。每个实现是一个**具体的领域类别**，决定采样值类型。
///
/// # 为什么有 `lerp`
///
/// 纹理采样（双线性、MIP 层间插值）的定义要求对通道值做插值。
/// "如何插值"由通道的语义决定——`f32` 线性、`Vec3` 分量、光谱空间
/// 插值。`lerp` 是 `Texture::eval` 得以成立的**定义性要求**，不是
/// 便利操作。
pub trait Channel {
    /// 采样值类型。
    type Value: Copy;

    /// 在通道值上插值。`t ∈ [0, 1]`，`t = 0` 返回 `a`，`t = 1` 返回 `b`。
    fn lerp(a: Self::Value, b: Self::Value, t: f32) -> Self::Value;
}

// ═══════════════════════════════════════════════════════════════
// 宏：定义通道
// ═══════════════════════════════════════════════════════════════

/// 定义 ZST 标量 / 向量通道。
///
/// ```ignore
/// channel!(
///     /// Alpha / cutout 通道。
///     AlphaChannel, f32
/// );
/// ```
#[macro_export]
macro_rules! channel {
    ($(#[$meta:meta])* $name:ident, f32) => {
        $(#[$meta])*
        pub struct $name;

        impl $crate::texture::Channel for $name {
            type Value = f32;
            #[inline]
            fn lerp(a: f32, b: f32, t: f32) -> f32 {
                a + (b - a) * t
            }
        }
    };
    ($(#[$meta:meta])* $name:ident, Vec3) => {
        $(#[$meta])*
        pub struct $name;

        impl $crate::texture::Channel for $name {
            type Value = Vec3;
            #[inline]
            fn lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
                a.lerp(b, t)
            }
        }
    };
}

/// 定义泛型光谱领域通道。以 newtype 形式重包装 [`SpectralChannel<S>`]，
/// 加领域标签。零成本——`SpectralChannel<S>` 是 ZST，包装后仍是 ZST。
#[macro_export]
macro_rules! spectral_channel {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        pub struct $name<S: $crate::SpectralAlgebra>($crate::texture::SpectralChannel<S>);

        impl<S: $crate::SpectralAlgebra> $crate::texture::Channel for $name<S> {
            type Value = S;
            #[inline]
            fn lerp(a: S, b: S, t: f32) -> S {
                S::lerp(a, b, t)
            }
        }
    };
}

// ═══════════════════════════════════════════════════════════════
// 光谱基础通道
// ═══════════════════════════════════════════════════════════════

/// 光谱代数基础通道。泛型 `S` 来自光谱代数系统——纹理系统只是引用它。
///
/// 本身通常不直接作为纹理通道使用（缺领域语义）；具体光谱领域通道
/// （如 [`AlbedoChannel`]）通过 newtype 重包装它。若某个 API 位置
/// 确实需要"任意光谱纹理"，可直接用它。
pub struct SpectralChannel<S: SpectralAlgebra> {
    _marker: PhantomData<S>,
}

impl<S: SpectralAlgebra> Channel for SpectralChannel<S> {
    type Value = S;
    #[inline]
    fn lerp(a: S, b: S, t: f32) -> S {
        S::lerp(a, b, t)
    }
}

// ═══════════════════════════════════════════════════════════════
// 具体通道：每个领域一个
// ═══════════════════════════════════════════════════════════════

// ── 标量领域 ──────────────────────────────────────────────

channel!(
    /// Alpha / cutout 通道。值 `f32 ∈ [0, 1]`，语义是"表面存在性"。
    ///
    /// 求交层用它做 alpha cutout：采样值低于阈值视为透明，命中丢弃。
    AlphaChannel,
    f32
);

channel!(
    /// 微表面粗糙度通道。值 `f32 ∈ [0, 1]`，语义是 GGX 的 `alpha`
    /// 参数或等价的感知粗糙度。
    RoughnessChannel,
    f32
);

channel!(
    /// 金属度通道。值 `f32 ∈ [0, 1]`，语义是"导体 vs 电介质"的线性
    /// 混合系数。
    MetallicChannel,
    f32
);

channel!(
    /// 折射率通道。值 `f32 > 0`，语义是电介质的 IOR。
    IorChannel,
    f32
);

channel!(
    /// 位移高度通道。值 `f32`，语义是沿法线的位移量。
    HeightChannel,
    f32
);

// ── 向量领域 ──────────────────────────────────────────────

channel!(
    /// 切线空间法线通道。值 `Vec3`，语义是编码后的切线空间法线。
    NormalChannel,
    Vec3
);

// ── 光谱领域 ──────────────────────────────────────────────

spectral_channel!(
    /// 漫反射反照率光谱通道。值 `S`，语义是 Lambertian 的漫反射反照率。
    AlbedoChannel
);

spectral_channel!(
    /// 发光光谱通道。值 `S`，语义是面光源/自发光材质的辐射亮度。
    EmissiveChannel
);

spectral_channel!(
    /// 透射光谱通道。值 `S`，语义是电介质折射后的透射衰减。
    TransmissionChannel
);

spectral_channel!(
    /// 导体反射光谱通道（金属的 F0）。值 `S`，语义是金属在垂直入射
    /// 时的反射率。
    ConductorF0Channel
);

// ═══════════════════════════════════════════════════════════════
// Differential
// ═══════════════════════════════════════════════════════════════

/// 微分偏移。着色阶段由 `SurfaceHit` 提供，用于 MIP 选择；
/// 求交阶段无屏幕空间信息，传 `None`。
///
/// 抽象层只定义结构，不定义"如何用微分选 MIP"——那是实现的策略。
#[derive(Clone, Copy, Debug)]
pub struct Differential {
    pub dudx: f32,
    pub dvdx: f32,
    pub dudy: f32,
    pub dvdy: f32,
}

// ═══════════════════════════════════════════════════════════════
// Texture —— 抽象层的唯一接口
// ═══════════════════════════════════════════════════════════════

/// 定义在通道 `C` 上的空间采样函数。
///
/// 抽象契约只有一件事：**给定 UV 和可选微分，返回通道值**。
///
/// - `differential = None`：求交路径，无屏幕空间微分；
/// - `differential = Some`：着色路径，可用于 MIP 选择。
///
/// 数据源（常量 / 图像 / 程序化）、MIP 构建、滤波策略、存储布局——
/// 全部是实现的细节。同一个纹理对象服务两条采样路径，无需两套接口。
pub trait Texture<C: Channel> {
    /// 在 UV 处求值。
    ///
    /// # 参数
    ///
    /// - `uv`：纹理坐标。UV 的定义域、wrap 模式由实现决定；
    /// - `differential`：屏幕空间微分。`None` 表示求交路径。
    fn eval(&self, uv: Vec2, differential: Option<&Differential>) -> C::Value;
}


// ═══════════════════════════════════════════════════════════════
// 常用纹理别名
// ═══════════════════════════════════════════════════════════════
//
// 为每个通道定义 trait object 别名，函数签名写 `&AlphaTexture`
// 而非 `&dyn Texture<AlphaChannel>`。别名带 `Send + Sync`——纹理要
// 跨线程采样（求交、着色、资源加载），无条件要求。
//
// `dyn Trait + Send + Sync` 默认 `'static`，别名无需显式生命周期。
//
// 用法：
//
// ```ignore
// fn alpha_cutout(tex: &AlphaTexture, uv: Vec2) -> bool {
//     tex.eval(uv, None) >= ALPHA_THRESHOLD
// }
//
// let lib: TextureLibrary;         // impl 层
// let boxed: Box<AlphaTexture>;    // 存储
// ```

// ── 标量通道 ──────────────────────────────────────────────

/// Alpha / cutout 纹理。`eval` 返回 `f32`。
pub type AlphaTexture = dyn Texture<AlphaChannel> + Send + Sync;

/// 微表面粗糙度纹理。`eval` 返回 `f32`。
pub type RoughnessTexture = dyn Texture<RoughnessChannel> + Send + Sync;

/// 金属度纹理。`eval` 返回 `f32`。
pub type MetallicTexture = dyn Texture<MetallicChannel> + Send + Sync;

/// 折射率纹理。`eval` 返回 `f32`。
pub type IorTexture = dyn Texture<IorChannel> + Send + Sync;

/// 位移高度纹理。`eval` 返回 `f32`。
pub type HeightTexture = dyn Texture<HeightChannel> + Send + Sync;

// ── 向量通道 ──────────────────────────────────────────────

/// 切线空间法线纹理。`eval` 返回 `Vec3`。
pub type NormalTexture = dyn Texture<NormalChannel> + Send + Sync;

// ── 光谱通道（泛型 `S` 来自光谱系统） ──────────────────────

/// 漫反射反照率纹理。`eval` 返回 `S`。
pub type AlbedoTexture<S> = dyn Texture<AlbedoChannel<S>> + Send + Sync;

/// 发光光谱纹理。`eval` 返回 `S`。
pub type EmissiveTexture<S> = dyn Texture<EmissiveChannel<S>> + Send + Sync;

/// 透射光谱纹理。`eval` 返回 `S`。
pub type TransmissionTexture<S> = dyn Texture<TransmissionChannel<S>> + Send + Sync;

/// 导体反射光谱纹理。`eval` 返回 `S`。
pub type ConductorF0Texture<S> = dyn Texture<ConductorF0Channel<S>> + Send + Sync;

// ── 泛型光谱兜底 ──────────────────────────────────────────
//
// 某个 API 位置确实需要"任意光谱纹理"时用；具体领域通道优先。

/// 光谱代数纹理（无领域标签）。`eval` 返回 `S`。
pub type SpectralTexture<S> = dyn Texture<SpectralChannel<S>> + Send + Sync;
