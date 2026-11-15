//! kzray-impl/src/texture/constant.rs
//!
//! 常量纹理。

use glam::Vec2;

use kzray_abs::texture::{Channel, Differential, Texture};

/// 常量纹理。所有 UV 返回同一个值。
///
/// 用途：
///
/// - 材质参数无空间变化（纯色 albedo、固定 roughness）；
/// - 调试——用常量替换纹理隔离问题；
/// - 默认值——未指定纹理时的 fallback。
///
/// # 例
///
/// ```ignore
/// // 不透明
/// let alpha_id = lib.add_alpha(
///     TextureId::from_index(0),
///     Box::new(ConstantTexture::<AlphaChannel>::new(1.0)),
/// );
///
/// // 灰色 albedo
/// let albedo_id = lib.add_albedo(
///     TextureId::from_index(1),
///     Box::new(ConstantTexture::<AlbedoChannel<Rgb>>::new(Rgb::new(0.5, 0.5, 0.5))),
/// );
/// ```
pub struct ConstantTexture<C: Channel> {
    value: C::Value,
}

impl<C: Channel> ConstantTexture<C> {
    /// 用常量值构造。
    pub fn new(value: C::Value) -> Self {
        Self { value }
    }

    /// 常量值。
    pub fn value(&self) -> C::Value {
        self.value
    }
}

impl<C: Channel> Texture<C> for ConstantTexture<C> {
    #[inline]
    fn eval(&self, _uv: Vec2, _differential: Option<&Differential>) -> C::Value {
        self.value
    }
}
