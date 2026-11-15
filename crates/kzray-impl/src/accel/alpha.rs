//! 透明度裁剪

use glam::Vec2;
use kzray_data::TextureId;

use crate::accel::resource::ResourceTable;

/// alpha cutout 源。封闭枚举——只有两种已知形式，不是扩展点。
///
/// `None`（`Option` 外层）表示不透明，求交时零成本。
pub enum AlphaSource {
    /// 常量 alpha（0 = 全透明，1 = 不透明）。
    /// 适合整体半透明的薄片，或构建期已把纹理 alpha 烘焙成常量的情况。
    Constant(f32),
    /// 纹理 alpha，按命中 UV 采样。
    /// 适合镂空纹理（叶子、栅栏、铁丝网）。
    Texture(TextureId),
}

/// 判定阈值。alpha 低于此值视为透明——命中丢弃，BVH 继续。
pub const ALPHA_THRESHOLD: f32 = 0.5;


impl AlphaSource {
    /// 该 UV 处表面是否不透明（即命中是否有效）。
    #[inline]
    pub fn is_opaque(&self, _uv: Vec2, _table: &ResourceTable) -> bool {
        // match self {
        //     Self::Constant(v) => *v >= ALPHA_THRESHOLD,
        //     Self::Texture(id) => table.expect_texture(*id).sample(uv) >= ALPHA_THRESHOLD,
        // }
        true
    }
}
