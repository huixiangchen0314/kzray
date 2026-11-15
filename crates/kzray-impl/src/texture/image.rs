//! kzray-impl/src/texture/image.rs
//!
//! 图像纹理。

use glam::Vec2;

use kzray_abs::texture::{Channel, Differential, Texture};

/// 图像纹理。像素存储为通道值 `C::Value`，双线性采样。
///
/// # 存储
///
/// `data` 是行主序像素缓冲，长度为 `width * height`。像素值已经是
/// `C::Value` 类型——解码（文件 RGBA → 通道值）在构造前完成，不在
/// 本类型内做。不同通道类型的解码规则不同（`AlphaChannel` 取单通道、
/// `NormalChannel` 把 `[0,1]` 映射到 `[-1,1]`、`AlbedoChannel<S>` 做
/// RGB → S 上采样），由各自的 loader 处理。
///
/// # 采样
///
/// 当前固定双线性，UV 超出 `[0, 1]` 时 clamp。MIP 在阶段 5 实现
/// （依赖相机微分）——届时 `eval` 按 `Differential` 选层级，无微分
/// 时仍走双线性。
///
/// # 尺寸约束
///
/// `width` 和 `height` 必须大于零。空图像（0×0）没有语义，构造时
/// 即 panic。这样 `sample_bilinear` 里 `width - 1` 不会下溢。
pub struct ImageTexture<C: Channel> {
    width: u32,
    height: u32,
    data: Vec<C::Value>,
}

impl<C: Channel> ImageTexture<C> {
    /// 从已解码的像素数据构造。
    ///
    /// # Panics
    ///
    /// - `width == 0` 或 `height == 0` 时 panic；
    /// - `data.len() != width * height` 时 panic。
    pub fn from_values(width: u32, height: u32, data: Vec<C::Value>) -> Self {
        assert!(
            width > 0 && height > 0,
            "image dimensions must be non-zero (got {width}x{height})"
        );
        assert_eq!(
            data.len() as u64,
            width as u64 * height as u64,
            "image data length ({}) must equal width * height ({width} * {height})",
            data.len()
        );
        Self { width, height, data }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// 双线性采样。UV 超出 `[0, 1]` 时 clamp。
    ///
    /// 采样点落在 texel 中心之间时，用 `C::lerp` 在四个邻居之间插值。
    /// 单像素图像（`1×1`）直接返回该像素——`x0 = x1 = 0`、`y0 = y1 = 0`，
    /// 插值退化为恒等。
    #[inline]
    fn sample_bilinear(&self, uv: Vec2) -> C::Value {
        // 构造时已保证 width/height > 0，`width - 1` 安全。
        let u = uv.x.clamp(0.0, 1.0) * (self.width - 1) as f32;
        let v = uv.y.clamp(0.0, 1.0) * (self.height - 1) as f32;

        let x0 = u.floor() as u32;
        let y0 = v.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = u - x0 as f32;
        let fy = v - y0 as f32;

        // 显式用 usize 计算索引，避免 `y * width` 在 u32 下溢出
        // （width * height 可能接近 u32::MAX，`y * width` 会先溢出）。
        let at = |x: u32, y: u32| {
            let idx = y as usize * self.width as usize + x as usize;
            self.data[idx]
        };

        let c00 = at(x0, y0);
        let c10 = at(x1, y0);
        let c01 = at(x0, y1);
        let c11 = at(x1, y1);

        // 先水平插值两条边，再垂直插值——与图像坐标 y 向下无关，
        // 插值本身的线性性保证结果与顺序无关。
        let top = C::lerp(c00, c10, fx);
        let bottom = C::lerp(c01, c11, fx);
        C::lerp(top, bottom, fy)
    }
}

impl<C: Channel> Texture<C> for ImageTexture<C> {
    #[inline]
    fn eval(&self, uv: Vec2, _differential: Option<&Differential>) -> C::Value {
        // MIP 暂不做，固定双线性。阶段 5 接入 Differential 后在此
        // 分支：`Some(diff)` 走 MIP 选择，`None` 保持双线性。
        self.sample_bilinear(uv)
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use kzray_abs::texture::{AlphaChannel, NormalChannel};

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    // ── f32 通道（AlphaChannel） ────────────────────────────

    #[test]
    fn f32_1x1_returns_pixel() {
        let tex = ImageTexture::<AlphaChannel>::from_values(1, 1, vec![0.7]);
        assert!(approx(tex.eval(Vec2::new(0.0, 0.0), None), 0.7));
        assert!(approx(tex.eval(Vec2::new(0.5, 0.5), None), 0.7));
        assert!(approx(tex.eval(Vec2::new(1.0, 1.0), None), 0.7));
    }

    #[test]
    fn f32_2x1_horizontal_lerp() {
        // 2×1 图像：[0.0, 1.0]
        let tex = ImageTexture::<AlphaChannel>::from_values(2, 1, vec![0.0, 1.0]);
        // 左端
        assert!(approx(tex.eval(Vec2::new(0.0, 0.5), None), 0.0));
        // 右端
        assert!(approx(tex.eval(Vec2::new(1.0, 0.5), None), 1.0));
        // 中点
        assert!(approx(tex.eval(Vec2::new(0.5, 0.5), None), 0.5));
        // 四分之一
        assert!(approx(tex.eval(Vec2::new(0.25, 0.5), None), 0.25));
    }

    #[test]
    fn f32_2x2_bilinear() {
        // 2×2 图像，四个角：
        // (0,0)=0.0  (1,0)=1.0
        // (0,1)=1.0  (1,1)=0.0
        // 数据行主序：[0, 1, 1, 0]
        let tex = ImageTexture::<AlphaChannel>::from_values(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
        // 中心：四角平均 = (0+1+1+0)/4 = 0.5
        assert!(approx(tex.eval(Vec2::new(0.5, 0.5), None), 0.5));
        // 每个角
        assert!(approx(tex.eval(Vec2::new(0.0, 0.0), None), 0.0));
        assert!(approx(tex.eval(Vec2::new(1.0, 0.0), None), 1.0));
        assert!(approx(tex.eval(Vec2::new(0.0, 1.0), None), 1.0));
        assert!(approx(tex.eval(Vec2::new(1.0, 1.0), None), 0.0));
    }

    #[test]
    fn f32_uv_clamped() {
        let tex = ImageTexture::<AlphaChannel>::from_values(2, 1, vec![0.0, 1.0]);
        // UV 小于 0 → clamp 到 0
        assert!(approx(tex.eval(Vec2::new(-0.5, 0.5), None), 0.0));
        // UV 大于 1 → clamp 到 1
        assert!(approx(tex.eval(Vec2::new(1.5, 0.5), None), 1.0));
    }

    // ── Vec3 通道（NormalChannel） ──────────────────────────

    #[test]
    fn vec3_lerp_componentwise() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 1.0, 1.0);
        let tex = ImageTexture::<NormalChannel>::from_values(2, 1, vec![a, b]);
        let mid = tex.eval(Vec2::new(0.5, 0.5), None);
        assert!(approx(mid.x, 0.5));
        assert!(approx(mid.y, 0.5));
        assert!(approx(mid.z, 0.5));
    }

    // ── 构造校验 ────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "dimensions must be non-zero")]
    fn reject_zero_width() {
        ImageTexture::<AlphaChannel>::from_values(0, 1, vec![]);
    }

    #[test]
    #[should_panic(expected = "dimensions must be non-zero")]
    fn reject_zero_height() {
        ImageTexture::<AlphaChannel>::from_values(1, 0, vec![]);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn reject_wrong_data_len() {
        ImageTexture::<AlphaChannel>::from_values(2, 2, vec![0.0, 1.0]);
    }

    // ── 通过 trait object 使用 ──────────────────────────────

    #[test]
    fn eval_through_trait_object() {
        use kzray_abs::AlphaTexture;

        let tex: Box<AlphaTexture> = Box::new(ImageTexture::<AlphaChannel>::from_values(
            2, 1, vec![0.0, 1.0],
        ));
        assert!(approx(tex.eval(Vec2::new(0.5, 0.5), None), 0.5));
    }
}
