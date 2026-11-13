//! 光源采样器。
//!
//! 两个实现：
//!
//! - [`UniformLightSampler`]：均匀选择，每个光源等概率；
//! - [`PowerLightSampler`]：按功率加权，概率正比于 `Light::phi` 的标量
//!   近似。
//!
//! 两者都消耗 1 维（选择光源）。`consumed` 恒为 1，
//! `max_sample_dimensions` 恒为 1。
//!
//! # 为什么不需要 `Light` 引用
//!
//! `LightSampler` 只负责选择，不负责采样光源内部。返回 `LightId` 后，
//! 调用方通过 `Scene::light(id)` 查 `&dyn Light<S>` 再调 `sample_li`。
//! 所以采样器只需要 `Vec<LightId>` + 概率表。

use kzray_abs::light::{
    LightSampleContext, LightSelection, LightSampler,
};
use kzray_abs::sampler::Sampler;
use kzray_data::common::LightId;

// ═══════════════════════════════════════════════════════════════
// UniformLightSampler
// ═══════════════════════════════════════════════════════════════

/// 均匀光源采样器。
///
/// 所有光源等概率选择。`pdf = 1/N`。
///
/// 适用场景：
///
/// - 调试 / 对比（验证 MIS 是否正确）；
/// - 所有光源功率接近时（均匀与功率加权效果相同，均匀更简单）。
#[derive(Clone, Debug)]
pub struct UniformLightSampler {
    light_ids: Vec<LightId>,
}

impl UniformLightSampler {
    /// 构造均匀采样器。
    ///
    /// `light_ids` 是光源 ID 列表，顺序无关。空列表时 `sample` 总是
    /// 返回 `None`。
    pub fn new(light_ids: Vec<LightId>) -> Self {
        UniformLightSampler { light_ids }
    }

    /// 光源数量。
    pub fn len(&self) -> usize {
        self.light_ids.len()
    }

    /// 是否无光源。
    pub fn is_empty(&self) -> bool {
        self.light_ids.is_empty()
    }
}

impl LightSampler for UniformLightSampler {
    fn sample(
        &self,
        _ctx: &LightSampleContext,
        sampler: &mut impl Sampler,
        dimension: u32,
    ) -> Option<LightSelection> {
        if self.light_ids.is_empty() {
            return None;
        }
        let u = sampler.next_f32(dimension);
        let n = self.light_ids.len();
        // [0, 1) × n → [0, n)，floor 到整数索引
        let idx = ((u * n as f32) as usize).min(n - 1);
        Some(LightSelection {
            light_id: self.light_ids[idx],
            light_pdf: 1.0 / n as f32,
            consumed: 1,
        })
    }

    fn pdf(&self, _ctx: &LightSampleContext, light_id: LightId) -> f32 {
        if self.light_ids.contains(&light_id) {
            1.0 / self.light_ids.len() as f32
        } else {
            0.0
        }
    }

    fn max_sample_dimensions(&self) -> u32 {
        1
    }
}

// ═══════════════════════════════════════════════════════════════
// PowerLightSampler
// ═══════════════════════════════════════════════════════════════

/// 功率加权光源采样器。
///
/// 概率正比于每个光源的 `phi`（标量近似）。构造时预计算累积分布，
/// 采样时二分查找。
///
/// # 标量转换
///
/// `Light::phi` 返回光谱值 `S`，采样器需要标量概率。标量转换由构造方
/// 显式完成——[`PowerLightSampler::from_weights`] 接收已转换的
/// `(LightId, f32)` 列表。
///
/// 常用转换：
///
/// - `Rgb`：`|phi|.max_abs()` 或亮度近似（如 `0.2126 R + 0.7152 G + 0.0722 B`）；
/// - `Spectrum<N>`：`|phi|.max_abs()` 或积分平均。
///
/// 构造方（场景构建层）负责选择合适的转换。采样器不暴露"模糊的标量
/// 功率近似"接口。
///
/// # 零权重
///
/// 权重为 0 的光源不参与选择（概率为 0）。所有光源权重为 0 时构造
/// panic——无效配置。
#[derive(Clone, Debug)]
pub struct PowerLightSampler {
    light_ids: Vec<LightId>,
    /// 累积分布：`cdf[i] = P(light ≤ i)`。
    cdf: Vec<f32>,
}

impl PowerLightSampler {
    /// 从 `(LightId, weight)` 列表构造。
    ///
    /// `weight` 是光源的标量功率近似，非负。权重为 0 的光源从采样中
    /// 排除（不进入 `light_ids`）。
    ///
    /// # Panics
    ///
    /// - `weights` 为空；
    /// - 所有 `weight` 之和为 0。
    pub fn from_weights(weights: Vec<(LightId, f32)>) -> Self {
        assert!(!weights.is_empty(), "PowerLightSampler needs at least one light");

        let total: f32 = weights.iter().map(|(_, w)| *w).sum();
        assert!(total > 0.0, "PowerLightSampler total weight must be positive");

        let mut light_ids = Vec::with_capacity(weights.len());
        let mut cdf = Vec::with_capacity(weights.len());
        let mut acc = 0.0_f32;
        for (id, w) in weights {
            if w <= 0.0 {
                continue;
            }
            acc += w;
            light_ids.push(id);
            cdf.push(acc / total);
        }

        PowerLightSampler { light_ids, cdf }
    }

    /// 光源数量（权重 > 0 的）。
    pub fn len(&self) -> usize {
        self.light_ids.len()
    }

    /// 是否无有效光源。
    pub fn is_empty(&self) -> bool {
        self.light_ids.is_empty()
    }
}

impl LightSampler for PowerLightSampler {
    fn sample(
        &self,
        _ctx: &LightSampleContext,
        sampler: &mut impl Sampler,
        dimension: u32,
    ) -> Option<LightSelection> {
        if self.light_ids.is_empty() {
            return None;
        }
        let u = sampler.next_f32(dimension);
        // 二分查找：找第一个 cdf[i] >= u 的索引
        let idx = self.cdf.partition_point(|&p| p < u);
        // 浮点误差兜底：u >= cdf[last] 时 partition_point 返回 len
        let idx = idx.min(self.light_ids.len() - 1);

        let lower = if idx == 0 { 0.0 } else { self.cdf[idx - 1] };
        let light_pdf = self.cdf[idx] - lower;

        Some(LightSelection {
            light_id: self.light_ids[idx],
            light_pdf,
            consumed: 1,
        })
    }

    fn pdf(&self, _ctx: &LightSampleContext, light_id: LightId) -> f32 {
        // 线性查找。光源数量通常小（< 100），可接受。
        // 如果光源多，可以用 HashMap<LightId, usize> 加速。
        for (idx, &id) in self.light_ids.iter().enumerate() {
            if id == light_id {
                let lower = if idx == 0 { 0.0 } else { self.cdf[idx - 1] };
                return self.cdf[idx] - lower;
            }
        }
        0.0
    }

    fn max_sample_dimensions(&self) -> u32 {
        1
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use glam::UVec2;

    struct FixedSampler {
        value: f32,
    }

    impl Sampler for FixedSampler {
        fn start_pixel_sample(&mut self, _pixel: UVec2, _sample_index: u32) {}
        fn next_f32(&mut self, _dimension: u32) -> f32 {
            self.value
        }
    }

    fn ctx() -> LightSampleContext {
        LightSampleContext {
            position: glam::Vec3::ZERO,
            normal: glam::Vec3::Y,
        }
    }

    // ─────────── UniformLightSampler ───────────

    #[test]
    fn uniform_empty_returns_none() {
        let s = UniformLightSampler::new(vec![]);
        let mut sampler = FixedSampler { value: 0.5 };
        assert!(s.sample(&ctx(), &mut sampler, 0).is_none());
    }

    #[test]
    fn uniform_selects_all_ids() {
        let ids = vec![LightId::from(10), LightId::from(20), LightId::from(30)];
        let s = UniformLightSampler::new(ids.clone());
        let n = ids.len() as f32;

        for (i, expected_id) in ids.iter().enumerate() {
            let u = (i as f32 + 0.5) / n;
            let mut sampler = FixedSampler { value: u };
            let sel = s.sample(&ctx(), &mut sampler, 0).unwrap();
            assert_eq!(sel.light_id, *expected_id);
            assert!((sel.light_pdf - 1.0 / n).abs() < 1e-6);
            assert_eq!(sel.consumed, 1);
        }
    }

    #[test]
    fn uniform_pdf_consistent() {
        let ids = vec![LightId::from(1), LightId::from(2), LightId::from(3)];
        let s = UniformLightSampler::new(ids.clone());
        for id in &ids {
            assert!((s.pdf(&ctx(), *id) - 1.0 / 3.0).abs() < 1e-6);
        }
        assert_eq!(s.pdf(&ctx(), LightId::from(99)), 0.0);
    }

    #[test]
    fn uniform_sample_pdf_consistent() {
        let ids = vec![LightId::from(1), LightId::from(2)];
        let s = UniformLightSampler::new(ids);
        let mut sampler = FixedSampler { value: 0.3 };
        let sel = s.sample(&ctx(), &mut sampler, 0).unwrap();
        let expected = s.pdf(&ctx(), sel.light_id);
        assert!((sel.light_pdf - expected).abs() < 1e-6);
    }

    #[test]
    fn uniform_u_boundary() {
        // u 接近 1.0 时不越界
        let s = UniformLightSampler::new(vec![LightId::from(1), LightId::from(2)]);
        let mut sampler = FixedSampler { value: 0.999999 };
        let sel = s.sample(&ctx(), &mut sampler, 0);
        assert!(sel.is_some());
    }

    // ─────────── PowerLightSampler ───────────

    #[test]
    fn power_pdf_proportional_to_weight() {
        let s = PowerLightSampler::from_weights(vec![
            (LightId::from(1), 3.0),
            (LightId::from(2), 1.0),
        ]);
        // 概率 0.75 / 0.25
        assert!((s.pdf(&ctx(), LightId::from(1)) - 0.75).abs() < 1e-6);
        assert!((s.pdf(&ctx(), LightId::from(2)) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn power_pdf_sums_to_one() {
        let s = PowerLightSampler::from_weights(vec![
            (LightId::from(1), 0.5),
            (LightId::from(2), 0.3),
            (LightId::from(3), 0.2),
        ]);
        let mut sum = 0.0;
        for id in [1u32, 2, 3] {
            sum += s.pdf(&ctx(), LightId::from(id as u64));
        }
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn power_sample_respects_cdf() {
        let s = PowerLightSampler::from_weights(vec![
            (LightId::from(1), 1.0),  // pdf 0.5, cdf [0, 0.5]
            (LightId::from(2), 1.0),  // pdf 0.5, cdf [0.5, 1.0]
        ]);

        // u=0.2 → 选 1
        let mut sampler = FixedSampler { value: 0.2 };
        let sel = s.sample(&ctx(), &mut sampler, 0).unwrap();
        assert_eq!(sel.light_id, LightId::from(1));

        // u=0.7 → 选 2
        let mut sampler = FixedSampler { value: 0.7 };
        let sel = s.sample(&ctx(), &mut sampler, 0).unwrap();
        assert_eq!(sel.light_id, LightId::from(2));
    }

    #[test]
    fn power_zero_weight_excluded() {
        let s = PowerLightSampler::from_weights(vec![
            (LightId::from(1), 1.0),
            (LightId::from(2), 0.0),  // 排除
        ]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.pdf(&ctx(), LightId::from(2)), 0.0);
    }

    #[test]
    #[should_panic(expected = "at least one light")]
    fn power_empty_weights_panics() {
        PowerLightSampler::from_weights(vec![]);
    }

    #[test]
    #[should_panic(expected = "total weight must be positive")]
    fn power_all_zero_weights_panics() {
        PowerLightSampler::from_weights(vec![
            (LightId::from(1), 0.0),
            (LightId::from(2), 0.0),
        ]);
    }

    #[test]
    fn power_u_boundary() {
        // u 接近 1.0，partition_point 可能返回 len，需兜底
        let s = PowerLightSampler::from_weights(vec![
            (LightId::from(1), 1.0),
            (LightId::from(2), 1.0),
        ]);
        let mut sampler = FixedSampler { value: 0.9999999 };
        let sel = s.sample(&ctx(), &mut sampler, 0);
        assert!(sel.is_some());
    }

    #[test]
    fn power_sample_pdf_consistent() {
        let s = PowerLightSampler::from_weights(vec![
            (LightId::from(1), 2.0),
            (LightId::from(2), 1.0),
            (LightId::from(3), 3.0),
        ]);
        for u in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let mut sampler = FixedSampler { value: u };
            let sel = s.sample(&ctx(), &mut sampler, 0).unwrap();
            let expected = s.pdf(&ctx(), sel.light_id);
            assert!(
                (sel.light_pdf - expected).abs() < 1e-5,
                "pdf mismatch at u={}: sample={}, expected={}",
                u,
                sel.light_pdf,
                expected
            );
        }
    }

    // ─────────── 通用 ───────────

    #[test]
    fn samplers_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<UniformLightSampler>();
        assert_send_sync::<PowerLightSampler>();
    }
}
