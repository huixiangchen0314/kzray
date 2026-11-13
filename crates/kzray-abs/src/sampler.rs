//! 确定性随机数生成。每线程独立，不共享。
//!
//! # `dimension` 的语义
//!
//! `dimension` 是 **1D 维度基址**，不是 2D 维度索引。约定：
//!
//! - `next_f32(d)` 消耗第 `d` 维，下一维是 `d + 1`；
//! - `next_2d(d)` 消耗第 `d`、`d + 1` 维，下一维是 `d + 2`；
//! - `next_array(d, out)` 消耗第 `d` 到 `d + out.len() - 1` 维。
//!
//! 所有方法都按"从 `d` 开始的连续维度"消耗，调用方用 `d + 消耗数`
//! 推进基址。这是维度追踪模型的基础——`Light` / `BSDF` / `LightSampler`
//! 的 `consumed` 都按这个语义计算。
//!
//! # 样本生命周期
//!
//! 采样器有**样本**的概念：每个像素的每个样本独立，样本内的维度
//! 按 `next_f32` / `next_2d` 的调用顺序消耗，样本间通过
//! [`Sampler::start_pixel_sample`] 重置。
//!
//! - **PCG**：无状态随机数，`start_pixel_sample` 可空操作，或用
//!   `(pixel, sample_index)` 做 seed 实现可复现；
//! - **Sobol / Halton**：`start_pixel_sample` 重置序列位置，后续
//!   `next_f32(d)` 返回 `Sobol(d, sample_index)`；
//! - **分层采样**：`start_pixel_sample` 记录样本索引，用于分层抖动。
//!
//! 没有 `start_pixel_sample` 时，采样器无法知道"哪个样本开始了"，
//! 低差异序列会在像素间错位——这是维度追踪模型缺失的语义。
//!
//! # 线程模型
//!
//! 只 `Send` 不 `Sync`——每个线程自己的 Sampler，不做共享。并行渲染
//! 时，每个线程构造自己的 `Sampler`，线程间不共享状态。这与
//! `Film: Send + Sync` 不同——Film 是共享累加容器，Sampler 是
//! 线程私有状态。

use glam::UVec2;

/// 采样器。
///
/// 构造时绑定 `(seed, sample_index)`，`seed` 由 `seed_for` 派生。
/// 每个线程自己的 Sampler，不做共享（`Send` 不 `Sync`）。
///
/// # 维度基址
///
/// `dimension` 是 1D 维度基址。`next_f32(d)` / `next_2d(d)` /
/// `next_array(d, out)` 从 `d` 开始消耗连续维度。调用方用
/// `d + 消耗数` 推进基址。
///
/// # 样本生命周期
///
/// 每个像素的每个样本开始时调用 [`Sampler::start_pixel_sample`]，
/// 采样器据此重置序列位置（低差异采样器）或记录样本索引（分层采样）。
/// 样本内的维度按 `next_*` 调用顺序消耗。
pub trait Sampler: Send {
    /// 开始新的像素样本。
    ///
    /// 调用时机：每个像素的每个样本开始时调用一次，在第一次
    /// `next_f32` / `next_2d` 之前。
    ///
    /// 参数：
    /// - `pixel`：像素坐标 `(x, y)`；
    /// - `sample_index`：该像素内的第几个样本（从 0 开始）。
    ///
    /// # 语义
    ///
    /// - **Sobol / Halton**：重置序列位置，后续 `next_f32(d)` 返回
    ///   `Sobol(d, sample_index)`。没有这个调用，低差异序列会在像素间
    ///   错位，视觉上出现规律性伪影；
    /// - **PCG**：可空操作（无状态随机数），或用 `(pixel, sample_index)`
    ///   做 seed 实现可复现——渲染中断后重跑，同一像素同一样本得到
    ///   同样的值；
    /// - **分层采样**：记录样本索引，用于分层抖动。
    ///
    /// # 契约
    ///
    /// - 同一像素内，不同 `sample_index` 的样本独立——同一次
    ///   `start_pixel_sample(pixel, i)` 和 `start_pixel_sample(pixel, j)`
    ///   之后（`i != j`），`next_f32(d)` 返回不同的值（低差异采样器）
    ///   或独立的值（无状态采样器）；
    /// - 不同像素的同索引样本独立——`start_pixel_sample(p, i)` 和
    ///   `start_pixel_sample(q, i)` 之后（`p != q`），`next_f32(d)` 返回
    ///   不同的值（否则像素间会相关，图像出现规律性伪影）；
    /// - 可复现采样器（用 seed hash）：`start_pixel_sample(p, i)` 之后，
    ///   `next_f32` 序列完全确定，与调用历史无关。
    fn start_pixel_sample(&mut self, pixel: UVec2, sample_index: u32);

    /// 返回第 `dimension` 维的 1D 样本。
    ///
    /// 消耗第 `dimension` 维。下一维是 `dimension + 1`。
    ///
    /// 返回值依赖当前样本（由 [`Sampler::start_pixel_sample`] 设置）。
    fn next_f32(&mut self, dimension: u32) -> f32;

    /// 返回第 `dimension`、`dimension + 1` 维的 2D 样本。
    ///
    /// 消耗 2 维。下一维是 `dimension + 2`。
    ///
    /// 默认实现调用两次 `next_f32`，保证维度顺序正确（先 `dimension`，
    /// 后 `dimension + 1`）。实现方可以覆盖以利用 2D 采样的低差异性
    /// 优化（如 Sobol 的 2D 联合分布）。
    fn next_2d(&mut self, dimension: u32) -> (f32, f32) {
        (
            self.next_f32(dimension),
            self.next_f32(dimension + 1),
        )
    }

    /// 填充 `out`，第 `i` 个元素来自第 `dimension + i` 维。
    ///
    /// 消耗 `out.len()` 维。下一维是 `dimension + out.len()`。
    fn next_array(&mut self, dimension: u32, out: &mut [f32]) {
        for (i, v) in out.iter_mut().enumerate() {
            *v = self.next_f32(dimension + i as u32);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────── 测试用 Sampler ───────────

    /// 可复现的伪采样器：用 `(pixel, sample_index, dimension)` 的
    /// hash 生成确定值。用于验证 `start_pixel_sample` 的语义。
    struct HashSampler {
        pixel: UVec2,
        sample_index: u32,
        seed: u64,
    }

    impl HashSampler {
        fn new(seed: u64) -> Self {
            HashSampler {
                pixel: UVec2::ZERO,
                sample_index: 0,
                seed,
            }
        }

        fn hash(&self, dimension: u32) -> f32 {
            // 简单的 FNV-1a 变体，保证确定性和低相关性
            let mut h = self.seed;
            h ^= (self.pixel.x as u64) << 32;
            h ^= self.pixel.y as u64;
            h ^= (self.sample_index as u64) << 16;
            h ^= dimension as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h ^= h >> 33;
            // 取低 24 位映射到 [0, 1)
            (h & 0xFFFFFF) as f32 / 0x1000000 as f32
        }
    }

    impl Sampler for HashSampler {
        fn start_pixel_sample(&mut self, pixel: UVec2, sample_index: u32) {
            self.pixel = pixel;
            self.sample_index = sample_index;
        }

        fn next_f32(&mut self, dimension: u32) -> f32 {
            self.hash(dimension)
        }
    }

    /// 计数采样器：记录所有调用，用于验证维度消耗顺序。
    struct RecordingSampler {
        calls: Vec<u32>,
    }

    impl RecordingSampler {
        fn new() -> Self {
            RecordingSampler { calls: Vec::new() }
        }
    }

    impl Sampler for RecordingSampler {
        fn start_pixel_sample(&mut self, _pixel: UVec2, _sample_index: u32) {}

        fn next_f32(&mut self, dimension: u32) -> f32 {
            self.calls.push(dimension);
            0.5
        }
    }

    // ─────────── 可复现性 ───────────

    #[test]
    fn same_pixel_same_sample_same_values() {
        let mut s1 = HashSampler::new(42);
        let mut s2 = HashSampler::new(42);
        s1.start_pixel_sample(UVec2::new(10, 20), 5);
        s2.start_pixel_sample(UVec2::new(10, 20), 5);
        for d in 0..10 {
            assert_eq!(s1.next_f32(d), s2.next_f32(d));
        }
    }

    #[test]
    fn different_sample_index_different_values() {
        let mut s = HashSampler::new(42);
        s.start_pixel_sample(UVec2::new(10, 20), 0);
        let v0 = s.next_f32(0);
        s.start_pixel_sample(UVec2::new(10, 20), 1);
        let v1 = s.next_f32(0);
        assert_ne!(v0, v1, "different sample indices should give different values");
    }

    #[test]
    fn different_pixel_different_values() {
        let mut s = HashSampler::new(42);
        s.start_pixel_sample(UVec2::new(10, 20), 0);
        let v1 = s.next_f32(0);
        s.start_pixel_sample(UVec2::new(10, 21), 0);
        let v2 = s.next_f32(0);
        assert_ne!(v1, v2, "different pixels should give different values");
    }

    #[test]
    fn different_dimension_different_values() {
        let mut s = HashSampler::new(42);
        s.start_pixel_sample(UVec2::new(10, 20), 0);
        let v0 = s.next_f32(0);
        let v1 = s.next_f32(1);
        assert_ne!(v0, v1);
    }

    #[test]
    fn values_in_unit_interval() {
        let mut s = HashSampler::new(42);
        s.start_pixel_sample(UVec2::new(0, 0), 0);
        for d in 0..100 {
            let v = s.next_f32(d);
            assert!((0.0..1.0).contains(&v), "value {v} out of [0, 1)");
        }
    }

    // ─────────── next_2d 维度语义 ───────────

    #[test]
    fn next_2d_consumes_two_consecutive_dimensions() {
        let mut s = RecordingSampler::new();
        let _ = s.next_2d(5);
        // next_2d(5) 应调用 next_f32(5) 和 next_f32(6)
        assert_eq!(s.calls, vec![5, 6]);
    }

    #[test]
    fn next_2d_then_next_2d_advances_by_two() {
        let mut s = RecordingSampler::new();
        let _ = s.next_2d(0);
        let _ = s.next_2d(2);
        // 第一次 [0, 1]，第二次 [2, 3]，连续
        assert_eq!(s.calls, vec![0, 1, 2, 3]);
    }

    #[test]
    fn next_f32_then_next_2d_advances_correctly() {
        // 模拟 bsdf 的组合采样：选择维 1 + 方向维 2
        let mut s = RecordingSampler::new();
        let _ = s.next_f32(0);      // 选择维
        let _ = s.next_2d(1);       // 方向维
        assert_eq!(s.calls, vec![0, 1, 2]); // 连续，不跳过
    }

    // ─────────── next_array 维度语义 ───────────

    #[test]
    fn next_array_consumes_consecutive_dimensions() {
        let mut s = RecordingSampler::new();
        let mut out = [0.0; 4];
        s.next_array(3, &mut out);
        assert_eq!(s.calls, vec![3, 4, 5, 6]);
    }

    #[test]
    fn next_array_empty_is_noop() {
        let mut s = RecordingSampler::new();
        let mut out = [];
        s.next_array(5, &mut out);
        assert!(s.calls.is_empty());
    }

    #[test]
    fn next_array_fills_values() {
        let mut s = HashSampler::new(42);
        s.start_pixel_sample(UVec2::ZERO, 0);
        let mut out = [0.0; 3];
        s.next_array(0, &mut out);
        for v in out {
            assert!((0.0..1.0).contains(&v));
        }
    }

    // ─────────── 维度顺序不跳过 ───────────

    #[test]
    fn dimensions_not_skipped_when_mixed() {
        // 混合调用 next_f32 / next_2d / next_array，验证维度连续
        let mut s = RecordingSampler::new();
        let _ = s.next_f32(0);          // dim 0
        let _ = s.next_2d(1);           // dim 1, 2
        let mut out = [0.0; 2];
        s.next_array(3, &mut out);      // dim 3, 4
        assert_eq!(s.calls, vec![0, 1, 2, 3, 4]);
    }

    // ─────────── Send 约束 ───────────

    #[test]
    fn sampler_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<HashSampler>();
        assert_send::<RecordingSampler>();
    }

    // 注意：Sampler 不要求 Sync，所以不测试 Sync
}
