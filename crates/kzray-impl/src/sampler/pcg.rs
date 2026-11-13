//! PCG 采样器。
//!
//! 使用 `rand_pcg::Pcg64Mcg` 作为底层 RNG。`start_pixel_sample` 通过
//! 混合 `(seed, pixel, sample_index)` 重置 RNG，保证：
//!
//! - 同一像素的同一样本索引，每次重跑得到相同的随机数序列（可复现）；
//! - 不同像素 / 不同样本索引 / 不同种子，序列独立；
//! - 值域 `[0, 1)`。
//!
//! `dimension` 参数在 PCG 中被忽略——PCG 是伪随机数生成器，不依赖
//! 维度索引。但接口保留 `dimension`，以便 `Sampler` trait 统一。

use glam::UVec2;
// rand_core 0.10: RngCore 已重命名为 Rng
use rand_core::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

use kzray_abs::sampler::Sampler;

// ═══════════════════════════════════════════════════════════════
// PCGSampler
// ═══════════════════════════════════════════════════════════════

/// PCG 采样器。
///
/// 构造时传入全局 `seed`。`start_pixel_sample` 用 `(seed, pixel,
/// sample_index)` 混合出新的种子，重置 RNG。`next_f32` 从 RNG 生成
/// `[0, 1)` 的浮点数。
///
/// # 可复现性
///
/// 对相同的 `(seed, pixel, sample_index)`，`start_pixel_sample` 后的
/// `next_f32` 序列完全确定。渲染中断后重跑，同一像素同一样本索引
/// 得到相同结果。
///
/// # 维度
///
/// PCG 不依赖 `dimension`——每次 `next_f32` 独立生成随机数，与维度
/// 索引无关。`dimension` 参数被忽略。
pub struct PCGSampler {
    /// 底层 RNG。`start_pixel_sample` 时重新 seed。
    rng: Pcg64Mcg,
    /// 全局种子。与 `pixel` / `sample_index` 混合后作为 RNG 的种子。
    seed: u64,
}

impl PCGSampler {
    /// 构造 PCG 采样器。
    ///
    /// `seed` 是全局种子，通常由渲染器在启动时指定（如命令行参数
    /// `--seed`）。相同 `seed` 的两次渲染产生相同结果。
    pub fn new(seed: u64) -> Self {
        PCGSampler {
            rng: Pcg64Mcg::seed_from_u64(seed),
            seed,
        }
    }

    /// 当前全局种子。
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl Sampler for PCGSampler {
    fn start_pixel_sample(&mut self, pixel: UVec2, sample_index: u32) {
        let combined = mix_seed(self.seed, pixel, sample_index);
        self.rng = Pcg64Mcg::seed_from_u64(combined);
    }

    fn next_f32(&mut self, _dimension: u32) -> f32 {
        // 取 u32 的高 24 位，映射到 [0, 1)。
        // 24 位精度足够渲染使用（f32 尾数 23 位），避免 1.0 被取到。
        let v = self.rng.next_u32() >> 8; // 24 位有效
        (v as f32) / ((1u32 << 24) as f32)
    }
}

// ═══════════════════════════════════════════════════════════════
// 种子混合
// ═══════════════════════════════════════════════════════════════

/// 混合 `(seed, pixel, sample_index)` 成一个 `u64` 种子。
///
/// 使用 splitmix64 的混合函数，保证：
///
/// - 输入的任何一位变化，输出的所有位都有约 50% 概率变化（雪崩效应）；
/// - 不同输入的混合结果不相关。
///
/// `pixel.x` / `pixel.y` / `sample_index` 先组合成一个 `u64`，再与
/// `seed` 混合。
fn mix_seed(seed: u64, pixel: UVec2, sample_index: u32) -> u64 {
    let mut h = seed;
    h ^= (pixel.x as u64).wrapping_mul(0x9E3779B97F4A7C15);
    h ^= (pixel.y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F);
    h ^= (sample_index as u64).wrapping_mul(0x165667B19E3779F9);

    // splitmix64 终混
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    h
}
