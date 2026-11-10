//! 采样器配置。**封闭域**——采样器种类是执行层核心组件，enum 保证类型安全。
//!
//! 参数不需要动画（种子、基数是静态配置），用普通标量。
//! 加新采样器 = 协议 major，因为执行层 `Sampler` trait 的实现必须同步。

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum SamplerConfig {
    Sobol { scramble_seed: u64 },
    Halton { base: u32 },
    Pcg { seed: u64 },
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig::Sobol { scramble_seed: 0 }
    }
}
