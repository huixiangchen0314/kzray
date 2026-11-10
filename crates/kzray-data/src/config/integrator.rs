//! 积分器配置。**封闭域**——积分器种类是执行层核心组件，enum 保证类型安全。
//!
//! 加新积分器 = 协议 major，因为执行层 `Integrator` trait 的实现必须同步。

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum IntegratorConfig {
    PathTracer(PathTracerConfig),
    PhotonMapper(PhotonMapperConfig),
}

impl Default for IntegratorConfig {
    fn default() -> Self {
        IntegratorConfig::PathTracer(PathTracerConfig::default())
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct PathTracerConfig {
    pub max_depth: u32,
    /// 俄罗斯轮盘赌起始深度（此后每 bounce 以概率 `rr_probability` 终止）。
    pub rr_start_depth: u32,
    /// 间接光照钳制（抑制 fireflies）。`0` = 不钳制。
    pub clamp_indirect: f32,
    /// 直接光照钳制。
    pub clamp_direct: f32,
}

impl Default for PathTracerConfig {
    fn default() -> Self {
        PathTracerConfig {
            max_depth: 8,
            rr_start_depth: 2,
            clamp_indirect: 10.0,
            clamp_direct: 0.0,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct PhotonMapperConfig {
    pub max_depth: u32,
    /// 发射光子总数。
    pub photon_count: u64,
    /// 密度估计搜索半径（世界单位）。
    pub search_radius: f32,
    /// 每个查询点最多取多少个邻近光子。
    pub max_photons_per_query: u32,
    /// Pass 2 的最终收集（final gather）开关。
    pub final_gather: bool,
}

impl Default for PhotonMapperConfig {
    fn default() -> Self {
        PhotonMapperConfig {
            max_depth: 8,
            photon_count: 1_000_000,
            search_radius: 0.1,
            max_photons_per_query: 100,
            final_gather: true,
        }
    }
}
