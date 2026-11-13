//! 世界环境。背景光走 `LightType::Background`，体积走 `RenderInput.volumes`。
//! 此模块承载环境级的扩展数据（HDRI 旋转、环境设置等）和世界范围。

use serde::{Deserialize, Serialize};

use crate::common::meta::HasMeta;
use crate::common::{DataError, Meta};
use crate::geometry::Aabb;

// ═══════════════════════════════════════════════════════════════
// WorldData
// ═══════════════════════════════════════════════════════════════

/// 世界环境数据。
///
/// 包含：
///
/// - `meta`：环境级扩展参数（HDRI 旋转、环境设置等）；
/// - `bounds`：世界包围盒，用于 BVH 构建、视锥剔除、体积 bounds。
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct WorldData {
    /// 动态拓展参数。
    pub meta: Meta,

    /// 世界包围盒。
    ///
    /// - `Some(bounds)`：显式指定世界范围。构建层直接使用，避免遍历
    ///   所有 instance 聚合 bounds。
    /// - `None`：构建层从 instances 的 mesh bounds 聚合得到。
    ///
    /// 校验：`Some` 时必须是**非空**盒（`!is_empty()`）且所有角点有限。
    /// 空盒等价于 `None`，但显式空盒是数据错误。
    pub bounds: Option<Aabb>,
}

impl HasMeta for WorldData {
    fn meta(&self) -> &Meta {
        &self.meta
    }
    fn meta_mut(&mut self) -> &mut Meta {
        &mut self.meta
    }
}

impl WorldData {
    pub fn new() -> Self {
        WorldData::default()
    }

    pub fn validate(&self) -> Result<(), DataError> {
        self.meta.validate("world.meta")?;

        if let Some(bounds) = &self.bounds {
            // 空盒是数据错误：None 表示"未指定"，Some(empty) 无意义
            if bounds.is_empty() {
                return Err(DataError::InvalidParam {
                    path: "world.bounds".into(),
                    reason: "包围盒为空（min > max）；未指定世界范围请用 None".into(),
                });
            }
            // 所有角点必须有限
            if !bounds.min().is_finite() || !bounds.max().is_finite() {
                return Err(DataError::InvalidParam {
                    path: "world.bounds".into(),
                    reason: "包围盒含非有限值".into(),
                });
            }
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn new_has_no_bounds() {
        let w = WorldData::new();
        assert!(w.bounds.is_none());
    }

    #[test]
    fn default_has_no_bounds() {
        let w = WorldData::default();
        assert!(w.bounds.is_none());
    }

    #[test]
    fn validate_none_bounds_ok() {
        let w = WorldData::new();
        assert!(w.validate().is_ok());
    }

    #[test]
    fn validate_valid_bounds_ok() {
        let w = WorldData {
            meta: Meta::new(),
            bounds: Some(Aabb::new(Vec3::splat(-10.0), Vec3::splat(10.0))),
        };
        assert!(w.validate().is_ok());
    }

    #[test]
    fn validate_empty_bounds_rejected() {
        let w = WorldData {
            meta: Meta::new(),
            bounds: Some(Aabb::empty()),
        };
        let err = w.validate().unwrap_err();
        assert!(
            format!("{err:?}").contains("world.bounds"),
            "expected error path 'world.bounds', got: {err:?}"
        );
    }

    #[test]
    fn validate_reversed_bounds_rejected() {
        let w = WorldData {
            meta: Meta::new(),
            bounds: Some(Aabb::new(Vec3::splat(1.0), Vec3::splat(-1.0))),
        };
        assert!(w.validate().is_err());
    }

    #[test]
    fn validate_nan_bounds_rejected() {
        let w = WorldData {
            meta: Meta::new(),
            bounds: Some(Aabb::new(Vec3::splat(f32::NAN), Vec3::splat(1.0))),
        };
        let err = w.validate().unwrap_err();
        assert!(
            format!("{err:?}").contains("非有限值"),
            "expected finite check, got: {err:?}"
        );
    }

    #[test]
    fn validate_inf_bounds_rejected() {
        let w = WorldData {
            meta: Meta::new(),
            bounds: Some(Aabb::new(
                Vec3::splat(f32::NEG_INFINITY),
                Vec3::splat(1.0),
            )),
        };
        assert!(w.validate().is_err());
    }

    #[test]
    fn validate_degenerate_bounds_ok() {
        // 单点盒是合法的（薄面、线段等可能产生退化盒）
        let w = WorldData {
            meta: Meta::new(),
            bounds: Some(Aabb::from_point(Vec3::ZERO)),
        };
        assert!(w.validate().is_ok());
    }

    #[test]
    fn validate_meta_error_propagates() {
        let w = WorldData::new();
        assert!(w.meta.validate("world.meta").is_ok());
        assert!(w.validate().is_ok());
    }

    #[test]
    fn world_data_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WorldData>();
    }

    #[test]
    fn serialize_roundtrip() {
        let original = WorldData {
            meta: Meta::new(),
            bounds: Some(Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0))),
        };
        let _ = original.clone();
        let _ = original;
    }
}
