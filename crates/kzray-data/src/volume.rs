//! 体积。**独立表**——体积是场景实体，有边界、有位置、有优先级。
//!
//! 散射模型（`VolumeData`）开放 `kind`；边界（`VolumeBounds`）是封闭 enum
//! ——形状种类有限，封闭可枚举。两者组合成 `VolumeInstance`。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::common::meta::HasMeta;
use crate::common::{DataError, MeshId, Meta, Param, Transform};

/// 散射模型的描述。**开放 `kind`**——物理模型（homogeneous /
/// heterogeneous / atmosphere / 第三方）由执行层注册表解释。
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct VolumeData {
    /// 分派键。常见：`"homogeneous"` / `"heterogeneous"` / `"atmosphere"`。
    pub kind: String,
    /// 参数。开放 map，值可以是常量或 Field 引用（密度贴图等）。
    pub params: BTreeMap<String, Param>,
}

impl VolumeData {
    pub fn new(kind: impl Into<String>) -> Self {
        VolumeData { kind: kind.into(), params: BTreeMap::new() }
    }

    pub fn with_param(mut self, key: impl Into<String>, value: Param) -> Self {
        self.params.insert(key.into(), value);
        self
    }

    pub fn validate(&self, path: &str) -> Result<(), DataError> {
        for (k, v) in &self.params {
            v.validate(&format!("{path}.{k}"))?;
        }
        Ok(())
    }
}

/// 体积实例：散射模型 + 边界 + 优先级 + 扩展数据。
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct VolumeInstance {
    pub volume: VolumeData,
    pub bounds: VolumeBounds,
    /// 多体积重叠时的叠加优先级。执行层按此决定混合顺序。
    /// 默认 0。
    pub priority: i32,
    pub meta: Meta,
}

impl HasMeta for VolumeInstance {
    fn meta(&self) -> &Meta { &self.meta }
    fn meta_mut(&mut self) -> &mut Meta { &mut self.meta }
}

impl VolumeInstance {
    pub fn new(volume: VolumeData, bounds: VolumeBounds) -> Self {
        VolumeInstance { volume, bounds, priority: 0, meta: Meta::new() }
    }

    pub fn validate(&self, path: &str) -> Result<(), DataError> {
        self.volume.validate(&format!("{path}.volume"))?;
        self.bounds.validate(&format!("{path}.bounds"))?;
        self.meta.validate(&format!("{path}.meta"))?;
        Ok(())
    }
}

/// 体积边界。**封闭 enum**——形状种类物理上有限。
///
/// 边界不需要动画：单帧渲染中体积的"移动"由上游烘焙成每帧的 `transform`
/// （`TransformedBox`）或变换后的顶点数据。
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum VolumeBounds {
    /// 无界——整个场景。全局雾、大气。
    Infinite,
    /// 轴对齐包围盒。
    Aabb { min: [f32; 3], max: [f32; 3] },
    /// 球体。
    Sphere { center: [f32; 3], radius: f32 },
    /// 变换后的盒体。`half_extents` 沿局部 X/Y/Z 轴。
    TransformedBox { transform: Transform, half_extents: [f32; 3] },
    /// 网格包络（指向 `RenderInput.geometries` 中的 mesh）。
    /// **该 mesh 的拓扑用于边界判定，不参与表面着色**。
    MeshEnvelope(MeshId),
}

impl VolumeBounds {
    pub fn validate(&self, path: &str) -> Result<(), DataError> {
        match self {
            VolumeBounds::Infinite => Ok(()),
            VolumeBounds::Aabb { min, max } => {
                if !min.iter().chain(max.iter()).all(|x| x.is_finite()) {
                    return Err(DataError::InvalidParam {
                        path: path.into(),
                        reason: "AABB 含非有限值".into(),
                    });
                }
                if min.iter().zip(max.iter()).any(|(a, b)| a > b) {
                    return Err(DataError::InvalidParam {
                        path: path.into(),
                        reason: "AABB min 必须 ≤ max".into(),
                    });
                }
                Ok(())
            }
            VolumeBounds::Sphere { center, radius } => {
                if !center.iter().chain(std::iter::once(radius)).all(|x| x.is_finite()) {
                    return Err(DataError::InvalidParam {
                        path: path.into(),
                        reason: "Sphere 含非有限值".into(),
                    });
                }
                if *radius <= 0.0 {
                    return Err(DataError::InvalidParam {
                        path: path.into(),
                        reason: "Sphere radius 必须 > 0".into(),
                    });
                }
                Ok(())
            }
            VolumeBounds::TransformedBox { transform, half_extents } => {
                if !transform.validate() {
                    return Err(DataError::InvalidParam {
                        path: path.into(),
                        reason: "Transform 含非有限值".into(),
                    });
                }
                if !half_extents.iter().all(|x| x.is_finite() && *x > 0.0) {
                    return Err(DataError::InvalidParam {
                        path: path.into(),
                        reason: "half_extents 必须有限且 > 0".into(),
                    });
                }
                Ok(())
            }
            // MeshEnvelope 的引用可达性在 `RenderInput::validate` 里检查
            VolumeBounds::MeshEnvelope(_) => Ok(()),
        }
    }
}
