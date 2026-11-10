//! 世界变换。列主序 4×4 浮点矩阵。

use serde::{Deserialize, Serialize};

/// **这是世界变换，已预乘**——层级、父子关系在上游（编辑器）已经消失。
/// 字段命名避免 `matrix` 之类的模糊词，直接说"世界变换"。
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(transparent)]
pub struct Transform(pub [[f32; 4]; 4]);

impl Transform {
    pub const IDENTITY: Transform = Transform([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    pub const fn from_cols(m: [[f32; 4]; 4]) -> Self {
        Transform(m)
    }

    pub fn validate(&self) -> bool {
        self.0.iter().all(|row| row.iter().all(|x| x.is_finite()))
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}
