//! 世界环境。背景光走 `LightType::Background`，体积走 `RenderInput.volumes`。
//! 此模块只承载环境级的扩展数据（HDRI 旋转、环境设置等）。

use serde::{Deserialize, Serialize};

use crate::common::meta::HasMeta;
use crate::common::{DataError, Meta};

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct WorldData {
    /// 动态拓展参数（HDRI 旋转、环境强度、世界级脚本配置等）。
    pub meta: Meta,
}

impl HasMeta for WorldData {
    fn meta(&self) -> &Meta { &self.meta }
    fn meta_mut(&mut self) -> &mut Meta { &mut self.meta }
}

impl WorldData {
    pub fn new() -> Self { WorldData::default() }

    pub fn validate(&self) -> Result<(), DataError> {
        self.meta.validate("world.meta")
    }
}
