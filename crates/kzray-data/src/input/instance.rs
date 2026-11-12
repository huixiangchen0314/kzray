//! 可渲染实例。**世界变换已预乘**，无层级、无父子。

use serde::{Deserialize, Serialize};

use crate::common::meta::HasMeta;
use crate::common::{FieldId, MeshId, MaterialId, Meta, Transform};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub struct VisibilityFlags {
    pub camera: bool,
    pub shadow: bool,
    pub diffuse: bool,
    pub glossy: bool,
    pub transmission: bool,
}

impl Default for VisibilityFlags {
    fn default() -> Self {
        VisibilityFlags {
            camera: true,
            shadow: true,
            diffuse: true,
            glossy: true,
            transmission: true,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct RenderInstance {
    pub geometry: MeshId,
    pub material_refs: Vec<MaterialId>,
    pub world_transform: Transform,
    /// 置换：`f(uv) -> height`。挂实例级（同一 mesh 不同实例可不同置换）。
    pub displacement: Option<FieldId>,
    pub visibility: VisibilityFlags,
    pub meta: Meta,
}

impl HasMeta for RenderInstance {
    fn meta(&self) -> &Meta { &self.meta }
    fn meta_mut(&mut self) -> &mut Meta { &mut self.meta }
}
