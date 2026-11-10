//! 材质。**只有 BSDF Field 引用**——不硬编码 displacement / normal。
//!
//! 具体有哪些输入（albedo、roughness、normal、置换高度...）由 BSDF 节点的
//! `kind` 和 `inputs` 决定。置换挂在 `RenderInstance` 上（同一 mesh 被不同
//! instance 用不同置换是合法的）；着色法线是 BSDF 节点的输入。

use serde::{Deserialize, Serialize};

use crate::common::FieldId;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Material {
    /// BSDF 的 Field。执行层要求该 Field 的 `FieldExpr` 是 `NodeOutput`，
    /// 且根节点端口输出的 kind 属于 BSDF 类。
    pub bsdf: FieldId,
}
