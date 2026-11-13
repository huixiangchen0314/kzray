//! 材质。**只有 BSDF 表达式**——不硬编码 displacement / normal。
//!
//! 具体有哪些输入（albedo、roughness、normal、置换高度...）由 BSDF 节点的
//! `kind` 和 `inputs` 决定。置换挂在 `RenderInstance` 上（同一 mesh 被不同
//! instance 用不同置换是合法的）；着色法线是 BSDF 节点的输入。
//!
//! # `bsdf` 字段：内联 `FieldExpr` 而非 `FieldId`
//!
//! 与早期版本的区别：`bsdf` 直接持有 [`FieldExpr`]（内联），不引用
//! [`crate::common::FieldId`]。理由：
//!
//! - **自包含**：材质不依赖 `RenderInput.fields` 表，构建层直接求值；
//! - **减少间接**：`FieldId → fields[id] → FieldExpr` 变成直接 `FieldExpr`；
//! - **扁平化**：与"扁平、世界空间、无层级"的设计目标一致。
//!
//! `FieldExpr::NodeOutput` 仍然引用 `nodes` 表——节点图的共享是必要的
//! （多个材质可能引用同一个节点），但 `fields` 表的间接层不需要。
//!
//! `RenderInput.fields` 表保留，用于 `Param::Field(FieldId)`——参数引用
//! 命名 Field 的场景（光源参数、相机参数、体积参数等）。
//!
//! # 校验
//!
//! `Material` 是纯数据，无 `validate` 方法。校验在
//! [`crate::input::RenderInput::validate`] 中完成：
//!
//! - `FieldExpr::Constant(Value)`：`Value::validate` 通过；
//! - `FieldExpr::Bitmap(_)`：纹理资源引用，数据层不检查内容；
//! - `FieldExpr::NodeOutput(no)`：`no.node` 必须存在于 `nodes` 表。

use serde::{Deserialize, Serialize};

use crate::common::FieldExpr;

/// 材质数据。
///
/// 只承载 BSDF 表达式。构建层求值成运行时 `BSDFBuilder`。
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Material {
    /// BSDF 表达式。构建层求值成运行时 `BSDFBuilder` 实现。
    ///
    /// 内联 `FieldExpr` 而非持有 `FieldId`：材质是**自包含**的，不依赖
    /// `RenderInput.fields` 表。`FieldExpr::NodeOutput` 仍然引用 `nodes`
    /// 表（节点图的共享是必要的），但 `fields` 表的间接层不需要。
    ///
    /// 执行层要求该 FieldExpr 的根输出 kind 属于 BSDF 类（由节点图的
    /// 节点类型约定保证，数据层不做类型检查）。
    pub bsdf: FieldExpr,
}
