//! Field：**场**——一个可被引用的函数/值。命名实体，本体在 `RenderInput.fields`。
//!
//! 节点图是一种场的表达：节点的某个端口产生的值/函数，
//! 通过 `FieldExpr::NodeOutput { node, port }` 命名成一个 Field。
//!
//! **场的组合在图里表达，不在数据层递归**——`FieldExpr` 三态封闭，
//! 不存在"一个 Field 里嵌套另一个 FieldExpr"的结构。

use serde::{Deserialize, Serialize};

use super::id::TextureId;
use super::node::NodeOutput;
use super::value::Value;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum FieldExpr {
    /// 常量场：输入无关，输出固定。用 `Value` 而非 `NodeInput`——
    /// **常量场里不可能有引用**，形状上就不可能。
    Constant(Value),

    /// 直接引用外部 bitmap：`f(uv) -> color`。
    Bitmap(TextureId),

    /// **节点图的某个输出端口**就是这个场。场的身份是 `(node, port)`。
    NodeOutput(NodeOutput),
}

pub use super::id::FieldId;
