//! 节点图的节点。DAG 的顶点。**节点不是场——是多输出端口的计算器**。
//!
//! 节点图的输入是**扁平的**：`BTreeMap<String, NodeInput>`，没有嵌套结构。
//! 数组输入、嵌套图这类"需要渲染器运行时展开"的结构不在协议里——
//! 上游要么展开成独立输入槽（`layer_0`, `layer_1`, ...），要么拉进节点图
//! 让数组结构本身成为图节点。渲染器看到的永远是确定的一层引用。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::error::DataError;
use super::id::{FieldId, NodeId};
use super::value::Value;

/// DAG 边的一端点：`(节点, 端口名)`。
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
pub struct NodeOutput {
    pub node: NodeId,
    pub port: String,
}

impl NodeOutput {
    pub fn new(node: NodeId, port: impl Into<String>) -> Self {
        NodeOutput { node, port: port.into() }
    }

    pub fn single(node: NodeId) -> Self {
        NodeOutput { node, port: "out".to_string() }
    }
}

/// 节点的一个输入槽。**三态封闭**：
///
/// - `Const(Value)`：常量参数值
/// - `Node(NodeOutput)`：引用同 DAG 内某节点某端口
/// - `Field(FieldId)`：引用命名 Field（跨实体共享）
///
/// **无数组变体**。需要"数组输入"的 kind 由上游展开成独立槽位或拉进图。
/// 这保证节点图的输入结构是静态扁平的——引用深度有界，不存在运行时解包。
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum NodeInput {
    Const(Value),
    Node(NodeOutput),
    Field(FieldId),
}

impl NodeInput {
    pub fn const_f32(v: f32) -> Self { NodeInput::Const(Value::Float(v)) }
    pub fn const_vec3(v: [f32; 3]) -> Self { NodeInput::Const(Value::Vec3(v)) }
    pub fn const_str(s: impl Into<String>) -> Self { NodeInput::Const(Value::String(s.into())) }
    pub fn const_bool(v: bool) -> Self { NodeInput::Const(Value::Bool(v)) }

    pub fn validate(&self, path: &str) -> Result<(), DataError> {
        match self {
            NodeInput::Const(v) => v.validate(path),
            NodeInput::Node(_) | NodeInput::Field(_) => Ok(()),
        }
    }
}

/// DAG 的一个节点。
///
/// **不声明输出端口**——端口集由 `kind` 静态决定，执行层注册表持有。
/// 类型安全分两层：
/// 1. **结构层（编译期）**：`NodeInput` 三态封闭，值 vs 引用互斥
/// 2. **参数层（运行期）**：每个 kind 的参数 schema（如 `lambertian` 的
///    `reflectance: Vec3`）由注册表校验，不进数据协议
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Node {
    pub kind: String,
    pub inputs: BTreeMap<String, NodeInput>,
}

impl Node {
    pub fn new(kind: impl Into<String>) -> Self {
        Node {
            kind: kind.into(),
            inputs: BTreeMap::new(),
        }
    }

    pub fn with_input(mut self, name: impl Into<String>, value: NodeInput) -> Self {
        self.inputs.insert(name.into(), value);
        self
    }

    /// 上游端口引用。**直接扫描一层，不递归**——`NodeInput` 没有嵌套。
    pub fn upstream_outputs(&self) -> Vec<NodeOutput> {
        self.inputs
            .values()
            .filter_map(|i| match i {
                NodeInput::Node(no) => Some(no.clone()),
                _ => None,
            })
            .collect()
    }

    /// 上游节点 id。
    pub fn upstream_nodes(&self) -> Vec<NodeId> {
        self.upstream_outputs().into_iter().map(|o| o.node).collect()
    }

    /// 跨 Field 引用的上游 Field 列表。
    pub fn upstream_fields(&self) -> Vec<FieldId> {
        self.inputs
            .values()
            .filter_map(|i| match i {
                NodeInput::Field(id) => Some(*id),
                _ => None,
            })
            .collect()
    }
}
