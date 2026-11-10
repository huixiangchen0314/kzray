//! 参数。一个参数值是**常量值**或**对一个 Field 的引用**。
//!
//! 这是"参数携带"的统一抽象：任何字段，只要是参数，就是 `Param`。
//! 灯光/相机的物理参数、节点图的常量槽、材质的可调参数——全部走这一个类型。

use serde::{Deserialize, Serialize};

use super::error::DataError;
use super::id::FieldId;
use super::value::Value;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum Param {
    /// 常量值。
    Value(Value),
    /// 引用一个命名 Field，消费方按参数语义解释其输入输出。
    Field(FieldId),
}

impl Param {
    pub fn from_f32(v: f32) -> Self { Param::Value(Value::Float(v)) }
    pub fn from_i64(v: i64) -> Self { Param::Value(Value::Int(v)) }
    pub fn from_bool(v: bool) -> Self { Param::Value(Value::Bool(v)) }
    pub fn from_vec3(v: [f32; 3]) -> Self { Param::Value(Value::Vec3(v)) }
    pub fn from_str(s: impl Into<String>) -> Self { Param::Value(Value::String(s.into())) }

    pub fn as_value(&self) -> Option<&Value> {
        match self { Param::Value(v) => Some(v), Param::Field(_) => None }
    }
    pub fn as_field(&self) -> Option<FieldId> {
        match self { Param::Field(id) => Some(*id), Param::Value(_) => None }
    }

    pub fn validate(&self, path: &str) -> Result<(), DataError> {
        match self {
            Param::Value(v) => v.validate(path),
            Param::Field(_) => Ok(()),
        }
    }
}
