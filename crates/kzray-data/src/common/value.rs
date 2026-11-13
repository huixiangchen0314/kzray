//! 纯值。**不含引用、不含语义标注、数组只装纯值**。
//!
//! - 颜色不是独立类型：albedo 的 `[0.8, 0.2, 0.2]` 是 `Vec3`，语义由参数定义承担
//! - 引用不是值：`NodeOutput` / `FieldId` 在 `NodeInput` 里，不在 `Value` 里
//! - `Array(Vec<Value>)`：常量数组，元素是纯值，**不可能藏引用**
//! - `String` 保留：枚举型配置（`"srgb"` / `"ggx"` / `"box"`）需要

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::error::DataError;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum Value {
    Float(f32),
    Int(i64),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Mat4([[f32; 4]; 4]),
    String(String),
    /// **纯值数组**。元素是 `Value`，无引用。渲染器拿到的常量数组
    /// 永远是确定的值序列，不存在需要运行时解包的子结构。
    Array(Vec<Value>),
    /// **纯值 Map**：键是字符串，值是纯值。无引用、无函数。
    ///
    /// 用途：结构化参数传递（材质参数打包、节点配置等）。例如
    /// `{ albedo: [0.8, 0.2, 0.2], roughness: 0.2, model: "ggx" }`。
    ///
    /// 用 `BTreeMap` 而非 `HashMap`：
    /// - 序列化顺序确定（diff、测试、调试友好）；
    /// - `PartialEq` 不依赖 hash；
    /// - 无哈希随机性（可复现）。
    Map(BTreeMap<String, Value>),
}

impl Value {
    pub fn as_float(&self) -> Option<f32> {
        match self { Value::Float(v) => Some(*v), _ => None }
    }
    pub fn as_vec3(&self) -> Option<[f32; 3]> {
        match self { Value::Vec3(v) => Some(*v), _ => None }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self { Value::String(s) => Some(s), _ => None }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self { Value::Array(v) => Some(v), _ => None }
    }

    pub fn validate(&self, path: &str) -> Result<(), DataError> {
        let bad = |reason: &str| DataError::InvalidParam {
            path: path.to_string(),
            reason: reason.to_string(),
        };
        match self {
            Value::Float(v) if !v.is_finite() => return Err(bad("Float 不是有限值")),
            Value::Vec2(v) if !v.iter().all(|x| x.is_finite()) => {
                return Err(bad("Vec2 含非有限值"))
            }
            Value::Vec3(v) if !v.iter().all(|x| x.is_finite()) => {
                return Err(bad("Vec3 含非有限值"))
            }
            Value::Vec4(v) if !v.iter().all(|x| x.is_finite()) => {
                return Err(bad("Vec4 含非有限值"))
            }
            Value::Mat4(m) => {
                if !m.iter().all(|row| row.iter().all(|x| x.is_finite())) {
                    return Err(bad("Mat4 含非有限值"));
                }
            }
            Value::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    item.validate(&format!("{path}[{i}]"))?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}
