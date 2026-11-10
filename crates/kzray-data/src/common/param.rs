//! 参数值。`structural` 数据层：参数名开放（`BTreeMap<String, _>`），参数值是封闭集合。
//!
//! **不是注册表**——参数值是"读出来的东西"，不是"注入的行为"。

use serde::{Deserialize, Serialize};

use super::id::EntityId;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum ParamValue {
    Float(f32),
    Int(i64),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Mat4([[f32; 4]; 4]),
    /// 颜色（线性 RGB）。语义上区别于 `Vec3`：不是方向或位置。
    Color([f32; 3]),
    String(String),
    /// 引用一个 Field（纹理 / 动画 / 曲线）。
    /// 消费方按参数语义解释它的输入输出：albedo 期望 `f(uv) -> color`，
    /// intensity 期望 `f(t) -> scalar`。
    Field(EntityId),
    Array(Vec<ParamValue>),
}

impl ParamValue {
    pub fn as_float(&self) -> Option<f32> {
        match self {
            ParamValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_vec3(&self) -> Option<[f32; 3]> {
        match self {
            ParamValue::Vec3(v) | ParamValue::Color(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            ParamValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// 校验：拒绝 NaN / Inf。渲染参数里这两个值几乎总意味着上游出错。
    pub fn validate(&self, path: &str) -> Result<(), super::error::DataError> {
        use super::error::DataError;
        let bad = |reason: &str| DataError::InvalidParam {
            path: path.to_string(),
            reason: reason.to_string(),
        };
        match self {
            ParamValue::Float(v) => {
                if !v.is_finite() {
                    return Err(bad("Float 不是有限值"));
                }
            }
            ParamValue::Vec2(v) if !v.iter().all(|x| x.is_finite()) => return Err(bad("Vec2 含非有限值")),
            ParamValue::Vec3(v) | ParamValue::Color(v) if !v.iter().all(|x| x.is_finite()) => {
                return Err(bad("Vec3/Color 含非有限值"))
            }
            ParamValue::Vec4(v) if !v.iter().all(|x| x.is_finite()) => return Err(bad("Vec4 含非有限值")),
            ParamValue::Mat4(m) => {
                if !m.iter().all(|row| row.iter().all(|x| x.is_finite())) {
                    return Err(bad("Mat4 含非有限值"));
                }
            }
            ParamValue::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    item.validate(&format!("{path}[{i}]"))?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}
