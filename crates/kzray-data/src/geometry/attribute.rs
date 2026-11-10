//! 属性缓冲。SoA 扁平存储，按 `stride` 分组。

use serde::{Deserialize, Serialize};

/// 属性值。i/f/b 三种数值类型，加无符号整数变体（索引/权重常用）。
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum AttributeValues {
    F32(Vec<f32>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    Bool(Vec<bool>),
}

impl AttributeValues {
    pub fn len(&self) -> usize {
        match self {
            AttributeValues::F32(v) => v.len(),
            AttributeValues::I32(v) => v.len(),
            AttributeValues::U32(v) => v.len(),
            AttributeValues::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 属性缓冲。`values.len()` 必须能被 `stride` 整除。
///
/// **`stride` 是元素分量数**（CG 惯例）：`vec3` = 3，`mat3` = 9，标量 = 1。
/// 不加 `1..=4` 的约束——矩阵属性、打包属性都可能 > 4。
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct AttributeBuffer {
    pub values: AttributeValues,
    /// components per element。允许任意正整数。
    pub stride: u32,
}

impl AttributeBuffer {
    pub fn new(values: AttributeValues, stride: u32) -> Self {
        debug_assert!(stride > 0, "stride 必须 > 0");
        debug_assert_eq!(
            values.len() % stride as usize,
            0,
            "values.len() ({}) 不是 stride ({}) 的整数倍",
            values.len(),
            stride
        );
        AttributeBuffer { values, stride }
    }

    pub fn f32(stride: u32, data: Vec<f32>) -> Self {
        Self::new(AttributeValues::F32(data), stride)
    }

    pub fn i32(stride: u32, data: Vec<i32>) -> Self {
        Self::new(AttributeValues::I32(data), stride)
    }

    pub fn u32(stride: u32, data: Vec<u32>) -> Self {
        Self::new(AttributeValues::U32(data), stride)
    }

    pub fn bool(stride: u32, data: Vec<bool>) -> Self {
        Self::new(AttributeValues::Bool(data), stride)
    }

    /// 元素数量 = `values.len() / stride`。
    /// 反序列化不经过 `new`，因此 stride=0 必须防御性返回 0。
    pub fn count(&self) -> usize {
        if self.stride == 0 {
            0
        } else {
            self.values.len() / self.stride as usize
        }
    }

    /// 取第 `i` 个元素的 f32 切片。类型不匹配或越界返回 `None`。
    pub fn get_f32(&self, i: usize) -> Option<&[f32]> {
        let AttributeValues::F32(v) = &self.values else {
            return None;
        };
        let s = self.stride as usize;
        if s == 0 || i >= self.count() {
            return None;
        }
        Some(&v[i * s..(i + 1) * s])
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.values {
            AttributeValues::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<&[u32]> {
        match &self.values {
            AttributeValues::U32(v) => Some(v),
            _ => None,
        }
    }
}
