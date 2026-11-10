//! 动态拓展数据的统一载体。
//!
//! 核心集封闭（`LightType` 是 enum，`CameraProjection` 是 enum），但**任何
//! 实体都可以携带额外的扩展参数**，类似 Clojure 的 `meta` / `with-meta`。
//!
//! 典型用途：用户自定义属性、上游工具链的标注、脚本/滤镜按名字读取的槽位。
//! 核心执行层不知道也不关心这些 key——它只保证 meta 原样传递和序列化。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::param::Param;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
#[serde(transparent)]
pub struct Meta(pub BTreeMap<String, Param>);

impl Meta {
    pub fn new() -> Self {
        Meta(BTreeMap::new())
    }

    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    pub fn len(&self) -> usize { self.0.len() }

    pub fn get(&self, key: &str) -> Option<&Param> {
        self.0.get(key)
    }

    pub fn insert(&mut self, key: impl Into<String>, value: Param) -> Option<Param> {
        self.0.insert(key.into(), value)
    }

    pub fn remove(&mut self, key: &str) -> Option<Param> {
        self.0.remove(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Param)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn validate(&self, path: &str) -> Result<(), super::error::DataError> {
        for (k, v) in &self.0 {
            v.validate(&format!("{path}.{k}"))?;
        }
        Ok(())
    }
}

/// 携带动态拓展数据的能力。
///
/// **核心集封闭 + 允许附加 meta**——这是 Clojure `with-meta` 的 Rust 落地。
/// 实现方提供对自身 `Meta` 字段的访问，消费方可以通过 trait 统一处理，
/// 不必知道具体类型是 `LightData` 还是 `CameraData`。
pub trait HasMeta {
    fn meta(&self) -> &Meta;
    fn meta_mut(&mut self) -> &mut Meta;

    /// Clojure 风格的 `with-meta`：返回附带了 meta 的新值。
    fn with_meta(mut self, meta: Meta) -> Self
    where
        Self: Sized,
    {
        *self.meta_mut() = meta;
        self
    }
}
