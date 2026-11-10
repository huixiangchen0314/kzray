//! 分布式原生实体 ID：u64 的 newtype + 类型化引用 + 名字哈希。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::error::DataError;

/// 分布式原生实体 ID。u64，跨进程稳定，JNI 侧按位 reinterpret 成 `jlong`。
///
/// 为什么是 newtype 而不是裸 `u64`：防止"把材质 id 传给几何参数"变成合法代码。
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[serde(transparent)]
pub struct EntityId(pub u64);

impl EntityId {
    /// 从字符串确定性哈希到 u64。使用 FNV-1a 64 位。
    ///
    /// **这是纯函数**：同一字符串永远得到同一 id，跨进程、跨语言可复现。
    /// 碰撞由调用方通过 `IdMap` 检测，`from_name` 本身不做 I/O。
    pub fn from_name(name: &str) -> Self {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in name.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        EntityId(h)
    }
}

// ── 类型化引用：编译期防混淆，运行时就是 u64 ──

macro_rules! typed_ref {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
        #[serde(transparent)]
        pub struct $name(pub EntityId);
    };
}

typed_ref!(GeometryRef, "指向 `RenderInput.geometries` 的引用。");
typed_ref!(MaterialRef, "指向 `RenderInput.materials` 的引用。");
typed_ref!(TextureId, "指向 `RenderInput.fields` 中纹理类 Field 的引用。");
typed_ref!(FieldId, "指向 `RenderInput.fields` 中任意 Field 的引用（纹理/动画/曲线）。");
typed_ref!(LightRef, "指向 `RenderInput.lights` 的引用。");
typed_ref!(VolumeRef, "指向 `RenderInput.volumes` 的引用。");

/// 名字 → id 的注册表。用于加载器：从字符串名字分配稳定 id，碰撞时显式报错。
///
/// 不放在 `EntityId` 上是因为它**有状态**，状态属于使用方。
pub struct IdMap<V> {
    entries: BTreeMap<EntityId, (String, V)>,
}

impl<V> IdMap<V> {
    pub fn new() -> Self {
        IdMap { entries: BTreeMap::new() }
    }

    /// 从字符串名字分配 id。若该 id 已被不同名字占用，返回 `IdCollision`。
    /// 若该 id 已被同名占用，返回 `DuplicateId`（调用方通常应改用 `get`）。
    pub fn insert_named(&mut self, name: &str, value: V) -> Result<EntityId, DataError> {
        let id = EntityId::from_name(name);
        self.insert(id, name, value)?;
        Ok(id)
    }

    /// 用显式 id 插入。测试碰撞路径用，也用于上游已有稳定 id 的场景。
    pub fn insert(&mut self, id: EntityId, name: &str, value: V) -> Result<(), DataError> {
        if let Some((existing_name, _)) = self.entries.get(&id) {
            if existing_name != name {
                return Err(DataError::IdCollision {
                    new_name: name.to_string(),
                    existing_name: existing_name.clone(),
                    id,
                });
            }
            return Err(DataError::DuplicateId { id });
        }
        self.entries.insert(id, (name.to_string(), value));
        Ok(())
    }

    pub fn get(&self, id: EntityId) -> Option<&V> {
        self.entries.get(&id).map(|(_, v)| v)
    }

    pub fn name_of(&self, id: EntityId) -> Option<&str> {
        self.entries.get(&id).map(|(n, _)| n.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (EntityId, &str, &V)> {
        self.entries.iter().map(|(&id, (n, v))| (id, n.as_str(), v))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<V> Default for IdMap<V> {
    fn default() -> Self {
        Self::new()
    }
}
