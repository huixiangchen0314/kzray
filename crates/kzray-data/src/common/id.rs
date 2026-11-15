//! 分布式原生实体 ID：u64 的 newtype + 类型化引用 + 名字哈希。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::error::DataError;

/// 分布式原生实体 ID。u64，跨进程稳定，JNI 侧按位 reinterpret 成 `jlong`。
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[serde(transparent)]
pub struct EntityId(pub u64);

impl EntityId {
    /// 从字符串确定性哈希到 u64（FNV-1a 64 位）。
    pub fn from_name(name: &str) -> Self {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in name.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        EntityId(h)
    }

    /// 取原始 u64 位模式（JNI / FFI / 序列化场景直接使用）。
    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

macro_rules! typed_ref {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(
            Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug,
        )]
        #[serde(transparent)]
        pub struct $name(pub EntityId);

        impl $name {
            /// 取原始 u64 位模式（JNI / FFI / 序列化场景直接使用）。
            #[inline]
            pub fn as_u64(self) -> u64 {
                self.0.as_u64()
            }
        }

        impl From<EntityId> for $name {
            fn from(id: EntityId) -> Self {
                $name(id)
            }
        }

        impl From<$name> for EntityId {
            fn from(t: $name) -> Self {
                t.0
            }
        }

        impl From<u64> for $name {
            fn from(v: u64) -> Self {
                $name(EntityId(v))
            }
        }

        impl From<$name> for u64 {
            fn from(t: $name) -> Self {
                (t.0).0
            }
        }
    };
}

typed_ref!(InstanceId, "渲染实例的唯一标识");
typed_ref!(SphereId, "球数据Id");
typed_ref!(PlaneId, "平面数据Id");
typed_ref!(MeshId, "指向 `RenderInput.meshes` 的 key。");
typed_ref!(MaterialId, "指向 `RenderInput.materials` 的 key。");
typed_ref!(LightId, "指向 `RenderInput.lights` 的 key。");
typed_ref!(FieldId, "指向 `RenderInput.fields` 的 key。");
typed_ref!(NodeId, "指向 `RenderInput.nodes` 的 key。");
typed_ref!(VolumeId, "指向 `RenderInput.volumes` 的 key。");

// 非表引用：外部资源 id，不指向 `RenderInput` 中的任何表
typed_ref!(
    TextureId,
    "外部 bitmap 纹理资源 id（磁盘文件 / 外部 asset）。"
);

/// 名字 → id 的注册表。加载器用，碰撞显式报错。
pub struct IdMap<V> {
    entries: BTreeMap<EntityId, (String, V)>,
}

impl<V> IdMap<V> {
    pub fn new() -> Self {
        IdMap {
            entries: BTreeMap::new(),
        }
    }

    pub fn insert_named(&mut self, name: &str, value: V) -> Result<EntityId, DataError> {
        let id = EntityId::from_name(name);
        self.insert(id, name, value)?;
        Ok(id)
    }

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

    pub fn into_typed<K>(self) -> BTreeMap<K, V>
    where
        K: From<EntityId> + Ord,
    {
        self.entries
            .into_iter()
            .map(|(id, (_, v))| (K::from(id), v))
            .collect()
    }
}

impl<V> Default for IdMap<V> {
    fn default() -> Self {
        Self::new()
    }
}
