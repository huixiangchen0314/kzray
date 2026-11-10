//! 分布式渲染清单：输入的分区视图。

use serde::{Deserialize, Serialize};

use crate::common::meta::HasMeta;
use crate::common::Meta;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct RenderManifest {
    pub job_id: u64,
    pub partitions: Vec<PartitionMeta>,
    /// 分区级粗 BVH（跨区路由用）。`None` = 不用顶层 BVH，靠 halo 复制。
    pub top_level_bvh: Option<Vec<u8>>,
    pub meta: Meta,
}

impl HasMeta for RenderManifest {
    fn meta(&self) -> &Meta { &self.meta }
    fn meta_mut(&mut self) -> &mut Meta { &mut self.meta }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct PartitionMeta {
    pub id: u32,
    /// 世界空间 AABB `[min, max]`。
    pub region: [[f32; 3]; 2],
    pub prim_count: u64,
    pub bvh_bytes: u64,
    /// 光子图内存预算字节数。`0` = 该分区无光子图。
    pub photon_bytes: u64,
}
