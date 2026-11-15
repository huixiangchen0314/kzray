//! kzray-impl/src/accel/resource.rs

use std::collections::BTreeMap;

use kzray_data::{Mesh, MeshId, TextureId};

/// 场景加速资源表（**编译产物，只读**）。
///
/// 只持有**必须在实例间共享的大块数据**——目前仅 mesh。mesh 的顶点/三角形
/// 数据量大（KB–MB 级），不能内联进每个实例，必须按 id 共享。
///
/// 解析形状（球、平面）不在此表：它们是值语义的"公式"，已内联进
/// [`crate::accel::instance::ShapeInstance`]，求交时零查表。表示层的
/// `scene::ResourceTable` 另持有标量 `Sphere`/`Plane` 供编辑与序列化，
/// 编译阶段投影为内联的 `AnyShapeA`。
///
/// 本表不提供插入、删除、可变访问——构建后只读。
#[derive(Debug)]
pub struct ResourceTable {
    /// 网格表。mesh 无 SIMD 镜像，直接持有。
    pub meshes: BTreeMap<MeshId, Mesh>,
}

impl ResourceTable {
    /// 由网格表构造。通常由编译阶段 `collect()` 得到。
    pub fn new(meshes: BTreeMap<MeshId, Mesh>,
        _textures: BTreeMap<TextureId, f32>) -> Self {
        Self { meshes }
    }

    /// 网格表为空。
    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }

    /// 查网格。
    pub fn get_mesh(&self, id: MeshId) -> Option<&Mesh> {
        self.meshes.get(&id)
    }

    /// 查网格，不存在则 panic。用于编译期已校验 id 的热点路径
    /// （BVH 实例引用的 mesh 必然在表里）。
    #[track_caller]
    pub fn expect_mesh(&self, id: MeshId) -> &Mesh {
        self.meshes
            .get(&id)
            .unwrap_or_else(|| panic!("mesh {id:?} not found in ResourceTable"))
    }
}
