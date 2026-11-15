use glam::Vec3;
use kzray_data::MeshId;


pub struct MeshTriangle{
    pub id: MeshId,
}

pub struct MeshTriangleHit{
    /// 三角形索引
    pub tri_idx: u32,
    /// 重心坐标
    pub cerciron: Vec3, 
}

