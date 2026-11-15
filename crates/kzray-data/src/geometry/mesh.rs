//! 三角网格。**上游已完成三角化、顶点分裂/焊接、mikk 处理。**
//!
//! **没有 Vertex/Corner 之分**：分裂焊接后 UV、法线、切线都在同一个顶点索引
//! 空间里。这是渲染器的数据，不是编辑器的网格——不存在"一个网格同时有 vertex
//! 和 corner 两种索引空间"这回事。
//!
//! 网格的索引空间按**结构**分开存放：
//! - `vertex_attributes`：索引 = 顶点索引（position / normal / uv / tangent / color ...）
//! - `edge_attributes`：索引 = 边索引（edge highlight / crease weight ...）
//! - `face_attributes`：索引 = 三角形索引（face highlight / face id ...）
//!
//! 三者互相独立，各自有各自的元素数量约束。

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

use crate::geometry::attribute::AttributeBuffer;


/// 子网格：连续三角形共享一个材质。三角形按子网格顺序排序。
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub struct SubMesh {
    /// 三角形区间 `[tri_start, tri_end)`。
    pub tri_start: u32,
    pub tri_end: u32,
    pub material_index: u32,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct Mesh {
    /// 每顶点属性。约定名：`"position"`(stride=3) / `"normal"`(stride=3) /
    /// `"uv"`(stride=2) / `"tangent"`(stride=4) / `"color"`(stride=3 或 4)。
    /// 元素数量 = 顶点数（由 `indices` 的最大值 + 1 界定，或由 position 数量决定）。
    pub vertex_attributes: BTreeMap<String, AttributeBuffer>,

    /// 三角形索引。必须存在。每个三角形 3 个顶点索引。
    pub indices: Vec<[u32; 3]>,

    /// 显式边拓扑（去重后的顶点对）。**空 = 不需要边级数据**。
    pub edges: Vec<[u32; 2]>,

    /// 每边属性。元素数量 = `edges.len()`。边域属性。
    pub edge_attributes: BTreeMap<String, AttributeBuffer>,

    /// 每面属性。元素数量 = `indices.len()`。面域属性。
    pub face_attributes: BTreeMap<String, AttributeBuffer>,

    /// 子网格。空 = 全三角形用材质 0。非空必须完整覆盖、不重叠、按起点升序。
    pub sub_meshes: Vec<SubMesh>,
}

impl Mesh {
    pub fn triangle_count(&self) -> usize { self.indices.len() }
    pub fn edge_count(&self) -> usize { self.edges.len() }

    /// 顶点数：优先用 position 的 count，回退到 max index + 1。
    pub fn vertex_count(&self) -> usize {
        if let Some(p) = self.vertex_attributes.get("position") {
            return p.count();
        }
        self.indices.iter().flatten().copied().max().map(|m| m as usize + 1).unwrap_or(0)
    }

    pub fn vertex_attribute(&self, name: &str) -> Option<&AttributeBuffer> {
        self.vertex_attributes.get(name)
    }
    pub fn edge_attribute(&self, name: &str) -> Option<&AttributeBuffer> {
        self.edge_attributes.get(name)
    }
    pub fn face_attribute(&self, name: &str) -> Option<&AttributeBuffer> {
        self.face_attributes.get(name)
    }

    pub fn material_index(&self, tri: u32) -> u32 {
        if self.sub_meshes.is_empty() { return 0; }
        for s in &self.sub_meshes {
            if tri >= s.tri_start && tri < s.tri_end { return s.material_index; }
        }
        0
    }

    /// 一致性校验。loader 调用，构造时不做。
    pub fn validate(&self) -> Result<(), String> {
        let vc = self.vertex_count();
        let ec = self.edge_count();
        let tc = self.triangle_count();

        // 顶点域
        for (name, buf) in &self.vertex_attributes {
            if buf.count() != vc {
                return Err(format!(
                    "顶点属性 `{name}` 有 {} 个元素，但顶点数 {vc}", buf.count()
                ));
            }
        }
        // 边域
        for (name, buf) in &self.edge_attributes {
            if buf.count() != ec {
                return Err(format!(
                    "边属性 `{name}` 有 {} 个元素，但边数 {ec}", buf.count()
                ));
            }
        }
        // 面域
        for (name, buf) in &self.face_attributes {
            if buf.count() != tc {
                return Err(format!(
                    "面属性 `{name}` 有 {} 个元素，但三角形数 {tc}", buf.count()
                ));
            }
        }
        // 索引不越界
        for (tri, idx) in self.indices.iter().enumerate() {
            for &i in idx {
                if (i as usize) >= vc {
                    return Err(format!("三角形 {tri} 索引 {i} 越界（顶点数 {vc}）"));
                }
            }
        }
        // 边索引不越界
        for (ei, edge) in self.edges.iter().enumerate() {
            for &i in edge {
                if (i as usize) >= vc {
                    return Err(format!("边 {ei} 顶点索引 {i} 越界（顶点数 {vc}）"));
                }
            }
        }
        // sub_meshes 覆盖
        if !self.sub_meshes.is_empty() {
            let mut expected = 0u32;
            for s in &self.sub_meshes {
                if s.tri_start != expected {
                    return Err(format!("sub_meshes 不连续：期望 {expected}，实际 {}", s.tri_start));
                }
                if s.tri_end < s.tri_start {
                    return Err(format!("sub_mesh 终点 {} < 起点 {}", s.tri_end, s.tri_start));
                }
                expected = s.tri_end;
            }
            if expected as usize != tc {
                return Err(format!("sub_meshes 覆盖到 {expected}，三角形数 {tc}"));
            }
        }
        Ok(())
    }
}
