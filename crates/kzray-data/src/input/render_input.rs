//! 渲染器一次完整输入。扁平、世界空间、无层级。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::camera::CameraData;
use crate::common::{
    DataError, FieldExpr, FieldId, MeshId, LightId, MaterialId, Node, NodeId, 
    Param, VolumeId,
};
use crate::config::RenderConfig;
use crate::geometry::Mesh;
use crate::light::LightData;
use crate::shading::Material;
use crate::volume::VolumeInstance;
use crate::world::WorldData;

use super::instance::RenderInstance;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct RenderInput {
    /// 场
    pub fields: BTreeMap<FieldId, FieldExpr>,
    /// 节点图根节点
    pub nodes: BTreeMap<NodeId, Node>,
    /// 材质
    pub materials: BTreeMap<MaterialId, Material>,
    /// 网格
    pub meshes: BTreeMap<MeshId, Mesh>,
    /// 渲染实例
    pub instances: Vec<RenderInstance>,
    /// 灯光
    pub lights: BTreeMap<LightId, LightData>,
    /// 体积
    pub volumes: BTreeMap<VolumeId, VolumeInstance>,
    /// 相机配置
    pub camera: CameraData,
    /// 世界配置
    pub world: WorldData,
    /// 渲染配置
    pub render: RenderConfig,
}

impl RenderInput {
        pub fn new(camera: CameraData, render: RenderConfig) -> Self {
        RenderInput {
            fields: BTreeMap::new(),
            nodes: BTreeMap::new(),
            materials: BTreeMap::new(),
            meshes: BTreeMap::new(),
            instances: Vec::new(),
            lights: BTreeMap::new(),
            volumes: BTreeMap::new(),
            camera,
            world: WorldData::new(),
            render,
        }
    }

    pub fn validate(&self) -> Result<(), DataError> {
        // 1. 数据本身
        for (id, geom) in &self.meshes {
            geom.validate().map_err(|e| DataError::InvalidParam {
                path: format!("geometry[{id:?}]"),
                reason: e,
            })?;
        }
        for (id, light) in &self.lights {
            light.validate().map_err(|e| DataError::InvalidParam {
                path: format!("light[{id:?}]"),
                reason: format!("{e}"),
            })?;
        }
        for (id, vol) in &self.volumes {
            vol.validate(&format!("volume[{id:?}]"))?;
        }
        self.camera.validate()?;
        self.world.validate()?;

        // 2. 节点输入合法性
        for (id, node) in &self.nodes {
            for (k, v) in &node.inputs {
                v.validate(&format!("node[{id:?}].{k}"))?;
            }
        }

        // 3. Field 的节点引用可达
        for (id, expr) in &self.fields {
            if let FieldExpr::NodeOutput(no) = expr {
                if !self.nodes.contains_key(&no.node) {
                    return Err(DataError::InvalidParam {
                        path: format!("field[{id:?}]"),
                        reason: format!("引用的节点 {:?} 不存在", no.node),
                    });
                }
            }
        }

        // 4. 所有 Param::Field 引用的 FieldId 可达
        for (id, light) in &self.lights {
            self.check_light_field_refs(light, id)?;
        }
        self.check_camera_field_refs()?;
        for (i, inst) in self.instances.iter().enumerate() {
            if let Some(f) = inst.displacement {
                if !self.fields.contains_key(&f) {
                    return Err(DataError::InvalidParam {
                        path: format!("instance[{i}].displacement"),
                        reason: format!("引用的 Field {:?} 不存在", f),
                    });
                }
            }
        }
        // 体积参数中的 Field 引用
        for (id, vol) in &self.volumes {
            for (k, p) in &vol.volume.params {
                if let Param::Field(f) = p {
                    if !self.fields.contains_key(f) {
                        return Err(DataError::InvalidParam {
                            path: format!("volume[{id:?}].params.{k}"),
                            reason: format!("引用的 Field {f:?} 不存在"),
                        });
                    }
                }
            }
        }
        // 世界 meta 中的 Field 引用
        for (k, p) in self.world.meta.iter() {
            if let Param::Field(f) = p {
                if !self.fields.contains_key(f) {
                    return Err(DataError::InvalidParam {
                        path: format!("world.meta.{k}"),
                        reason: format!("引用的 Field {f:?} 不存在"),
                    });
                }
            }
        }

        // 5. 实例的 geometry / material 引用可达
        for (i, inst) in self.instances.iter().enumerate() {
            if !self.meshes.contains_key(&inst.geometry) {
                return Err(DataError::InvalidParam {
                    path: format!("instance[{i}].geometry"),
                    reason: format!("引用的几何 {:?} 不存在", inst.geometry),
                });
            }
            for (j, mat_ref) in inst.material_refs.iter().enumerate() {
                if !self.materials.contains_key(mat_ref) {
                    return Err(DataError::InvalidParam {
                        path: format!("instance[{i}].material_refs[{j}]"),
                        reason: format!("引用的材质 {:?} 不存在", mat_ref),
                    });
                }
            }
        }

        // 6. 体积的 MeshEnvelope 引用可达
        for (id, vol) in &self.volumes {
            if let crate::volume::VolumeBounds::MeshEnvelope(g) = &vol.bounds {
                if !self.meshes.contains_key(g) {
                    return Err(DataError::InvalidParam {
                        path: format!("volume[{id:?}].bounds"),
                        reason: format!("引用的几何 {g:?} 不存在"),
                    });
                }
            }
        }

        // 7. 材质的 BSDF Field 可达
        for (id, mat) in &self.materials {
            if !self.fields.contains_key(&mat.bsdf) {
                return Err(DataError::InvalidParam {
                    path: format!("material[{id:?}].bsdf"),
                    reason: format!("引用的 Field {:?} 不存在", mat.bsdf),
                });
            }
        }

        // 8. 节点图无环
        self.check_node_graph_acyclic()?;

        Ok(())
    }

    fn check_light_field_refs(&self, light: &LightData, id: &LightId) -> Result<(), DataError> {
        let mut ids = Vec::new();
        collect_light_param_fields(light, &mut ids);
        for f in ids {
            if !self.fields.contains_key(&f) {
                return Err(DataError::InvalidParam {
                    path: format!("light[{id:?}]"),
                    reason: format!("引用的 Field {f:?} 不存在"),
                });
            }
        }
        Ok(())
    }

    fn check_camera_field_refs(&self) -> Result<(), DataError> {
        let mut ids = Vec::new();
        collect_camera_param_fields(&self.camera, &mut ids);
        for f in ids {
            if !self.fields.contains_key(&f) {
                return Err(DataError::InvalidParam {
                    path: "camera".into(),
                    reason: format!("引用的 Field {f:?} 不存在"),
                });
            }
        }
        Ok(())
    }

    fn check_node_graph_acyclic(&self) -> Result<(), DataError> {
        // 从所有 Field 的根 NodeOutput 出发 DFS，检测环
        let mut visiting = std::collections::HashSet::new();
        let mut done = std::collections::HashSet::new();

        for expr in self.fields.values() {
            if let FieldExpr::NodeOutput(no) = expr {
                self.dfs_node(no.node, &mut visiting, &mut done)?;
            }
        }
        Ok(())
    }

    fn dfs_node(
        &self,
        n: NodeId,
        visiting: &mut std::collections::HashSet<NodeId>,
        done: &mut std::collections::HashSet<NodeId>,
    ) -> Result<(), DataError> {
        if done.contains(&n) {
            return Ok(());
        }
        if visiting.contains(&n) {
            return Err(DataError::NodeGraphCycle(n.0));
        }
        visiting.insert(n);
        if let Some(node) = self.nodes.get(&n) {
            for up in node.upstream_nodes() {
                self.dfs_node(up, visiting, done)?;
            }
        }
        visiting.remove(&n);
        done.insert(n);
        Ok(())
    }
}

fn collect_light_param_fields(light: &LightData, out: &mut Vec<FieldId>) {
    use crate::common::Param;
    use crate::light::{
        AreaLight, AreaShape, BackgroundLight, LightType, PointLight, SpotLight, SunLight,
    };
    fn p(param: &Param, out: &mut Vec<FieldId>) {
        if let Param::Field(f) = param { out.push(*f); }
    }
    match &light.kind {
        LightType::Point(PointLight { intensity, color, .. }) => {
            p(intensity, out); p(color, out);
        }
        LightType::Spot(SpotLight { cone_angle, penumbra, intensity, color, .. }) => {
            p(cone_angle, out); p(penumbra, out); p(intensity, out); p(color, out);
        }
        LightType::Area(AreaLight { shape, intensity, color, .. }) => {
            match shape {
                AreaShape::Rectangle { size_u, size_v } => { p(size_u, out); p(size_v, out); }
                AreaShape::Disk { radius } | AreaShape::Sphere { radius } => p(radius, out),
            }
            p(intensity, out); p(color, out);
        }
        LightType::Sun(SunLight { angle, intensity, color, .. }) => {
            p(angle, out); p(intensity, out); p(color, out);
        }
        LightType::Background(BackgroundLight { color, intensity }) => {
            p(color, out); p(intensity, out);
        }
    }
}

fn collect_camera_param_fields(camera: &CameraData, out: &mut Vec<FieldId>) {
    use crate::camera::CameraProjection;
    use crate::common::Param;
    fn p(param: &Param, out: &mut Vec<FieldId>) {
        if let Param::Field(f) = param { out.push(*f); }
    }
    match &camera.projection {
        CameraProjection::Perspective(c) => {
            p(&c.fov_y, out); p(&c.aperture, out); p(&c.focus_distance, out);
        }
        CameraProjection::Orthographic(c) => p(&c.ortho_scale, out),
        CameraProjection::Panorama(c) => p(&c.fov_y, out),
    }
}
