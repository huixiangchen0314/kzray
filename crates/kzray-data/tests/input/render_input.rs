//! 对应 `src/input/render_input.rs`。
//!
//! 覆盖：形状往返、引用可达性校验、DAG 无环、体积独立表。

use std::collections::BTreeMap;

use kzray_data::camera::{CameraData, CameraProjection, PerspectiveCamera};
use kzray_data::common::{
    EntityId, FieldExpr, FieldId, GeometryRef, HasMeta, LightRef, MaterialRef, Meta, Node, NodeId,
    NodeInput, NodeOutput, Param, Transform, VolumeRef,
};
use kzray_data::config::{
    ChannelDataType, ChannelSampleMode, FilmConfig, FilterType, IntegratorConfig,
    RenderChannelDesc, RenderConfig, SamplerConfig,
};
use kzray_data::geometry::{AttributeBuffer, Mesh};
use kzray_data::input::{RenderInput, RenderInstance, VisibilityFlags};
use kzray_data::light::{LightData, LightType, PointLight};
use kzray_data::shading::Material;
use kzray_data::volume::{VolumeBounds, VolumeData, VolumeInstance};
use kzray_data::world::WorldData;
use kzray_data::DataError;

// ── 辅助构造 ──

fn minimal_render_config() -> RenderConfig {
    RenderConfig {
        width: 64,
        height: 64,
        samples: 4,
        sampler: SamplerConfig::default(),
        integrator: IntegratorConfig::default(),
        channels: vec![RenderChannelDesc {
            name: "beauty".into(),
            data_type: ChannelDataType::Color,
            sample_mode: ChannelSampleMode::PerSample,
        }],
        film: FilmConfig {
            filter: FilterType::Box,
            filter_width: 1.0,
            exposure: Param::from_f32(1.0),
            color_space: "linear".into(),
        },
        meta: Meta::new(),
    }
}

fn minimal_camera() -> CameraData {
    CameraData::new(
        Transform::IDENTITY,
        CameraProjection::Perspective(PerspectiveCamera {
            fov_y: Param::from_f32(0.7),
            near: 0.01,
            far: 100.0,
            aperture: Param::from_f32(0.0),
            focus_distance: Param::from_f32(5.0),
        }),
    )
}

fn empty_input() -> RenderInput {
    RenderInput::new(minimal_camera(), minimal_render_config())
}

fn roundtrip<T>(v: &T) -> Vec<u8>
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let bytes = bincode::serialize(v).expect("serialize");
    let back: T = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(&back, v);
    let bytes2 = bincode::serialize(&back).expect("reserialize");
    assert_eq!(bytes, bytes2, "二次序列化字节不一致");
    bytes
}

// ── 往返 ──

#[test]
fn empty_input_roundtrip_and_validates() {
    let input = empty_input();
    assert!(input.validate().is_ok());
    roundtrip(&input);
}

#[test]
fn input_with_quad_mesh_roundtrip() {
    let mut input = empty_input();
    let geom_ref = GeometryRef(EntityId::from_name("quad"));

    let mut v_attrs = BTreeMap::new();
    v_attrs.insert(
        "position".to_string(),
        AttributeBuffer::f32(
            3,
            vec![
                0.0, 0.0, 0.0,
                1.0, 0.0, 0.0,
                1.0, 1.0, 0.0,
                0.0, 1.0, 0.0,
            ],
        ),
    );
    input.geometries.insert(
        geom_ref,
        Mesh {
            vertex_attributes: v_attrs,
            indices: vec![[0, 1, 2], [0, 2, 3]],
            edges: vec![],
            edge_attributes: BTreeMap::new(),
            face_attributes: BTreeMap::new(),
            sub_meshes: vec![],
        },
    );

    assert!(input.validate().is_ok());
    roundtrip(&input);
}

// ── 引用可达性 ──

#[test]
fn missing_geometry_reference_fails() {
    let mut input = empty_input();
    input.instances.push(RenderInstance {
        geometry: GeometryRef(EntityId::from_name("missing")),
        material_refs: vec![],
        world_transform: Transform::IDENTITY,
        displacement: None,
        visibility: VisibilityFlags::default(),
        meta: Meta::new(),
    });
    assert!(input.validate().is_err());
}

#[test]
fn missing_material_reference_fails() {
    let mut input = empty_input();
    let geom_ref = GeometryRef(EntityId::from_name("quad"));
    input.geometries.insert(geom_ref, Mesh::default());
    input.instances.push(RenderInstance {
        geometry: geom_ref,
        material_refs: vec![MaterialRef(EntityId::from_name("nope"))],
        world_transform: Transform::IDENTITY,
        displacement: None,
        visibility: VisibilityFlags::default(),
        meta: Meta::new(),
    });
    assert!(input.validate().is_err());
}

#[test]
fn missing_field_in_material_fails() {
    let mut input = empty_input();
    let mat_ref = MaterialRef(EntityId::from_name("m"));
    input.materials.insert(
        mat_ref,
        Material {
            bsdf: FieldId(EntityId::from_name("missing")),
        },
    );
    assert!(input.validate().is_err());
}

#[test]
fn missing_displacement_field_fails() {
    let mut input = empty_input();
    let geom_ref = GeometryRef(EntityId::from_name("g"));
    input.geometries.insert(geom_ref, Mesh::default());
    input.instances.push(RenderInstance {
        geometry: geom_ref,
        material_refs: vec![],
        world_transform: Transform::IDENTITY,
        displacement: Some(FieldId(EntityId::from_name("missing"))),
        visibility: VisibilityFlags::default(),
        meta: Meta::new(),
    });
    assert!(input.validate().is_err());
}

#[test]
fn missing_field_in_light_fails() {
    let mut input = empty_input();
    let light_ref = LightRef(EntityId::from_name("l"));
    input.lights.insert(
        light_ref,
        LightData::new(LightType::Point(PointLight {
            position: [0.0, 0.0, 0.0],
            intensity: Param::Field(FieldId(EntityId::from_name("missing"))),
            color: Param::from_vec3([1.0, 1.0, 1.0]),
        })),
    );
    assert!(input.validate().is_err());
}

#[test]
fn missing_field_in_camera_fails() {
    let mut input = empty_input();
    input.camera = CameraData::new(
        Transform::IDENTITY,
        CameraProjection::Perspective(PerspectiveCamera {
            fov_y: Param::Field(FieldId(EntityId::from_name("missing"))),
            near: 0.01,
            far: 100.0,
            aperture: Param::from_f32(0.0),
            focus_distance: Param::from_f32(5.0),
        }),
    );
    assert!(input.validate().is_err());
}

#[test]
fn missing_field_in_world_meta_fails() {
    let mut input = empty_input();
    let mut meta = Meta::new();
    meta.insert(
        "env_curve",
        Param::Field(FieldId(EntityId::from_name("missing"))),
    );
    input.world = WorldData::default().with_meta(meta);
    assert!(input.validate().is_err());
}

// ── 节点图 ──

#[test]
fn missing_node_in_field_fails() {
    let mut input = empty_input();
    input.fields.insert(
        FieldId(EntityId::from_name("f")),
        FieldExpr::NodeOutput(NodeOutput::new(
            NodeId(EntityId::from_name("missing")),
            "out",
        )),
    );
    assert!(input.validate().is_err());
}

#[test]
fn node_graph_cycle_fails() {
    let mut input = empty_input();
    let a = NodeId(EntityId::from_name("a"));
    let b = NodeId(EntityId::from_name("b"));
    input.nodes.insert(
        a,
        Node::new("n1").with_input("x", NodeInput::Node(NodeOutput::single(b))),
    );
    input.nodes.insert(
        b,
        Node::new("n2").with_input("y", NodeInput::Node(NodeOutput::single(a))),
    );
    input.fields.insert(
        FieldId(EntityId::from_name("f")),
        FieldExpr::NodeOutput(NodeOutput::single(a)),
    );
    assert!(matches!(
        input.validate(),
        Err(DataError::NodeGraphCycle(_))
    ));
}

#[test]
fn node_graph_self_cycle_fails() {
    let mut input = empty_input();
    let a = NodeId(EntityId::from_name("a"));
    input.nodes.insert(
        a,
        Node::new("n").with_input("self", NodeInput::Node(NodeOutput::single(a))),
    );
    input.fields.insert(
        FieldId(EntityId::from_name("f")),
        FieldExpr::NodeOutput(NodeOutput::single(a)),
    );
    assert!(matches!(
        input.validate(),
        Err(DataError::NodeGraphCycle(_))
    ));
}

#[test]
fn node_graph_dag_ok() {
    // a -> b -> c，无环
    let a = NodeId(EntityId::from_name("a"));
    let b = NodeId(EntityId::from_name("b"));
    let c = NodeId(EntityId::from_name("c"));
    let mut input = empty_input();
    input.nodes.insert(
        a,
        Node::new("noise").with_input("scale", NodeInput::const_f32(1.0)),
    );
    input.nodes.insert(
        b,
        Node::new("remap").with_input("input", NodeInput::Node(NodeOutput::single(a))),
    );
    input.nodes.insert(
        c,
        Node::new("mix").with_input("a", NodeInput::Node(NodeOutput::single(b))),
    );
    input.fields.insert(
        FieldId(EntityId::from_name("f")),
        FieldExpr::NodeOutput(NodeOutput::single(c)),
    );
    assert!(input.validate().is_ok());
}

#[test]
fn node_graph_shared_upstream_ok() {
    // 同一节点被两处引用（DAG 而非树）
    let shared = NodeId(EntityId::from_name("shared"));
    let mix = NodeId(EntityId::from_name("mix"));
    let mut input = empty_input();
    input.nodes.insert(
        shared,
        Node::new("noise").with_input("scale", NodeInput::const_f32(1.0)),
    );
    input.nodes.insert(
        mix,
        Node::new("add")
            .with_input("a", NodeInput::Node(NodeOutput::single(shared)))
            .with_input("b", NodeInput::Node(NodeOutput::single(shared))),
    );
    input.fields.insert(
        FieldId(EntityId::from_name("f")),
        FieldExpr::NodeOutput(NodeOutput::single(mix)),
    );
    assert!(input.validate().is_ok());
}

// ── 完整场景 ──

#[test]
fn valid_scene_with_material_and_light() {
    let mut input = empty_input();

    let geom_ref = GeometryRef(EntityId::from_name("quad"));
    input.geometries.insert(geom_ref, Mesh::default());

    let bsdf_node = NodeId(EntityId::from_name("bsdf"));
    input.nodes.insert(
        bsdf_node,
        Node::new("lambertian")
            .with_input("reflectance", NodeInput::const_vec3([0.8, 0.2, 0.2])),
    );

    let bsdf_field = FieldId(EntityId::from_name("bsdf_field"));
    input.fields.insert(
        bsdf_field,
        FieldExpr::NodeOutput(NodeOutput::new(bsdf_node, "bsdf")),
    );

    let mat_ref = MaterialRef(EntityId::from_name("mat_red"));
    input
        .materials
        .insert(mat_ref, Material { bsdf: bsdf_field });

    input.instances.push(RenderInstance {
        geometry: geom_ref,
        material_refs: vec![mat_ref],
        world_transform: Transform::IDENTITY,
        displacement: None,
        visibility: VisibilityFlags::default(),
        meta: Meta::new(),
    });

    let light_ref = LightRef(EntityId::from_name("key_light"));
    input.lights.insert(
        light_ref,
        LightData::new(LightType::Point(PointLight {
            position: [0.0, 3.0, 0.0],
            intensity: Param::from_f32(100.0),
            color: Param::from_vec3([1.0, 1.0, 1.0]),
        })),
    );

    assert!(input.validate().is_ok());
    roundtrip(&input);
}

// ── 体积独立表 ──

#[test]
fn volumes_independent_from_world() {
    let mut input = empty_input();

    // 全局雾
    let fog_ref = VolumeRef(EntityId::from_name("global_fog"));
    input.volumes.insert(
        fog_ref,
        VolumeInstance::new(
            VolumeData::new("homogeneous").with_param("density", Param::from_f32(0.001)),
            VolumeBounds::Infinite,
        ),
    );

    // 局部烟
    let smoke_ref = VolumeRef(EntityId::from_name("smoke"));
    let mut smoke = VolumeInstance::new(
        VolumeData::new("heterogeneous").with_param("density", Param::from_f32(0.5)),
        VolumeBounds::Aabb {
            min: [0.0, 0.0, 0.0],
            max: [2.0, 3.0, 2.0],
        },
    );
    smoke.priority = 10;
    input.volumes.insert(smoke_ref, smoke);

    assert_eq!(input.volumes.len(), 2);
    assert!(input.validate().is_ok());
    roundtrip(&input);
}

#[test]
fn volume_mesh_envelope_missing_geometry_fails() {
    let mut input = empty_input();
    input.volumes.insert(
        VolumeRef(EntityId::from_name("v")),
        VolumeInstance::new(
            VolumeData::new("homogeneous"),
            VolumeBounds::MeshEnvelope(GeometryRef(EntityId::from_name("missing"))),
        ),
    );
    assert!(input.validate().is_err());
}

#[test]
fn volume_mesh_envelope_with_existing_geometry_ok() {
    let mut input = empty_input();
    let geom_ref = GeometryRef(EntityId::from_name("volume_shape"));
    input.geometries.insert(geom_ref, Mesh::default());
    input.volumes.insert(
        VolumeRef(EntityId::from_name("v")),
        VolumeInstance::new(
            VolumeData::new("homogeneous"),
            VolumeBounds::MeshEnvelope(geom_ref),
        ),
    );
    assert!(input.validate().is_ok());
}

#[test]
fn volume_field_reference_missing_fails() {
    let mut input = empty_input();
    input.volumes.insert(
        VolumeRef(EntityId::from_name("v")),
        VolumeInstance::new(
            VolumeData::new("heterogeneous").with_param(
                "density",
                Param::Field(FieldId(EntityId::from_name("missing"))),
            ),
            VolumeBounds::Infinite,
        ),
    );
    assert!(input.validate().is_err());
}

#[test]
fn volume_field_reference_ok() {
    let mut input = empty_input();
    let density_field = FieldId(EntityId::from_name("density"));
    input.fields.insert(
        density_field,
        FieldExpr::Constant(kzray_data::common::Value::Float(0.5)),
    );
    input.volumes.insert(
        VolumeRef(EntityId::from_name("v")),
        VolumeInstance::new(
            VolumeData::new("heterogeneous").with_param("density", Param::Field(density_field)),
            VolumeBounds::Infinite,
        ),
    );
    assert!(input.validate().is_ok());
    roundtrip(&input);
}

// ── 数据合法性 ──

#[test]
fn invalid_mesh_fails() {
    let mut input = empty_input();
    let geom_ref = GeometryRef(EntityId::from_name("bad_mesh"));
    // 空索引但属性数量对不上
    let mut attrs = BTreeMap::new();
    attrs.insert("position".to_string(), AttributeBuffer::f32(3, vec![0.0; 6])); // 2 个顶点
    input.geometries.insert(
        geom_ref,
        Mesh {
            vertex_attributes: attrs,
            indices: vec![[0, 1, 5]], // 索引 5 越界
            edges: vec![],
            edge_attributes: BTreeMap::new(),
            face_attributes: BTreeMap::new(),
            sub_meshes: vec![],
        },
    );
    assert!(input.validate().is_err());
}

#[test]
fn nan_in_light_fails() {
    let mut input = empty_input();
    input.lights.insert(
        LightRef(EntityId::from_name("l")),
        LightData::new(LightType::Point(PointLight {
            position: [0.0, 0.0, 0.0],
            intensity: Param::from_f32(f32::NAN),
            color: Param::from_vec3([1.0, 1.0, 1.0]),
        })),
    );
    assert!(input.validate().is_err());
}

#[test]
fn nan_in_volume_fails() {
    let mut input = empty_input();
    input.volumes.insert(
        VolumeRef(EntityId::from_name("v")),
        VolumeInstance::new(
            VolumeData::new("homogeneous").with_param("density", Param::from_f32(f32::NAN)),
            VolumeBounds::Infinite,
        ),
    );
    assert!(input.validate().is_err());
}
