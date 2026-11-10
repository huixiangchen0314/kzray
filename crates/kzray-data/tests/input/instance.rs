//! 对应 `src/input/instance.rs`。

use kzray_data::common::{
    EntityId, FieldId, GeometryRef, HasMeta, MaterialRef, Meta, Param, Transform,
};
use kzray_data::input::{RenderInstance, VisibilityFlags};

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

#[test]
fn visibility_defaults_all_on() {
    let f = VisibilityFlags::default();
    assert!(f.camera);
    assert!(f.shadow);
    assert!(f.diffuse);
    assert!(f.glossy);
    assert!(f.transmission);
    roundtrip(&f);
}

#[test]
fn visibility_custom() {
    let f = VisibilityFlags {
        camera: false,
        shadow: true,
        diffuse: true,
        glossy: false,
        transmission: false,
    };
    roundtrip(&f);
}

#[test]
fn minimal_instance() {
    let inst = RenderInstance {
        geometry: GeometryRef(EntityId::from_name("quad")),
        material_refs: vec![MaterialRef(EntityId::from_name("mat_red"))],
        world_transform: Transform::IDENTITY,
        displacement: None,
        visibility: VisibilityFlags::default(),
        meta: Meta::new(),
    };
    roundtrip(&inst);
}

#[test]
fn instance_with_displacement() {
    let inst = RenderInstance {
        geometry: GeometryRef(EntityId::from_name("sphere")),
        material_refs: vec![MaterialRef(EntityId::from_name("mat_glass"))],
        world_transform: Transform::IDENTITY,
        displacement: Some(FieldId(EntityId::from_name("bump"))),
        visibility: VisibilityFlags::default(),
        meta: Meta::new(),
    };
    roundtrip(&inst);
}

#[test]
fn instance_with_multiple_materials() {
    // 多子网格材质：material_refs 有多个槽
    let inst = RenderInstance {
        geometry: GeometryRef(EntityId::from_name("multi_mat_mesh")),
        material_refs: vec![
            MaterialRef(EntityId::from_name("mat_a")),
            MaterialRef(EntityId::from_name("mat_b")),
            MaterialRef(EntityId::from_name("mat_c")),
        ],
        world_transform: Transform::IDENTITY,
        displacement: None,
        visibility: VisibilityFlags::default(),
        meta: Meta::new(),
    };
    assert_eq!(inst.material_refs.len(), 3);
    roundtrip(&inst);
}

#[test]
fn instance_with_world_transform() {
    let inst = RenderInstance {
        geometry: GeometryRef(EntityId::from_name("box")),
        material_refs: vec![],
        world_transform: Transform::from_cols([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [5.0, 0.0, 0.0, 1.0],
        ]),
        displacement: None,
        visibility: VisibilityFlags::default(),
        meta: Meta::new(),
    };
    roundtrip(&inst);
}

#[test]
fn instance_carries_meta() {
    let mut meta = Meta::new();
    meta.insert("name", Param::from_str("hero_ball"));
    meta.insert("render_layer", Param::from_str("fg"));
    let inst = RenderInstance {
        geometry: GeometryRef(EntityId::from_name("sphere")),
        material_refs: vec![],
        world_transform: Transform::IDENTITY,
        displacement: None,
        visibility: VisibilityFlags::default(),
        meta,
    };
    assert_eq!(inst.meta().len(), 2);
    roundtrip(&inst);
}
