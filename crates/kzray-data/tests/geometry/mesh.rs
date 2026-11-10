//! 对应 `src/geometry/mesh.rs`。

use std::collections::BTreeMap;

use kzray_data::geometry::{AttributeBuffer, AttributeValues, Mesh, SubMesh};

use super::roundtrip;

fn quad() -> Mesh {
    let mut v = BTreeMap::new();
    v.insert(
        "position".to_string(),
        AttributeBuffer::f32(3, vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
        ]),
    );
    Mesh {
        vertex_attributes: v,
        indices: vec![[0, 1, 2], [0, 2, 3]],
        edges: vec![[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
        edge_attributes: BTreeMap::new(),
        face_attributes: BTreeMap::new(),
        sub_meshes: vec![],
    }
}

#[test]
fn basic_counts() {
    let m = quad();
    assert_eq!(m.vertex_count(), 4);
    assert_eq!(m.triangle_count(), 2);
    assert_eq!(m.edge_count(), 5);
}

#[test]
fn roundtrip_minimal_and_empty() {
    roundtrip(&quad());
    roundtrip(&Mesh::default());
}

#[test]
fn vertex_domain_attrs() {
    let mut m = quad();
    m.vertex_attributes.insert(
        "uv".to_string(),
        AttributeBuffer::f32(2, vec![
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ]),
    );
    m.vertex_attributes.insert(
        "normal".to_string(),
        AttributeBuffer::f32(3, vec![0.0, 0.0, 1.0].repeat(4)),
    );
    assert!(m.validate().is_ok());
    roundtrip(&m);
}

#[test]
fn edge_domain_attrs() {
    let mut m = quad();
    m.edge_attributes.insert(
        "highlight".to_string(),
        AttributeBuffer::bool(1, vec![false, false, true, false, false]),
    );
    assert!(m.validate().is_ok());
    let hl = m.edge_attribute("highlight").unwrap();
    assert!(matches!(&hl.values, AttributeValues::Bool(v) if v[2]));
    roundtrip(&m);
}

#[test]
fn face_domain_attrs() {
    let mut m = quad();
    m.face_attributes.insert(
        "face_id".to_string(),
        AttributeBuffer::u32(1, vec![10, 20]),
    );
    assert!(m.validate().is_ok());
    roundtrip(&m);
}

#[test]
fn three_domains_coexist() {
    let mut m = quad();
    m.vertex_attributes.insert("uv".to_string(), AttributeBuffer::f32(2, vec![0.0; 8]));
    m.edge_attributes.insert("highlight".to_string(), AttributeBuffer::bool(1, vec![false; 5]));
    m.face_attributes.insert("face_id".to_string(), AttributeBuffer::u32(1, vec![0, 1]));
    assert!(m.validate().is_ok());
    roundtrip(&m);
}

#[test]
fn sub_meshes() {
    let mut m = quad();
    m.sub_meshes = vec![
        SubMesh { tri_start: 0, tri_end: 1, material_index: 0 },
        SubMesh { tri_start: 1, tri_end: 2, material_index: 1 },
    ];
    assert_eq!(m.material_index(0), 0);
    assert_eq!(m.material_index(1), 1);
    assert!(m.validate().is_ok());
    roundtrip(&m);
}

#[test]
fn empty_sub_meshes_means_material_zero() {
    let m = quad();
    assert_eq!(m.material_index(0), 0);
    assert_eq!(m.material_index(1), 0);
}

#[test]
fn validate_rejects_wrong_vertex_count() {
    let mut m = quad();
    m.vertex_attributes.insert("uv".to_string(), AttributeBuffer::f32(2, vec![0.0; 6]));
    assert!(m.validate().is_err());
}

#[test]
fn validate_rejects_wrong_edge_count() {
    let mut m = quad();
    m.edge_attributes.insert("highlight".to_string(), AttributeBuffer::bool(1, vec![true; 4]));
    assert!(m.validate().is_err());
}

#[test]
fn validate_rejects_wrong_face_count() {
    let mut m = quad();
    m.face_attributes.insert("highlight".to_string(), AttributeBuffer::bool(1, vec![true; 3]));
    assert!(m.validate().is_err());
}

#[test]
fn validate_rejects_index_oob() {
    let mut m = quad();
    m.indices[0] = [0, 1, 10];
    assert!(m.validate().is_err());
}

#[test]
fn validate_rejects_edge_index_oob() {
    let mut m = quad();
    m.edges[0] = [0, 100];
    assert!(m.validate().is_err());
}

#[test]
fn validate_rejects_sub_mesh_gap() {
    let mut m = quad();
    m.sub_meshes = vec![SubMesh { tri_start: 0, tri_end: 1, material_index: 0 }];
    assert!(m.validate().is_err());
}

#[test]
fn validate_rejects_sub_mesh_overflow() {
    let mut m = quad();
    m.sub_meshes = vec![SubMesh { tri_start: 0, tri_end: 3, material_index: 0 }];
    assert!(m.validate().is_err());
}
