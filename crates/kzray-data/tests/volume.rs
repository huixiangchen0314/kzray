//! 对应 `src/volume.rs`。

use kzray_data::common::{EntityId, FieldId, MeshRef, HasMeta, Meta, Param, Transform};
use kzray_data::volume::{VolumeBounds, VolumeData, VolumeInstance};

fn roundtrip<T>(v: &T) -> Vec<u8>
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let bytes = bincode::serialize(v).expect("serialize");
    let back: T = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(&back, v);
    assert_eq!(bytes, bincode::serialize(&back).unwrap());
    bytes
}

#[test]
fn homogeneous_volume_data() {
    let v = VolumeData::new("homogeneous")
        .with_param("density", Param::from_f32(0.01))
        .with_param("scatter_albedo", Param::from_vec3([1.0, 1.0, 1.0]));
    assert!(v.validate("x").is_ok());
    roundtrip(&v);
}

#[test]
fn bounds_infinite() {
    roundtrip(&VolumeBounds::Infinite);
}

#[test]
fn bounds_aabb() {
    let b = VolumeBounds::Aabb {
        min: [-1.0, 0.0, -1.0],
        max: [1.0, 2.0, 1.0],
    };
    assert!(b.validate("x").is_ok());
    roundtrip(&b);
}

#[test]
fn bounds_sphere() {
    let b = VolumeBounds::Sphere {
        center: [0.0, 1.0, 0.0],
        radius: 2.0,
    };
    assert!(b.validate("x").is_ok());
    roundtrip(&b);
}

#[test]
fn bounds_transformed_box() {
    let b = VolumeBounds::TransformedBox {
        transform: Transform::IDENTITY,
        half_extents: [1.0, 2.0, 0.5],
    };
    assert!(b.validate("x").is_ok());
    roundtrip(&b);
}

#[test]
fn bounds_mesh_envelope() {
    let b = VolumeBounds::MeshEnvelope(MeshRef(EntityId::from_name("smoke_mesh")));
    assert!(b.validate("x").is_ok());
    roundtrip(&b);
}

#[test]
fn bounds_validate_rejects_bad_aabb() {
    let b = VolumeBounds::Aabb {
        min: [1.0, 0.0, 0.0],
        max: [-1.0, 1.0, 1.0],   // min.x > max.x
    };
    assert!(b.validate("x").is_err());
}

#[test]
fn bounds_validate_rejects_negative_radius() {
    let b = VolumeBounds::Sphere { center: [0.0; 3], radius: -1.0 };
    assert!(b.validate("x").is_err());
}

#[test]
fn volume_instance_roundtrip() {
    let vi = VolumeInstance::new(
        VolumeData::new("heterogeneous")
            .with_param("density", Param::from_f32(0.5)),
        VolumeBounds::Aabb { min: [0.0; 3], max: [1.0; 3] },
    );
    assert_eq!(vi.priority, 0);
    assert!(vi.validate("x").is_ok());
    roundtrip(&vi);
}

#[test]
fn volume_instance_with_field_density() {
    let density_field = FieldId(EntityId::from_name("smoke_density"));
    let vi = VolumeInstance::new(
        VolumeData::new("heterogeneous").with_param("density", Param::Field(density_field)),
        VolumeBounds::Aabb { min: [0.0; 3], max: [1.0; 3] },
    );
    assert!(vi.validate("x").is_ok());
    roundtrip(&vi);
}

#[test]
fn volume_instance_with_meta_and_priority() {
    let mut meta = Meta::new();
    meta.insert("render_layer", Param::from_str("smoke"));
    let mut vi = VolumeInstance::new(
        VolumeData::new("homogeneous"),
        VolumeBounds::Infinite,
    );
    vi.priority = 5;
    vi.meta = meta;
    assert_eq!(vi.meta().len(), 1);
    roundtrip(&vi);
}
