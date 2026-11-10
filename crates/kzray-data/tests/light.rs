//! 对应 `src/light.rs`。

use kzray_data::common::{EntityId, FieldId, HasMeta, Meta, Param, Transform};
use kzray_data::light::{
    AreaLight, AreaShape, BackgroundLight, LightData, LightFlags, LightType, PointLight, SpotLight,
    SunLight,
};

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

fn f32_param(v: f32) -> Param { Param::from_f32(v) }
fn vec3_param(v: [f32; 3]) -> Param { Param::from_vec3(v) }

#[test]
fn flags_default() {
    let f = LightFlags::default();
    assert!(!f.camera_visible);
    assert!(f.diffuse_visible);
    assert!(f.cast_shadows);
    roundtrip(&f);
}

#[test]
fn point_light() {
    let l = LightData::point([0.0, 3.0, 0.0], f32_param(100.0), vec3_param([1.0, 1.0, 0.9]));
    assert!(matches!(l.kind, LightType::Point(_)));
    assert!(l.validate().is_ok());
    roundtrip(&l);
}

#[test]
fn point_light_with_field_intensity() {
    let curve = FieldId(EntityId::from_name("intensity_curve"));
    let l = LightData::new(LightType::Point(PointLight {
        position: [0.0, 3.0, 0.0],
        intensity: Param::Field(curve),
        color: vec3_param([1.0, 1.0, 1.0]),
    }));
    assert!(l.validate().is_ok());
    roundtrip(&l);
}

#[test]
fn spot_light() {
    let l = LightData::new(LightType::Spot(SpotLight {
        position: [0.0, 5.0, 0.0],
        direction: [0.0, -1.0, 0.0],
        cone_angle: f32_param(0.5),
        penumbra: f32_param(0.3),
        intensity: f32_param(200.0),
        color: vec3_param([1.0, 1.0, 1.0]),
    }));
    assert!(l.validate().is_ok());
    roundtrip(&l);
}

#[test]
fn area_rectangle() {
    let l = LightData::new(LightType::Area(AreaLight {
        transform: Transform::IDENTITY,
        shape: AreaShape::Rectangle {
            size_u: f32_param(2.0),
            size_v: f32_param(1.0),
        },
        intensity: f32_param(50.0),
        color: vec3_param([1.0, 0.95, 0.9]),
    }));
    assert!(l.validate().is_ok());
    roundtrip(&l);
}

#[test]
fn area_disk_and_sphere() {
    roundtrip(&AreaShape::Disk { radius: f32_param(0.5) });
    roundtrip(&AreaShape::Sphere { radius: f32_param(1.0) });
}

#[test]
fn sun_light() {
    let l = LightData::new(LightType::Sun(SunLight {
        direction: [0.0, -1.0, -0.2],
        angle: f32_param(0.0046),
        intensity: f32_param(1.0),
        color: vec3_param([1.0, 0.95, 0.85]),
    }));
    assert!(l.validate().is_ok());
    roundtrip(&l);
}

#[test]
fn background_light() {
    let l = LightData::new(LightType::Background(BackgroundLight {
        color: vec3_param([0.5, 0.7, 1.0]),
        intensity: f32_param(1.0),
    }));
    assert!(l.validate().is_ok());
    roundtrip(&l);
}

#[test]
fn validate_rejects_nan_intensity() {
    let l = LightData::point([0.0, 0.0, 0.0], f32_param(f32::NAN), vec3_param([1.0, 1.0, 1.0]));
    assert!(l.validate().is_err());
}

// ── HasMeta：核心集封闭 + 动态拓展 ──

#[test]
fn light_carries_meta() {
    let mut meta = Meta::new();
    meta.insert("render_layer", Param::from_str("fg"));
    meta.insert("custom_priority", Param::from_i64(10));

    let l = LightData::point([0.0, 0.0, 0.0], f32_param(1.0), vec3_param([1.0, 1.0, 1.0]))
        .with_meta(meta);

    assert_eq!(l.meta().len(), 2);
    assert_eq!(
        l.meta().get("render_layer").and_then(|p| p.as_value()).and_then(|v| v.as_str()),
        Some("fg")
    );
    assert!(l.validate().is_ok());
    roundtrip(&l);
}

#[test]
fn has_meta_trait_works_on_light() {
    // 通过 trait 统一访问
    fn meta_len<T: HasMeta>(t: &T) -> usize { t.meta().len() }

    let mut l = LightData::point([0.0, 0.0, 0.0], f32_param(1.0), vec3_param([1.0, 1.0, 1.0]));
    assert_eq!(meta_len(&l), 0);
    l.meta_mut().insert("x", Param::from_f32(1.0));
    assert_eq!(meta_len(&l), 1);
}
