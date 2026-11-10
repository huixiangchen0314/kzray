//! 对应 `src/common/meta.rs`。

use kzray_data::common::{EntityId, FieldId, Meta, Param};

use super::roundtrip;

#[test]
fn empty_meta_roundtrip() {
    roundtrip(&Meta::new());
    let m = Meta::default();
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
}

#[test]
fn meta_with_params() {
    let mut m = Meta::new();
    m.insert("render_layer", Param::from_str("hero"));
    m.insert("custom_id", Param::from_i64(42));
    assert_eq!(m.len(), 2);
    assert_eq!(m.get("render_layer").and_then(|p| p.as_value()).and_then(|v| v.as_str()), Some("hero"));
    roundtrip(&m);
}

#[test]
fn meta_with_field_ref() {
    let mut m = Meta::new();
    let f = FieldId(EntityId::from_name("anim"));
    m.insert("animation", Param::Field(f));
    roundtrip(&m);
}

#[test]
fn remove() {
    let mut m = Meta::new();
    m.insert("x", Param::from_f32(1.0));
    assert!(m.remove("x").is_some());
    assert!(m.is_empty());
    assert!(m.remove("x").is_none());
}

#[test]
fn iter() {
    let mut m = Meta::new();
    m.insert("a", Param::from_f32(1.0));
    m.insert("b", Param::from_f32(2.0));
    let keys: Vec<&str> = m.iter().map(|(k, _)| k).collect();
    assert_eq!(keys, vec!["a", "b"]);
}

#[test]
fn validate_rejects_nan_in_meta() {
    let mut m = Meta::new();
    m.insert("bad", Param::from_f32(f32::NAN));
    assert!(m.validate("x").is_err());
}
