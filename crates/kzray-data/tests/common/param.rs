//! 对应 `src/common/param.rs`。

use kzray_data::common::{EntityId, FieldId, Param, Value};

use super::roundtrip;

#[test]
fn value_roundtrip() {
    roundtrip(&Param::from_f32(0.5));
    roundtrip(&Param::from_vec3([0.8, 0.2, 0.2]));
    roundtrip(&Param::from_str("ggx"));
    roundtrip(&Param::from_bool(true));
}

#[test]
fn field_roundtrip() {
    let f = FieldId(EntityId::from_name("intensity_curve"));
    roundtrip(&Param::Field(f));
}

#[test]
fn accessors() {
    let v = Param::from_f32(0.5);
    assert_eq!(v.as_value().and_then(|v| v.as_float()), Some(0.5));
    assert_eq!(v.as_field(), None);

    let f = Param::Field(FieldId(EntityId::from_name("x")));
    assert!(f.as_value().is_none());
    assert!(f.as_field().is_some());
}

#[test]
fn validate_rejects_nan() {
    assert!(Param::from_f32(f32::NAN).validate("x").is_err());
    assert!(Param::from_f32(1.0).validate("x").is_ok());
    // Field 引用不做数据校验（由下游解析后检查）
    let f = Param::Field(FieldId(EntityId::from_name("x")));
    assert!(f.validate("x").is_ok());
}

#[test]
fn value_variant_carries_value() {
    let p = Param::Value(Value::Vec3([1.0, 0.0, 0.0]));
    if let Param::Value(Value::Vec3(v)) = p {
        assert_eq!(v, [1.0, 0.0, 0.0]);
    } else {
        panic!("期望 Value(Vec3)");
    }
}
