//! 对应 `src/common/value.rs`。

use kzray_data::common::Value;
use kzray_data::DataError;

use super::roundtrip;

#[test]
fn all_variants_roundtrip() {
    for v in [
        Value::Float(3.14),
        Value::Int(-42),
        Value::Bool(true),
        Value::Vec2([1.0, 2.0]),
        Value::Vec3([1.0, 2.0, 3.0]),
        Value::Vec4([1.0, 2.0, 3.0, 4.0]),
        Value::Mat4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        Value::String("ggx".into()),
        Value::Array(vec![Value::Float(1.0), Value::Float(2.0)]),
    ] {
        roundtrip(&v);
    }
}

#[test]
fn array_is_pure_values() {
    let v = Value::Array(vec![Value::Float(1.0), Value::Vec3([0.0, 1.0, 0.0])]);
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[1].as_vec3(), Some([0.0, 1.0, 0.0]));
}

#[test]
fn validate_rejects_nan_inf() {
    assert!(matches!(
        Value::Float(f32::NAN).validate("x"),
        Err(DataError::InvalidParam { .. })
    ));
    assert!(matches!(
        Value::Vec3([1.0, f32::INFINITY, 0.0]).validate("x"),
        Err(DataError::InvalidParam { .. })
    ));
    assert!(Value::Float(1.0).validate("x").is_ok());
}
