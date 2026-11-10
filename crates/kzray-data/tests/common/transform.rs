//! 对应 `src/common/transform.rs`。

use kzray_data::common::Transform;

use super::roundtrip;

#[test]
fn identity_roundtrip() {
    roundtrip(&Transform::IDENTITY);
    assert!(Transform::IDENTITY.validate());
}

#[test]
fn scale_roundtrip() {
    roundtrip(&Transform::from_cols([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]));
}

#[test]
fn validate_rejects_non_finite() {
    let m = Transform([
        [f32::NAN, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    assert!(!m.validate());
}
