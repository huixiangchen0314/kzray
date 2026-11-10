//! 对应 `src/geometry/attribute.rs`。

use kzray_data::geometry::{AttributeBuffer, AttributeValues};

use super::roundtrip;

#[test]
fn f32_vec3() {
    let a = AttributeBuffer::f32(3, vec![
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    ]);
    assert_eq!(a.count(), 3);
    assert_eq!(a.get_f32(1), Some(&[1.0, 0.0, 0.0][..]));
    assert_eq!(a.get_f32(3), None);
    roundtrip(&a);
}

#[test]
fn mat3_stride9() {
    // stride 不限 1..=4
    let a = AttributeBuffer::f32(9, (0..18).map(|x| x as f32).collect());
    assert_eq!(a.count(), 2);
    roundtrip(&a);
}

#[test]
fn i32_u32_bool() {
    let i = AttributeBuffer::i32(2, vec![1, 2, 3, 4]);
    assert_eq!(i.count(), 2);
    assert!(matches!(i.values, AttributeValues::I32(_)));
    roundtrip(&i);

    let u = AttributeBuffer::u32(1, vec![10, 20, 30]);
    assert_eq!(u.as_u32(), Some(&[10, 20, 30][..]));
    roundtrip(&u);

    roundtrip(&AttributeBuffer::bool(1, vec![true, false, true]));
}

#[test]
fn empty_values() {
    let a = AttributeBuffer::f32(3, vec![]);
    assert_eq!(a.count(), 0);
    assert!(a.values.is_empty());
    roundtrip(&a);
}

#[test]
fn wrong_type_returns_none() {
    let a = AttributeBuffer::u32(1, vec![1, 2, 3]);
    assert_eq!(a.get_f32(0), None);
    assert_eq!(a.as_f32(), None);
}
