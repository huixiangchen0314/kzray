//! 对应 `src/common/rect.rs`。

use kzray_data::common::Rect;

use super::roundtrip;

#[test]
fn rect_roundtrip() {
    roundtrip(&Rect::new(10, 20, 32, 32));
}

#[test]
fn area_and_empty() {
    let r = Rect::new(10, 20, 32, 32);
    assert_eq!(r.area(), 1024);
    assert!(!r.is_empty());
    assert!(Rect::new(0, 0, 0, 10).is_empty());
    assert!(Rect::new(0, 0, 10, 0).is_empty());
}

#[test]
fn intersects() {
    let a = Rect::new(0, 0, 10, 10);
    let b = Rect::new(5, 5, 10, 10);
    let c = Rect::new(20, 20, 5, 5);
    assert!(a.intersects(&b));
    assert!(b.intersects(&a));
    assert!(!a.intersects(&c));
}
