//! 对应 `src/shading/material.rs`。

use kzray_data::common::{EntityId, FieldId};
use kzray_data::shading::Material;

use super::roundtrip;

#[test]
fn material_rooted_at_field() {
    let bsdf = FieldId(EntityId::from_name("my_bsdf_field"));
    let m = Material { bsdf };
    roundtrip(&m);
    assert_eq!(m.bsdf, bsdf);
}

#[test]
fn material_only_carries_bsdf() {
    // 编译期保证：`Material` 只有 `bsdf` 一个字段。
    // 加任何 `displacement` / `normal` 字段都会破坏这个测试。
    let _m = Material { bsdf: FieldId(EntityId::from_name("x")) };
}
