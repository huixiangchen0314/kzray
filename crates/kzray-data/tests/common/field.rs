//! 对应 `src/common/field.rs`。

use kzray_data::common::{EntityId, FieldExpr, NodeId, NodeOutput, Value};

use super::roundtrip;

#[test]
fn constant_roundtrip() {
    roundtrip(&FieldExpr::Constant(Value::Float(0.5)));
    roundtrip(&FieldExpr::Constant(Value::Array(vec![
        Value::Float(1.0),
        Value::Vec3([0.0, 1.0, 0.0]),
    ])));
}

#[test]
fn node_output_roundtrip() {
    let n = NodeId(EntityId::from_name("bsdf_1"));
    let f = FieldExpr::NodeOutput(NodeOutput::new(n, "color"));
    roundtrip(&f);
    if let FieldExpr::NodeOutput(o) = f {
        assert_eq!(o.node, n);
        assert_eq!(o.port, "color");
    } else {
        panic!("期望 NodeOutput");
    }
}

#[test]
fn same_node_multiple_ports_distinct_fields() {
    let n = NodeId(EntityId::from_name("material_node"));
    let f_color = FieldExpr::NodeOutput(NodeOutput::new(n, "color"));
    let f_normal = FieldExpr::NodeOutput(NodeOutput::new(n, "normal"));
    assert_ne!(f_color, f_normal);
}

#[test]
fn constant_excludes_references() {
    // 编译期保证：`Constant` 只接受 `Value`，无法装 `NodeOutput` / `FieldId`。
    // 下面两行会编译失败：
    //   FieldExpr::Constant(NodeOutput::single(...));
    //   FieldExpr::Constant(FieldId(...));
    let _ = FieldExpr::Constant(Value::Float(1.0));
}
