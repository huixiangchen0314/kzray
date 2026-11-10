//! 对应 `src/common/node.rs`。

use kzray_data::common::{
    EntityId, FieldId, Node, NodeId, NodeInput, NodeOutput, Value,
};

use super::roundtrip;

#[test]
fn node_output_roundtrip() {
    let n = NodeId(EntityId::from_name("n"));
    let o = NodeOutput::new(n, "value");
    roundtrip(&o);
    assert_eq!(o.node, n);
    assert_eq!(o.port, "value");
}

#[test]
fn node_output_single_defaults_to_out() {
    let n = NodeId(EntityId::from_name("n"));
    assert_eq!(NodeOutput::single(n).port, "out");
}

#[test]
fn input_const_variants() {
    roundtrip(&NodeInput::const_f32(0.5));
    roundtrip(&NodeInput::const_vec3([0.8, 0.2, 0.2]));
    roundtrip(&NodeInput::const_bool(true));
    roundtrip(&NodeInput::const_str("ggx"));
}

#[test]
fn input_const_array_of_pure_values() {
    let v = NodeInput::Const(Value::Array(vec![
        Value::Float(1.0),
        Value::Float(2.0),
        Value::Float(3.0),
    ]));
    roundtrip(&v);
    if let NodeInput::Const(Value::Array(items)) = &v {
        assert_eq!(items.len(), 3);
    } else {
        panic!("期望常量数组");
    }
}

#[test]
fn input_node_ref_roundtrip() {
    let n = NodeId(EntityId::from_name("noise"));
    roundtrip(&NodeInput::Node(NodeOutput::new(n, "value")));
}

#[test]
fn input_field_ref_roundtrip() {
    let f = FieldId(EntityId::from_name("mask"));
    roundtrip(&NodeInput::Field(f));
}

#[test]
fn node_no_inputs() {
    let n = Node::new("constant");
    roundtrip(&n);
    assert!(n.upstream_nodes().is_empty());
    assert!(n.upstream_fields().is_empty());
}

#[test]
fn upstream_is_flat_scan() {
    let a = NodeId(EntityId::from_name("a"));
    let b = NodeId(EntityId::from_name("b"));
    let f = FieldId(EntityId::from_name("f"));
    let n = Node::new("mix")
        .with_input("input_a", NodeInput::Node(NodeOutput::single(a)))
        .with_input("input_b", NodeInput::Node(NodeOutput::single(b)))
        .with_input("mask", NodeInput::Field(f))
        .with_input("weight", NodeInput::const_f32(0.5));

    assert_eq!(n.upstream_nodes(), vec![a, b]);
    assert_eq!(n.upstream_fields(), vec![f]);
    roundtrip(&n);
}

#[test]
fn upstream_outputs_carry_ports() {
    let noise = NodeId(EntityId::from_name("noise"));
    let n = Node::new("mix")
        .with_input("a", NodeInput::Node(NodeOutput::new(noise, "value")))
        .with_input("b", NodeInput::Node(NodeOutput::new(noise, "gradient")));
    let ups = n.upstream_outputs();
    assert_eq!(ups.len(), 2);
    assert_eq!(ups[0].port, "value");
    assert_eq!(ups[1].port, "gradient");
}
