//! 阶段 1 验收：`common` 下所有类型 bincode 往返字节确定。

use kzray_data::{
    common::id::IdMap,
    DataError, EntityId, ParamValue, Rect, Transform,
};

fn roundtrip<T>(v: &T) -> Vec<u8>
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let bytes = bincode::serialize(v).expect("serialize");
    let back: T = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(&back, v);
    // 二次序列化必须字节相同（协议稳定性最低要求）
    let bytes2 = bincode::serialize(&back).expect("reserialize");
    assert_eq!(bytes, bytes2, "二次序列化字节不一致");
    bytes
}

#[test]
fn entity_id_roundtrip() {
    for v in [0u64, 1, u64::MAX, 0xDEAD_BEEF_CAFE_F00D] {
        roundtrip(&EntityId(v));
    }
}

#[test]
fn from_name_is_deterministic() {
    assert_eq!(EntityId::from_name("floor"), EntityId::from_name("floor"));
    assert_eq!(EntityId::from_name(""), EntityId::from_name(""));
}

#[test]
fn from_name_distinguishes() {
    let names = ["floor", "wall", "ceiling", "camera", "light_sun", "material_wood"];
    let ids: Vec<EntityId> = names.iter().map(|n| EntityId::from_name(n)).collect();
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            assert_ne!(ids[i], ids[j], "{} 和 {} 哈希碰撞", names[i], names[j]);
        }
    }
}

#[test]
fn param_value_roundtrip_all_variants() {
    let cases = vec![
        ParamValue::Float(3.14),
        ParamValue::Int(-42),
        ParamValue::Bool(true),
        ParamValue::Vec2([1.0, 2.0]),
        ParamValue::Vec3([1.0, 2.0, 3.0]),
        ParamValue::Vec4([1.0, 2.0, 3.0, 4.0]),
        ParamValue::Mat4([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        ParamValue::Color([0.8, 0.6, 0.4]),
        ParamValue::String("hello".into()),
        ParamValue::Field(EntityId::from_name("intensity_curve")),
        ParamValue::Array(vec![
            ParamValue::Float(1.0),
            ParamValue::Vec3([0.0, 1.0, 0.0]),
        ]),
    ];
    for v in cases {
        roundtrip(&v);
    }
}

#[test]
fn param_value_validate_rejects_nan() {
    let v = ParamValue::Float(f32::NAN);
    assert!(matches!(v.validate("x"), Err(DataError::InvalidParam { .. })));

    let v = ParamValue::Vec3([1.0, f32::INFINITY, 0.0]);
    assert!(matches!(v.validate("x"), Err(DataError::InvalidParam { .. })));

    let v = ParamValue::Float(1.0);
    assert!(v.validate("x").is_ok());
}

#[test]
fn rect_roundtrip_and_ops() {
    let r = Rect::new(10, 20, 32, 32);
    roundtrip(&r);
    assert_eq!(r.area(), 1024);
    assert!(!r.is_empty());

    let a = Rect::new(0, 0, 10, 10);
    let b = Rect::new(5, 5, 10, 10);
    let c = Rect::new(20, 20, 5, 5);
    assert!(a.intersects(&b));
    assert!(!a.intersects(&c));
}

#[test]
fn transform_roundtrip() {
    roundtrip(&Transform::IDENTITY);
    roundtrip(&Transform::from_cols([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [1.0, 2.0, 3.0, 1.0],
    ]));
    assert!(Transform::IDENTITY.validate());
}

#[test]
fn id_map_insert_and_lookup() {
    let mut map: IdMap<u32> = IdMap::new();
    let id = map.insert_named("floor", 1).unwrap();
    assert_eq!(id, EntityId::from_name("floor"));
    assert_eq!(map.get(id), Some(&1));
    assert_eq!(map.name_of(id), Some("floor"));
    assert_eq!(map.len(), 1);
}

#[test]
fn id_map_detects_collision() {
    let mut map: IdMap<u32> = IdMap::new();
    // 用显式 id 模拟"两个名字映射到同一个 u64"
    let fake_id = EntityId(42);
    map.insert(fake_id, "name_a", 1).unwrap();
    let err = map.insert(fake_id, "name_b", 2).unwrap_err();
    match err {
        DataError::IdCollision { new_name, existing_name, id } => {
            assert_eq!(new_name, "name_b");
            assert_eq!(existing_name, "name_a");
            assert_eq!(id, fake_id);
        }
        other => panic!("期望 IdCollision，实际 {other:?}"),
    }
}

#[test]
fn id_map_detects_duplicate() {
    let mut map: IdMap<u32> = IdMap::new();
    map.insert_named("floor", 1).unwrap();
    let err = map.insert_named("floor", 2).unwrap_err();
    assert!(matches!(err, DataError::DuplicateId { .. }));
}
