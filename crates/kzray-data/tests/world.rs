//! 对应 `src/world.rs`。

use kzray_data::common::{HasMeta, Meta, Param};
use kzray_data::world::WorldData;

fn roundtrip<T>(v: &T) -> Vec<u8>
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let bytes = bincode::serialize(v).expect("serialize");
    let back: T = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(&back, v);
    assert_eq!(bytes, bincode::serialize(&back).unwrap());
    bytes
}

#[test]
fn default_world() {
    roundtrip(&WorldData::default());
    assert!(WorldData::default().validate().is_ok());
}

#[test]
fn world_with_meta() {
    let mut meta = Meta::new();
    meta.insert("hdri_rotation", Param::from_f32(0.5));
    meta.insert("env_strength", Param::from_f32(1.5));
    let w = WorldData::default().with_meta(meta);
    assert_eq!(w.meta().len(), 2);
    assert!(w.validate().is_ok());
    roundtrip(&w);
}

#[test]
fn validate_rejects_nan_in_meta() {
    let mut meta = Meta::new();
    meta.insert("bad", Param::from_f32(f32::NAN));
    let w = WorldData::default().with_meta(meta);
    assert!(w.validate().is_err());
}
