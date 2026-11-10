//! 对应 `src/common/id.rs`。

use kzray_data::common::{EntityId, IdMap};
use kzray_data::DataError;

use super::roundtrip;

#[test]
fn entity_id_roundtrip() {
    for v in [0u64, 1, u64::MAX, 0xDEAD_BEEF_CAFE_F00D] {
        roundtrip(&EntityId(v));
    }
}

#[test]
fn from_name_deterministic_and_distinct() {
    assert_eq!(EntityId::from_name("floor"), EntityId::from_name("floor"));
    assert_ne!(EntityId::from_name("floor"), EntityId::from_name("wall"));
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
    let fake = EntityId(42);
    map.insert(fake, "name_a", 1).unwrap();
    let err = map.insert(fake, "name_b", 2).unwrap_err();
    assert!(matches!(err, DataError::IdCollision { .. }));
}

#[test]
fn id_map_detects_duplicate() {
    let mut map: IdMap<u32> = IdMap::new();
    map.insert_named("floor", 1).unwrap();
    let err = map.insert_named("floor", 2).unwrap_err();
    assert!(matches!(err, DataError::DuplicateId { .. }));
}
