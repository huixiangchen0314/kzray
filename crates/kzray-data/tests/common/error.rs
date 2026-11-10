//! 对应 `src/common/error.rs`。

use kzray_data::common::EntityId;
use kzray_data::DataError;

#[test]
fn id_collision_holds_names() {
    let e = DataError::IdCollision {
        new_name: "b".into(),
        existing_name: "a".into(),
        id: EntityId(7),
    };
    match e {
        DataError::IdCollision { new_name, existing_name, id } => {
            assert_eq!(new_name, "b");
            assert_eq!(existing_name, "a");
            assert_eq!(id, EntityId(7));
        }
        _ => panic!("期望 IdCollision"),
    }
}

#[test]
fn display_contains_id() {
    let e = DataError::DuplicateId { id: EntityId(42) };
    let s = format!("{e}");
    assert!(s.contains("42"), "Display 输出应含 id: {s}");
}
