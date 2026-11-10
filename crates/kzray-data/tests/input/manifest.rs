//! 对应 `src/input/manifest.rs`。

use kzray_data::common::{HasMeta, Meta, Param};
use kzray_data::input::{PartitionMeta, RenderManifest};

fn roundtrip<T>(v: &T) -> Vec<u8>
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let bytes = bincode::serialize(v).expect("serialize");
    let back: T = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(&back, v);
    let bytes2 = bincode::serialize(&back).expect("reserialize");
    assert_eq!(bytes, bytes2, "二次序列化字节不一致");
    bytes
}

#[test]
fn partition_meta_roundtrip() {
    let p = PartitionMeta {
        id: 0,
        region: [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
        prim_count: 1000,
        bvh_bytes: 65536,
        photon_bytes: 1_048_576,
    };
    roundtrip(&p);
}

#[test]
fn partition_meta_zero_photon() {
    // 纯几何分区，无光子图
    let p = PartitionMeta {
        id: 1,
        region: [[10.0, 0.0, 0.0], [20.0, 10.0, 10.0]],
        prim_count: 2000,
        bvh_bytes: 131_072,
        photon_bytes: 0,
    };
    assert_eq!(p.photon_bytes, 0);
    roundtrip(&p);
}

#[test]
fn manifest_no_top_level_bvh() {
    let m = RenderManifest {
        job_id: 42,
        partitions: vec![
            PartitionMeta {
                id: 0,
                region: [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
                prim_count: 1000,
                bvh_bytes: 65536,
                photon_bytes: 0,
            },
            PartitionMeta {
                id: 1,
                region: [[10.0, 0.0, 0.0], [20.0, 10.0, 10.0]],
                prim_count: 2000,
                bvh_bytes: 131_072,
                photon_bytes: 0,
            },
        ],
        top_level_bvh: None,
        meta: Meta::new(),
    };
    assert_eq!(m.partitions.len(), 2);
    roundtrip(&m);
}

#[test]
fn manifest_with_top_level_bvh() {
    let m = RenderManifest {
        job_id: 42,
        partitions: vec![PartitionMeta {
            id: 0,
            region: [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
            prim_count: 1000,
            bvh_bytes: 65536,
            photon_bytes: 0,
        }],
        top_level_bvh: Some(vec![0u8; 128]),
        meta: Meta::new(),
    };
    roundtrip(&m);
}

#[test]
fn manifest_carries_meta() {
    let m = RenderManifest {
        job_id: 1,
        partitions: vec![],
        top_level_bvh: None,
        meta: {
            let mut mm = Meta::new();
            mm.insert("total_samples", Param::from_i64(1000));
            mm.insert("render_layer", Param::from_str("beauty"));
            mm
        },
    };
    assert_eq!(m.meta().len(), 2);
    roundtrip(&m);
}

#[test]
fn empty_manifest() {
    let m = RenderManifest {
        job_id: 0,
        partitions: vec![],
        top_level_bvh: None,
        meta: Meta::new(),
    };
    roundtrip(&m);
}
