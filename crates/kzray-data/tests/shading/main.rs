//! `shading` 模块测试入口。子模块与 `src/shading/` 一一对应。

pub fn roundtrip<T>(v: &T) -> Vec<u8>
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

mod material;
