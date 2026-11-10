//! 渲染器数据协议层。
//!
//! 红线（不要违反）：
//! - 不依赖任何执行层 crate（rayon / rand / jni / image）
//! - 不出现 `Box<dyn ...>`（trait object 属于执行层）
//! - 所有类型必须 `Serialize + Deserialize` 且往返字节确定
//!
//! 分层：
//! - `common`：跨域基础类型（id / value / node / field / rect / transform / error）
//! - `geometry`：几何数据（attribute / mesh）
//! - `shading`：材质（material，本质是对 BSDF Field 的引用）
//! - `light` / `camera` / `world` / `config` / `input` / `task`：阶段 2 后续实现
//!
//! 序列化:
//! **协议铁律**：`Serialize` / `Deserialize` 的结构体不得使用
//! `#[serde(skip_serializing_if)]`。bincode 是非自描述格式，字段省略会导致
//! 反序列化字节错位（`UnexpectedEof`）。字段永远序列化，空值用 0 / 空 Vec。

pub mod camera;
pub mod common;
pub mod config;
pub mod geometry;
pub mod input;
pub mod light;
pub mod shading;
pub mod task;
pub mod world;
pub mod volume;
// ── common 重导出 ──

pub use common::{
    DataError, EntityId, FieldExpr, FieldId, MeshRef, HasMeta, IdMap, LightRef, MaterialRef,
    Meta, Node, NodeId, NodeInput, NodeOutput, Param, Rect, TextureId, Transform, Value, VolumeRef,
};
pub use volume::{VolumeBounds, VolumeData, VolumeInstance};
pub use world::WorldData;

// ── geometry 重导出 ──

pub use geometry::{AttributeBuffer, AttributeValues, Mesh, SubMesh};

// ── shading 重导出 ──

pub use shading::Material;
