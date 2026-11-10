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

pub mod camera;
pub mod common;
pub mod config;
pub mod geometry;
pub mod input;
pub mod light;
pub mod shading;
pub mod task;
pub mod world;

// ── common 重导出 ──

pub use common::{
    DataError, EntityId, FieldExpr, FieldId, GeometryRef, IdMap, LightRef, MaterialRef, Node,
    NodeId, NodeInput, NodeOutput, Rect, TextureId, Transform, Value, VolumeRef,
};

// ── geometry 重导出 ──

pub use geometry::{AttributeBuffer, AttributeValues, Mesh, SubMesh};

// ── shading 重导出 ──

pub use shading::Material;
