//! 跨域基础类型。`common` 不依赖任何其他模块。

pub mod error;
pub mod field;
pub mod id;
pub mod meta;
pub mod node;
pub mod param;
pub mod rect;
pub mod transform;
pub mod value;

pub use error::DataError;
pub use field::{FieldExpr, FieldId};
pub use id::{
    EntityId, MeshRef, IdMap, LightRef, MaterialRef,  VolumeRef, NodeId, TextureId, 
};
pub use meta::{HasMeta, Meta};
pub use node::{Node, NodeInput, NodeOutput};
pub use param::Param;
pub use rect::Rect;
pub use transform::Transform;
pub use value::Value;
