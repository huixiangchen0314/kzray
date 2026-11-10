
//! 几何数据。属性层 + 三角化索引，与对象变换分离。

pub mod attribute;
pub mod curve;
pub mod mesh;

pub use attribute::{AttributeBuffer, AttributeValues};
pub use mesh::{Mesh, SubMesh};
