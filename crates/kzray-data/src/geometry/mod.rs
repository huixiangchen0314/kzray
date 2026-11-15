
//! 几何数据。属性层 + 三角化索引，与对象变换分离。

pub mod attribute;
pub mod curve;
pub mod mesh;
pub mod surface;
pub mod aabb;
pub mod shape;


pub use attribute::{AttributeBuffer, AttributeValues};
pub use mesh::{Mesh, SubMesh};
pub use surface::VisibleSurface;
pub use aabb::Aabb;

pub use shape::{Plane, Sphere};
