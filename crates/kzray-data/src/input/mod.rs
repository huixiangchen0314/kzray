//! 渲染输入的组织形态。
//! - `instance`：可渲染实例
//! - `render_input`：单机完整输入
//! - `manifest`：分布式分区视图

pub mod instance;
pub mod manifest;
pub mod render_input;

pub use instance::{MeshInstance, VisibilityFlags, };
pub use manifest::{PartitionMeta, RenderManifest};
pub use render_input::RenderInput;
