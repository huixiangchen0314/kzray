//! kzray-impl/src/accel.rs
//! 加速层：加速结构 + 图元 + 实例 + 资源表。

pub mod accelerator;
pub mod alpha;
pub mod animated;
pub mod brute_force;
pub mod instance;
pub mod resource;

// 延后接入，实现后取消注释：
// pub mod binned_bvh;
// pub mod mesh;

// ═══════════════════════════════════════════════════════════════
// 对外重导出
// ═══════════════════════════════════════════════════════════════
//
// 只重导出**上层 trace / occluded 调用点直接用到**的类型。
// 内部构造类型（ShapeInstance / AnimatedShapeInstance 的字段、
// lerp_affine、intersect_with_transform 等）通过模块路径访问，
// 不在 accel 根上暴露——避免类型爆炸和意外的依赖耦合。

// 加速结构接口与实现
pub use accelerator::{Accelerator, Bounded, Hit, Primitive};
pub use brute_force::BruteForce;

// 命中结果与 alpha（上层 shading 阶段消费）
pub use instance::InstanceHit;
pub use alpha::AlphaSource;

// 资源表（上层构建场景时构造，求交时透传）
pub use resource::ResourceTable;
