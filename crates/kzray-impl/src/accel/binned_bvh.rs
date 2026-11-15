
use crate::accel::accelerator::Primitive;

pub struct BinnedBvh{
    /// SOA AABB [min-x, min-y, min-z, max-x, max-y, max-z]
    pub aabbs: Vec<f32>,
    /// 图元, 可能是实例图元，也可能是网格三角形图元
    pub prims: Vec<Primitive>,
}

