//! 类型级物理量区分


pub struct Meters(pub f32);
pub struct Radians(pub f32);
pub struct Steradians(pub f32);

pub struct Pdf<M>(pub f32, PhantomData<M>);  // 连续密度，M 指定测度
pub struct Pmf(pub f32);                     // 离散质量
pub struct Weight(pub f32);                  // MIS / BSDF 权重
