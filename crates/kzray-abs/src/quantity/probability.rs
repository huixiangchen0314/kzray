//! kzray-abs/src/quantity/probability.rs

use std::marker::PhantomData;

/// 测度标记：仅用于编译期区分，零大小
pub trait Measure {}

/// 立体角测度（pdf 对 ω 积分）
pub struct SolidAngle;
impl Measure for SolidAngle {}

/// 面积测度（pdf 对表面积积分，如 light sampling）
pub struct Area;
impl Measure for Area {}

/// 路径空间测度：pdf 是路径空间上的联合密度（面积^n / 体积^n）
/// 用于需要跨策略 MIS 比较、做测度转换的场景
pub struct PathSpace;
impl Measure for PathSpace {}

/// 随机游走测度：pdf 是随机游走过程产出的数值
/// 用于内部传递、不跨测度比较的场景
pub struct RandomWalk;
impl Measure for RandomWalk {}

/// δ 分布标记（镜面/点光，pdf 无普通密度值）
pub struct Delta {}

/// 连续密度：关于测度 M
pub struct Pdf<M: Measure>(pub f32, PhantomData<M>);

impl<M: Measure> Pdf<M> {
    pub fn new(v: f32) -> Self { Pdf(v, PhantomData) }
}

/// MIS 里一个采样策略的 pdf：要么 δ，要么普通密度
pub enum MisPdf<M: Measure> {
    Delta,
    Pdf(Pdf<M>),
}
