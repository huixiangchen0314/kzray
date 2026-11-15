//! 积分器契约。
//!
//! 积分器是渲染器的**算法核心**——给定场景、光线、采样器，计算沿光线
//! 到达的辐射亮度。渲染方程的全部求解逻辑（路径追踪、BDPT、光子映射）
//! 都在这一层。
//!
//! # 为什么是窄接口
//!
//! 契约层只定义**入口**，不定义**过程**。`li` 一个方法足够：
//!
//! - **算法差异全在实现层**：PT 的路径状态机、BDPT 的子路径连接、
//!   PM 的光子映射——这些是算法，不是契约。
//! - **契约层不假设算法结构**：如果契约层定义 `sample_path` / `connect` /
//!   `mnee` 之类的方法，就等于把某种算法结构强加给所有实现。
//! - **配置是实现的属性**：最大深度、RR 阈值、光源采样策略、MIS 权重——
//!   这些是积分器的构造参数，不是 `li` 的输入。
//!
//! 对比 `Light` / `BSDF` / `Camera`：它们的契约有多个方法，因为它们的
//! 行为是**协议的多个面**（采样 / 求值 / PDF），而积分器的行为是
//! **一个整体计算**——给定输入，返回输出，中间的算法过程不对外暴露。
//!
//! # 支持范围
//!
//! 本契约服务于"**每次调用独立**"的积分器：
//!
//! - 路径追踪（PT）：每次 `li` 调用追踪一条完整路径；
//! - 双向路径追踪（BDPT）：每次 `li` 调用构造 eye subpath，内部与
//!   light subpath 连接（light subpath 是 `li` 的内部实现，不是独立入口）；
//! - 光子映射（PM）的辐射亮度估计：从场景点出发的光线追踪。
//!
//! **MLT 不直接实现本契约**。马尔可夫链需要跨调用状态（当前路径、
//! 突变策略、接受概率），而 `li` 是 `&self`、每次调用独立的纯函数。
//! MLT 用独立接口驱动：
//!
//! - `MltIntegrator::run_chain(&mut self, ...)` 维护链式状态，内部每一步
//!   用 `Scene` / `Light` / `BSDF` 契约组件计算路径贡献；
//! - `Integrator` 契约本身**不服务于** MLT——这是算法差异，不是契约层
//!   的职责。文档不把 MLT 列入"本契约覆盖的算法"。
//!
//! # 泛型参数
//!
//! - `S: SpectralAlgebra`：光谱类型；
//! - `LS: LightSampler`：光源采样器类型。与 [`crate::scene::Scene`] 的
//!   `LS` 一致——`LightSampler` 不可 dyn，所以通过泛型参数传递。
//!
//! 积分器本身是 **dyn compatible** 的：`li` 不是泛型方法，`S` / `LS` 是
//! trait 参数，`Box<dyn Integrator<S, LS>>` 可用。这允许实现层在运行时
//! 切换积分器（如 `--integrator pt` / `--integrator bdpt`）。
//!
//! # 为什么 `LS` 是泛型参数而非 `Box<dyn LightSampler>`
//!
//! `LightSampler::sample` 用了 `&mut impl Sampler` 换单态化，
//! `LightSampler` 因此 **不是 dyn compatible**（`&dyn LightSampler`
//! 不可用）。两个选择：
//!
//! 1. **改 `sample` 为 `&mut dyn Sampler`**：`LightSampler` 变 object-safe，
//!    `Scene` / `Integrator` 不需要 `LS` 泛型参数。代价是 `sample` 内部的
//!    `next_2d` / `next_f32` 走虚分派。
//! 2. **`LS` 泛型参数**（当前选择）：保留 `LightSampler` 的完整单态化，
//!    代价是 `Scene` / `Integrator` 多一个泛型参数。
//!
//! 当前选 2 是保守决策：单态化保证零开销，泛型参数的成本很低（只影响
//! 类型签名，不影响运行时）。如果**实测**显示虚分派可忽略（每次 `sample`
//! 内部的 2-3 次虚分派相比 BVH 遍历微不足道），可以改成方案 1——这是
//! **可逆决策**，改签名即可，不影响其他契约。
//!
//! # `&dyn Scene<S, LS>` 而非 `&impl Scene<S, LS>`
//!
//! `Integrator` 可能被 `Box<dyn Integrator<S, LS>>` 持有（运行时切换）。
//! 如果 `li` 接收 `&impl Scene<S, LS>`，`Integrator` 就不再 dyn compatible
//! ——每个 `Scene` 实现会单态化出一份 `Integrator`，`Box<dyn>` 无法表达。
//!
//! 代价是 `scene.intersect` / `scene.material` 等调用走虚分派。每次求交
//! 一次虚分派，相比 BVH 遍历（O(log n) 次盒求交）**理论上可忽略**，但
//! **具体数字需实测**。如果 profile 显示 `scene.intersect` 的虚分派是
//! 瓶颈，可以考虑在 `Integrator` 内部缓存热点数据（如最近的命中、当前
//! 的材质），减少虚分派次数。
//!
//! 这与 [`crate::scene::Scene`] 本身是 `&dyn Scene` 的取舍一致——`Scene`
//! 的方法不是泛型方法，所以 `Scene` 可以 dyn；`LightSampler` 的 `sample`
//! 用了 `&mut impl Sampler`，所以 `LightSampler` 不可 dyn，通过 `LS`
//! 泛型参数传递。
//!
//! # `Send + Sync` 与内部状态
//!
//! `Integrator` 是**配置对象**——跨线程共享，每个线程用自己的 `Sampler`
//! 独立调用 `li`。内部**不应**有跨调用的可变状态。需要缓存的算法
//! （光子图、预计算表）在构造期初始化（不可变），运行时只读；如果
//! 需要运行时更新，用 `Mutex` / `RwLock` 自行承担同步成本，或使用
//! 独立契约（如 `MltIntegrator`）。
//!
//! 这是"配置与状态分离"的设计——配置在积分器实例里（不可变或构造期
//! 初始化），状态在调用栈或显式管理的容器里。`Send + Sync` 强制这个
//! 分离，避免"看似无状态实则内部可变"的隐藏 bug。
//!
//! # 维度管理
//!
//! `li` 接收 `&mut dyn Sampler`，内部按**维度追踪**模型推进基址：
//!
//! ```text
//! → LightSampler::sample(ctx, sampler)      消耗 c1 维
//! → sampler.next_2d(dim1) 取 2D 样本       消耗 2 维
//! → BSDF::sample_f(sampler, dim2, mode)     消耗 c2 维
//! → sampler.next_f32(dim3) 取 RR 样本       消耗 1 维
//! ```
//!
//! 具体流程见 [`crate::conventions`] 的"维度流程"文档。积分器实现负责
//! 按这个流程推进，契约层不约束内部顺序——BDPT 的维度流程和 PT 不同，
//! 但都遵守"追踪实际消耗"的原则。
//!
//! 实现层按 `conventions` 的流程推进基址。`LightSampler::max_sample_dimensions`
//! 和 `BSDF::max_sample_dimensions` 提供失败恢复的**上界**；正常路径用
//! `consumed` **精确追踪**。契约层不提供统一的维度查询接口——各组件
//! 自己报告消耗是维度追踪模型的核心。
//!
//! # 与 `Camera` 的关系
//!
//! `Camera` 生成 primary ray，`Integrator::li` 消费它。积分器**不持有**
//! `Camera`——相机由积分器外部的调用方持有，`li` 只接收已生成的 ray。
//! 这是"数据流"的设计：相机 → ray → 积分器 → 辐射亮度 → 胶片。
//!
//! BDPT 的相机侧子路径由积分器内部调用 `Camera::sample_wi`（如果实现了
//! `SampleWiCamera`），但那是积分器的实现细节，不是契约层的输入。
//!
//! # 与 `Film` 的关系
//!
//! 积分器**不直接写 Film**。`li` 返回辐射亮度 `S`，调用方（渲染循环）
//! 把它累加到 `Film::add_sample`。这样积分器不需要知道 Film 的存在，
//! Film 也不需要知道积分器的算法——两者通过 `S` 解耦。
//!
//! 唯一的例外是 BDPT / PM 的 splat：`Film::splat` 由积分器在特定路径
//! 下调用（如光子映射的光子沉积）。这是实现层的选择，契约层不约束
//! ——`Integrator` 的实现可以持有 `&dyn Film`，也可以不持有。

use crate::AreaLight;
use crate::light::{InfiniteLight, Light, LightSampler};
use crate::ray::Ray;
use crate::sampler::Sampler;
use crate::scene::Scene;
use crate::spectrum::SpectralAlgebra;

// ═══════════════════════════════════════════════════════════════
// Integrator
// ═══════════════════════════════════════════════════════════════

/// 积分器契约。
///
/// 给定场景、光线、采样器、波长上下文，计算沿光线到达的辐射亮度。
/// 这是渲染方程的**唯一入口**——所有算法差异在实现层。
///
/// # 泛型参数
///
/// - `S: SpectralAlgebra`：光谱类型；
/// - `LS: LightSampler`：光源采样器类型，与 `Scene<S, LS>` 一致。
///
/// # Object-safety
///
/// `Integrator<S, LS>` 是 object-safe 的：`li` 不是泛型方法。
/// `Box<dyn Integrator<S, LS>>` / `Arc<dyn Integrator<S, LS>>` 可用。
///
/// # 生命周期
///
/// `li` 不持有 `scene` / `ray` / `sampler` 的引用，只读/借用它们。
/// 积分器内部可以创建临时状态（路径栈、子路径、MIS 权重数组），
/// 但生命周期不超出 `li` 的调用。
pub trait Integrator<S, LS, SC, L, IL, AL>: Send + Sync
where
    S: SpectralAlgebra,
    LS: LightSampler,
    SC: Scene<S, LS, L, IL, AL>,
    L: Light<S>,
    IL: InfiniteLight<S>,
    AL: AreaLight<S>,
{
    /// 计算沿 `ray` 到达的辐射亮度。
    ///
    /// 参数：
    /// - `scene`：场景只读视图。求交、查材质、查光源都通过它；
    /// - `ray`：光线。`ray.time` 已由相机填充；`ray.origin` 和
    ///   `ray.dir` 是相机生成的 primary ray，或 BDPT 的相机侧子路径；
    /// - `sampler`：采样器。积分器内部按维度追踪推进基址；
    /// - `wavelengths`：波长上下文。只读。
    ///
    /// 返回沿光线到达的辐射亮度 `S`。调用方把它累加到 `Film`。
    ///
    /// # 语义
    ///
    /// - **从相机出发**（`Radiance` 传输方向）：返回沿 `ray.dir` 看到的
    ///   **总辐射亮度**，**包含**自发光贡献。
    /// - **自发光与双重计数**：实现层通过 `specular_bounce` 标志（或
    ///   等价机制）决定何时累加自发光：
    ///   - **第一次反弹**（`bounces == 0`）或**上一次是镜面反弹**
    ///     （`specular_bounce == true`）：必须累加自发光——NEE 覆盖不到
    ///     镜面方向，光源命中是唯一的贡献路径；
    ///   - **漫反射反弹后**：跳过自发光——NEE 已采样同一光源，再累加会
    ///     双重计数。
    ///
    ///   实现层负责避免双重计数。契约层不强制具体机制，但语义要求
    ///   **镜面反射后的光源命中必须返回自发光**——否则镜面材质在无 NEE
    ///   覆盖的方向上会丢失能量。
    /// - **不乘相机权重**：`W_e` 由相机侧处理（`CameraSample::weight`），
    ///   `li` 返回的是场景辐射亮度，调用方乘 `W_e` 得到最终贡献。
    /// - **BDPT 的 endpoint 连接**：BDPT 在连接两端时会加入自发光和
    ///   BSDF 贡献，这是算法内部的处理，不改变 `li` 的语义——`li` 返回
    ///   的是"沿 ray 到达的辐射亮度"，BDPT 的连接逻辑在这个框架内计算。
    ///
    /// # 失败
    ///
    /// 返回 `S::zero()` 表示无贡献（如光线逃逸到无环境光的无限远、
    /// 路径被 RR 终止、采样失败）。`li` 不返回 `Option`——零贡献是
    /// 合法结果，调用方统一累加。
    ///
    /// # 维度
    ///
    /// `sampler` 的维度基址由调用方设置（`start_pixel_sample`）。`li`
    /// 内部按维度追踪推进，返回后调用方不需要知道消耗了多少维度——
    /// 下一次 `li` 调用前会重新 `start_pixel_sample`。
    ///
    /// # 线程安全
    ///
    /// `li` 是 `&self`，多线程可以并行调用。实现层不应在 `li` 内部修改
    /// `self` 的可变状态——需要缓存的算法（光子图、预计算表）在构造期
    /// 初始化，运行时只读；需要运行时更新的状态用 `Mutex` / `RwLock`
    /// 保护或使用独立契约（见模块文档"`Send + Sync` 与内部状态"）。
    fn li<SM: Sampler>(
        &self,
        scene: &SC,
        ray: &Ray,
        sampler: &mut SM,
        start_dim: u32,
        wavelengths: &S::WavelengthContext,
    ) -> S;
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════
//
// 契约层测试只验证**接口形态**：
//
// - 返回值正确性（`ConstantIntegrator` / `ZeroIntegrator`）；
// - object-safety（`Box<dyn Integrator<S, LS>>`）；
// - `Send + Sync`；
// - `&dyn Scene` 参数兼容性。
//
// **不验证**以下语义（需要在实现层用真实场景测试）：
//
// - **自发光语义**：需要镜面材质 + 无 NEE 的场景（如镜面天花板），
//   验证 `li` 在镜面反弹后返回自发光。契约层无场景，无法测。
// - **双重计数避免**：需要漫反射材质 + 光源的场景，验证 NEE 和自发光
//   不重复累加。契约层无光源采样实现，无法测。
// - **维度管理**：需要真实 `Sampler`（Sobol / Halton）验证维度基址
//   推进正确。契约层的 `TestSampler` 只验证签名，不追踪维度。
// - **能量守恒**：需要完整 PT 管线和白炉测试。契约层不涉及物理。
//
// 这些缺口是契约层测试的合理边界——契约验证"签名和 object-safety"，
// 物理语义由实现层测试。

