use crate::{Library, ray::Ray};
use crate::sampler::Sampler;
use crate::spectrum::SpectralAlgebra;
use glam::{Vec2, Vec3};
use kzray_data::LightId;

// ═══════════════════════════════════════════════════════════════
// 光源类型
// ═══════════════════════════════════════════════════════════════

/// 光源类型。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightType {
    /// 位置上成 Delta 分布的光源（如点光源）。
    DeltaPosition,
    /// 方向上成 Delta 分布的光源（如方向光/平行光）。
    DeltaDirection,
    /// 面光，绑定到网格。
    /// 注意，我们不索引网格顶点；
    /// 如果面光源引用的是网格的一部分，必须在构建层拆分成独立网格数据项。
    Area,
    /// 无限远的光源（如环境光/天空光）。
    Infinite,
}

impl LightType {
    /// 是否是 delta 光源。delta 光源不能和 BSDF 采样做 MIS。
    #[inline]
    pub fn is_delta(self) -> bool {
        matches!(self, LightType::DeltaPosition | LightType::DeltaDirection)
    }
}

// ═══════════════════════════════════════════════════════════════
// 着色点上下文
// ═══════════════════════════════════════════════════════════════

/// 光源采样上下文。
///
/// 携带采样光源时需要的**所有可能用得上的信息**——位置、法线、
/// 资产库。实现按需读取：
///
/// - `PointLight` / `ShapeAreaLight`：只用 `position` / `normal`，
///   忽略 `library`；
/// - 未来 `MeshAreaLight`：从 `library.expect_mesh(mesh_id)` 取顶点。
///
/// 把 `library` 放这里而非方法参数，是为了**接口稳定性**——
/// 未来新增 `MeshAreaLight` 不改 `Light::sample_li` 签名。
///
/// # 生命周期
///
/// `'a` 是 `Library<S>` 的借用生命周期，与 `Scene` 一致。`Copy`
/// 保留——`&'a Library<S>` 是 `Copy`，整个结构仍是 `Copy`。
#[derive(Clone, Copy)]
pub struct LightSampleContext<'a, S: SpectralAlgebra> {
    /// 着色点位置（世界空间）。
    pub position: Vec3,
    /// 着色点法线（世界空间，单位向量）。
    pub normal: Vec3,
    /// 资产库。当前所有实现可忽略；未来 mesh 面积光从这里取 mesh。
    pub library: &'a Library<S>,
}

// ═══════════════════════════════════════════════════════════════
// 采样几何
// ═══════════════════════════════════════════════════════════════

/// `LiSample` 使用的几何形态。
///
/// 区分"有具体光源点"和"只有方向"，避免无限远光源的采样结果被误用于
/// 阴影射线或面积↔立体角测度转换。
#[derive(Clone, Copy, Debug)]
pub enum LiGeometry {
    /// 面光源 / 点光源：有具体光源点与到着色点的有限距离。
    Surface {
        /// 光源上被采样的点（世界坐标）。
        position: Vec3,
        /// 着色点到光源点的距离。
        ///
        /// 应等于 `(position - ctx.position).length()`；实现时必须保持一致，
        /// 否则阴影射线 `t_max` 和面积↔立体角测度转换会出错。
        /// 跨实现测试应校验此等式。
        distance: f32,
    },
    /// 方向光 / 无限远光源：只有方向，`distance` 无定义。
    ///
    /// 调用方不应在此分支做阴影射线距离截断或面积↔立体角转换。
    Direction,
}

/// `LeSample` 使用的几何形态。
///
/// 光源侧采样点可能是具体表面点（面光源 / 点光源），也可能只有方向
/// （无限远光源）。与 [`LiGeometry`] 的区别：光源侧没有"到着色点的
/// 距离"参照点，但需要表面法线与 UV，供 BDPT 继续 trace 或做端点连接。
#[derive(Clone, Copy, Debug)]
pub enum LeGeometry {
    /// 面光源 / 点光源：有具体光源表面点。
    Surface {
        /// 光源表面被采样的点（世界坐标）。
        position: Vec3,
        /// 光源表面在该点的法线。
        normal: Vec3,
        /// 光源表面在该点的 UV。
        uv: Vec2,
    },
    /// 方向光 / 无限远光源：只有方向。
    Direction,
}

// ═══════════════════════════════════════════════════════════════
// 采样结果
// ═══════════════════════════════════════════════════════════════

/// 单次光源采样的结果。
///
/// 对应渲染方程中的入射项 `L_i`：
///
/// ```text
/// L_o(p, wo) = L_e(p, wo)
///            + ∫_Ω f(p, wi, wo) · L_i(p, wi) · cos(θ_i) dω_i
/// ```
///
/// 字段与方程项的对应（命名来源）：
///
/// - `incident` 对应 `L_i(p, wi)`，即从光源点沿 `-wi` 方向到达着色点的
///   入射辐射亮度。字段名取自方程符号 `L_i` 的语义读法 "incident radiance"，
///   避免 `li` 与 `LeSample::emitted` 只差一字母的视觉混淆。
/// - `wi` 对应 `ω_i`，从着色点指向光源点的单位入射方向。保留渲染方程
///   符号的 ASCII 转写，与 `pdf_li(ctx, wi)` 参数、文档 `ω_i` 三处一致。
/// - `pdf` 是光源内部采样该入射方向的概率密度（**立体角测度**），不包含
///   "选择该光源"的概率；总 PDF 由调用方把 `light_pdf * pdf` 得到。
/// - `geometry` 是光源一侧的采样几何，用于阴影射线、可见性判断，
///   以及面积测度到立体角测度的转换；无限远光源为
///   [`LiGeometry::Direction`]。
///
/// # 测度
///
/// `pdf` 是**立体角测度**（从着色点看光源方向的测度）。与 BSDF 的
/// `BxDFSample::pdf` 同测度、同参考点（均为着色点），可以直接在 MIS
/// 权重中相加。如果实现内部在面积测度上采样，返回前必须转换成立体角
/// 测度：`pdf_omega = pdf_area * distance^2 / |cos(theta_light)|`。
///
/// delta 光源的 `pdf` 无有限值，调用方不应读取此字段参与 MIS。
///
/// # delta 判断
///
/// `LiSample` 没有 `flags` 字段。调用方判断本次采样是否命中 delta 光源，
/// 应通过 `LightSelection::light_id` 查 `LightType::is_delta()`。
/// **不要通过 `pdf == 0.0` 反推**——非 delta 光源在退化方向也可能返回
/// 接近 0 的 PDF，两者语义不同。
///
/// # 可分解性
///
/// `LiSample` 没有 `weight` 字段。`incident` 是光源的入射辐射亮度
/// `L_i`（**不是** BSDF 值），不含余弦项、不含 PDF。光源采样路径的
/// 完整贡献是：
///
/// ```text
/// throughput *= f × incident × cos_theta_i / total_pdf
/// ```
///
/// 每个因子的来源：
///
/// | 因子 | 来源 | 说明 |
/// |---|---|---|
/// | `f` | `BSDF::eval(wi, mode)` | 着色点的 BSDF 值，由调用方从 BSDF 模块取 |
/// | `incident` | 本结构体字段 | 光源的入射辐射亮度 `L_i` |
/// | `cos_theta_i` | `ctx.normal · wi` | 着色点几何项，由调用方计算 |
/// | `total_pdf` | `light_pdf × LiSample::pdf` | 选择光源的概率 × 光源内部采样 PDF |
///
/// 也就是说，`LiSample` 只提供 `incident` 和 `pdf` 两个因子；`f` 来自
/// BSDF，`cos` 来自着色点几何，`light_pdf` 来自 [`LightSelection`]。
/// 四者相乘除以总 PDF 才是这一步的直接光照贡献。
///
/// 完整调用链示例（非 delta 光源，参与 MIS）：
///
/// ```text
/// // 调用方已持有：
/// // - ctx: LightSampleContext（着色点位置与法线）
/// // - light: &L（L: Light 的具体实现）
/// // - light_selection: LightSelection（LightSampler::sample 的结果）
/// // - sample_2d: Vec2（调用方预取的 2D 样本）
/// // - bsdf: BSDF（着色点的 BSDF）
/// // - wl: S::WavelengthContext
///
/// let li = light.sample_li(&ctx, sample_2d, wl)?;
///
/// // 光源采样给出 incident 和 pdf
/// let incident = li.incident;      // L_i
/// let pdf_li = li.pdf;             // 立体角测度
///
/// // 调用方从 BSDF 取 f
/// let f = bsdf.eval(li.wi, TransportMode::Radiance);
///
/// // 调用方从着色点几何算 cos
/// let cos_theta_i = ctx.normal.dot(li.wi).max(0.0);
///
/// // 总 PDF = 选择该光源的概率 × 光源内部采样 PDF
/// let total_pdf = light_selection.light_pdf * pdf_li;
///
/// // 蒙特卡洛估计量
/// throughput *= f * incident * cos_theta_i / total_pdf;
/// ```
///
/// 如果是 delta 光源（`light_type().is_delta() == true`），`pdf_li`
/// 无有限值，不参与 MIS。调用方直接用"采样命中"的贡献：
///
/// ```text
/// let li = light.sample_li(&ctx, sample_2d, wl)?;
/// let f = bsdf.eval(li.wi, TransportMode::Radiance);
/// let cos_theta_i = ctx.normal.dot(li.wi).max(0.0);
/// // delta 光源：light_pdf 作为离散概率，无连续 PDF
/// throughput *= f * incident * cos_theta_i / light_selection.light_pdf;
/// ```
///
/// 与 `BxDFSample::weight` 的对比：`BxDFSample` 直接返回
/// `weight = value × |cos| / pdf`，调用方乘 throughput 即可；但
/// `pdf` 和 `value` 的可分解性只在 `value` 字段上保留。`LiSample`
/// 不封装 `weight`，让所有因子显式可见，原因有二：
///
/// - **`cos` 的归属不同**：`BxDFSample` 的 `cos` 是 BSDF 半球积分的
///   固有部分，BSDF 自己知道 `wi` 与法线的关系；`LiSample` 的
///   `cos_theta_i` 是**着色点**的几何项，光源不知道着色点法线朝哪，
///   塞进 `LiSample` 会让"光源采样"和"着色点几何"耦合。
/// - **MIS 需要分别访问**：光源采样与 BSDF 采样的 MIS 权重计算需要
///   `pdf` 和 `incident` 的独立值；提前乘成 `weight` 会丢失可分解性。
#[derive(Clone, Copy, Debug)]
pub struct LiSample<S> {
    /// 光源一侧的采样几何：`Surface` 有具体点与距离，`Direction` 没有。
    pub geometry: LiGeometry,
    /// 从着色点指向光源点的单位方向，对应渲染方程中的 `ω_i`。
    ///
    /// 命名来源：渲染方程入射方向符号 `ω_i` 的 ASCII 转写。
    pub wi: Vec3,
    /// 入射辐射亮度，对应渲染方程中的 `L_i(p, wi)`。
    ///
    /// 命名来源：方程符号 `L_i` 的语义读法 "incident radiance"。
    ///
    /// 它乘上 BSDF `f`、余弦项 `cos(θ_i)` 和蒙特卡洛权重后，
    /// 贡献到渲染方程的反射积分项。
    pub incident: S,
    /// 光源内部采样 PDF，**立体角测度**。
    pub pdf: f32,
}

/// 光源侧方向采样结果（BDPT / 光子映射 / MLT 使用）。
///
/// 与 `LiSample` 的区别：`LiSample` 从着色点出发看向光源，`LeSample`
/// 从光源出发射向场景，方向约定相反。`LeSample` 额外携带光源表面点的
/// 法线和 UV，供 BDPT 继续 trace 或做端点连接；无限远光源没有表面点，
/// 对应 [`LeGeometry::Direction`]。
///
/// 对应渲染方程中的发射项 `L_e`：
///
/// ```text
/// L_o(p, wo) = L_e(p, wo) + ∫_Ω ...
/// ```
///
/// # 传输方向（隐含 Importance）
///
/// 本方法族的传输方向**隐含为 `Importance`**（从光源出发，追踪重要性），
/// 因此没有 `TransportMode` 参数。对比：
///
/// - `Light::sample_li` / `pdf_li` / `AreaLight::emitted_radiance` 隐含
///   `Radiance`（从着色点看向光源）；
/// - `InfiniteLight::sample_le` / `pdf_le` / `emitted_radiance` 隐含
///   `Importance`（从光源射向场景）；
/// - `BxDF` 方法全部显式接收 `TransportMode`。
///
/// 光源侧不需要 `mode`，因为"光源发射什么"是物理量，与"从哪边看"
/// 无关。BDPT 连接两端时，光源侧用 `Importance`，相机侧用 `Radiance`，
/// 中间通过 BSDF 的 `mode` 参数切换。
///
/// # 测度
///
/// `pdf` 是**立体角测度**（从光源点射出方向的测度）。对于面光源，
/// 光源侧采样通常先在面积上采点（`pdf_area`），再在方向半球上采样
/// 方向（`pdf_dir`）；`LeSample::pdf` 只报告方向部分的立体角测度，
/// 面积部分的测度由调用方从 `LeGeometry::Surface` 的采样结构另行处理。
///
/// 与 `LiSample::pdf` 的对比：两者都是立体角测度，但参考系不同
/// （`LiSample::pdf` 从着色点看，`LeSample::pdf` 从光源点看）。BDPT
/// 做端点连接时，需要把光源侧方向测度转换到连接方向的测度，转换因子
/// 由几何项（距离平方、余弦项）提供。
#[derive(Clone, Copy, Debug)]
pub struct LeSample<S> {
    /// 光源侧采样几何：`Surface` 有具体表面点，`Direction` 没有。
    pub geometry: LeGeometry,
    /// 从光源出发、射向场景的方向。
    ///
    /// 与 [`InfiniteLight::emitted_radiance`] 求值时的 `ray.direction`
    /// **反向**：
    ///
    /// - `emitted_radiance(ray)` 的 `ray` 从着色点出发，`ray.direction` 是
    ///   视线方向（从着色点射向光源），返回沿该方向到达着色点的辐射亮度。
    /// - `LeSample::direction` 从光源出发射向场景，是 `ray.direction` 的
    ///   反向（即 `-ray.direction`）。
    ///
    /// 两者反向的原因是光源侧采样和着色点侧求值的参考点不同：
    /// `emitted_radiance` 以着色点为原点问"朝这个方向看到什么"，
    /// `sample_le` 以光源为原点问"光源朝这个方向发射什么"。
    ///
    /// 该方向无对应渲染方程符号（方程中 `ω_o` 是着色点出射方向），
    /// 因此使用工程化名称 `direction` 而非符号转写。
    pub direction: Vec3,
    /// 发射辐射亮度，对应渲染方程中的 `L_e`。
    ///
    /// 命名来源：方程符号 `L_e` 的语义读法 "emitted radiance"。
    pub emitted: S,
    /// 方向测度 PDF，必须与 `pdf_le` 来自同一采样分布。
    pub pdf: f32,
}

// ═══════════════════════════════════════════════════════════════
// 光源能力 trait
// ═══════════════════════════════════════════════════════════════

/// 所有光源共享的采样契约。
///
/// `LightSampler` 只负责"在多个光源之间选一个"；本 trait 负责
/// "给定一个光源，采样它的点/方向、计算 PDF、给出总功率"。
///
/// 面光源求值见 [`AreaLight::emitted_radiance`]，无限远光源求值与
/// 光源侧采样（`sample_le` / `pdf_le` / `emitted_radiance`）见
/// [`InfiniteLight`]。拆分 trait 的目的是让"不适用的能力"在类型层面
/// 不存在，而不是用 `S::zero()` 或 `unreachable!()` 兜底。具体某个
/// `Light` 实例具备哪些能力，由实现层自行枚举分派，抽象层不做类型枚举。
///
/// 波长上下文由 `S::WavelengthContext` 提供：
/// - `Rgb` 配 `NoWavelengths`，零大小；
/// - `Spectrum<N>` 配 `SampledWavelengths<N>`，N 强绑定。
///
/// # 维度管理
///
/// 本 trait 的所有采样方法（`sample_li` / `sample_le`）接收**调用方预取
/// 的 `Vec2` 样本**，不管理 `Sampler` 状态、不消耗维度。维度消耗由
/// 调用方通过 `sampler.next_2d(dimension)` 显式完成。这与
/// [`LightSampler::sample`] 接收 `&mut impl Sampler`、内部消耗维度的
/// 模型不同——分工见 [`LightSampler::sample`] 的文档。
///
/// # PDF 一致性契约
///
/// 对每个 `Light` 实现，必须满足：
///
/// - **`sample_li` / `pdf_li` 一致性**：对任意
///   `s = sample_li(ctx, sample, wavelengths)`，应有
///   `pdf_li(ctx, s.wi) == s.pdf`（delta 光源除外，两者均为 0）。
/// - **`sample_li` / `sample_le` 一致性**（仅 [`InfiniteLight`]）：
///   对同一方向 `d`，`sample_li` 返回的 `incident` / `pdf` 必须与
///   `sample_le` 返回的 `emitted` / `pdf` 在 `wi = -d` 上一致；
///   推荐复用 [`InfiniteLight::sample_li_via_le`]。
/// - **`sample_le` / `pdf_le` 一致性**（仅 [`InfiniteLight`]）：
///   对任意 `s = sample_le(sample, wavelengths)`，应有
///   `pdf_le(s.direction, wavelengths) == s.pdf`（delta 光源除外）。
///
/// 跨实现测试必须校验上述等式；MIS 权重的正确性依赖它们。
pub trait Light<S: SpectralAlgebra>: Send + Sync {
    /// 光源类型。
    ///
    /// 与能力 trait 的契约（实现层测试应校验）：
    /// - 实现 [`AreaLight`] ⇒ 返回 [`LightType::Area`]；
    /// - 实现 [`InfiniteLight`] 且非 delta ⇒ 返回 [`LightType::Infinite`]；
    /// - 实现 [`InfiniteLight`] 且 delta（方向光）⇒ 返回
    ///   [`LightType::DeltaDirection`]；
    /// - 点光源 ⇒ 返回 [`LightType::DeltaPosition`]；
    /// - 一个类型不应同时实现 [`AreaLight`] 和 [`InfiniteLight`]。
    fn light_type(&self) -> LightType;

    /// 采样该光源的一个点/方向。
    ///
    /// 返回 `None` 表示采样失败（如方向在表面背面、距离退化、样本无效）。
    ///
    /// `pdf` 字段是光源内部采样 PDF，立体角测度，不包含"选择该光源"
    /// 的概率；delta 光源的 `pdf` 无意义，调用方应根据
    /// [`LightType::is_delta`] 决定是否使用。
    ///
    /// 命名说明：方法名 `sample_li` 中的 `li` 来自渲染方程入射项 `L_i`
    /// （light incident），与返回类型 [`LiSample`] 对应。
    ///
    /// 参数 `sample` 是调用方从 sampler 取出的 2D 均匀样本；本方法不管理
    /// sampler 状态、不消耗维度，与 [`LightSampler::sample`] 的分工见
    /// 该方法的文档。调用方应在 `dimension` 处取 2D 样本后传入，下一次
    /// 采样的基址推进 2。
    ///
    /// 对 [`AreaLight`] 实现，返回的 `incident` 应等于
    /// `self.emitted_radiance(light_position, light_normal, uv, -wi, wavelengths)`
    /// 的求值结果（其中 `light_normal` / `uv` 是光源表面被采样点的法线和
    /// UV，由实现方内部持有；`LiSample` 不携带这两个字段，调用方不应尝试
    /// 从 `LiSample` 复算）。
    ///
    /// 对 [`InfiniteLight`] 实现，返回的 `geometry` 应为
    /// [`LiGeometry::Direction`]，且 `incident` / `pdf` 必须与
    /// [`InfiniteLight::sample_le`] 在同一方向上一致；推荐直接复用
    /// [`InfiniteLight::sample_li_via_le`] 保证两条路径 PDF 不分叉。
    ///
    /// # 泛型 `S`
    ///
    /// `ctx` 携带 `&Library<S>`，`S` 从 `ctx` 的类型推断。`Light` trait
    /// 本身带 `S` 泛型，方法参数不再显式标注 `S`——直接用
    /// `LightSampleContext<'_, S>` 即可。
    fn sample_li(
        &self,
        ctx: &LightSampleContext<'_, S>,
        sample: Vec2,
        wavelengths: &S::WavelengthContext,
    ) -> Option<LiSample<S>>;

    /// 给定方向，返回光源内部采样该方向的 PDF（立体角测度）。
    ///
    /// 命名说明：方法名 `pdf_li` 中的 `li` 来自渲染方程入射项 `L_i`，
    /// 与 [`Light::sample_li`] 对应，参数 `wi` 是渲染方程入射方向 `ω_i`。
    ///
    /// delta 光源返回 0：连续方向采样采不到 delta 方向。
    /// 非 delta 光源返回有限值，用于 MIS 权重计算。
    ///
    /// 契约：对任意 `s = sample_li(ctx, sample, wavelengths)`，应有
    /// `pdf_li(ctx, s.wi) == s.pdf`（delta 光源除外）。
    fn pdf_li(&self, ctx: &LightSampleContext<'_, S>, wi: Vec3) -> f32;

    /// 光源的总辐射功率，逐波长（光谱值）。
    ///
    /// 用于 `LightSampler` 在构造阶段计算光源选择概率。选择概率需要
    /// 标量，标量转换（如亮度近似或参考波长）由 `LightSampler` 实现
    /// 显式完成，不在本接口上暴露模糊的"标量功率近似"。
    ///
    /// 返回的 `S` 是光谱值（`Spectrum<N>` 或 `Rgb`），不是标量；
    /// `S::Scalar` 才是标量。调用方需要标量时，通过
    /// `S::WavelengthContext` 和光谱到标量的转换函数显式完成。
    ///
    /// 构造阶段通常使用参考波长/代表值把 `phi` 转成标量；逐路径波长
    /// 采样时选择概率仍固定为构造值，MIS 会修正由此引入的偏差。
    fn phi(&self, wavelengths: &S::WavelengthContext) -> S;

}

/// 面光源能力。
///
/// 仅 [`LightType::Area`] 的实现方需要实现本 trait。抽象层不提供
/// "从 `dyn Light` 找回 `AreaLight`"的能力查询；持有具体实例的
/// 实现层（场景构建 / 积分器分发）自行枚举并 match。
pub trait AreaLight<S: SpectralAlgebra>: Light<S> {
    /// 给定光源表面上的点、法线、UV、出射方向，返回发射辐射亮度。
    ///
    /// 命名说明：方法名来自渲染方程发射项 `L_e` 的语义读法
    /// "emitted radiance"。不使用单字母 `l`（与数字 `1`、大写 `I`
    /// 易混淆，且脱离方程上下文后语义不明）。`wo` 参数保留渲染方程
    /// 出射方向符号 `ω_o` 的 ASCII 转写。
    ///
    /// 参数说明：
    /// - `point` / `normal` / `uv` 是**光源表面**上的几何信息，不是着色点。
    /// - `wo` 是从光源表面点出发、射向着色点的方向，对应渲染方程
    ///   出射方向 `ω_o`。`LiSample::wi` 从着色点指向光源点，因此
    ///   `wo = -wi`。
    ///
    /// 传输方向隐含 `Radiance`（从着色点看向光源），与
    /// [`InfiniteLight::emitted_radiance`] 的 `Importance` 不同；原因见
    /// [`LeSample`] 的"传输方向"文档。
    ///
    /// 与 [`Light::light_type`] 的契约：实现本 trait ⇒ `light_type()` 必须
    /// 返回 [`LightType::Area`]。
    ///
    /// 与 [`Light::sample_li`] 的一致性约束：`sample_li` 返回的 `incident`
    /// 应等于在采样点处调用本方法的结果。`LiSample` 不携带
    /// `light_normal` / `uv`，所以调用方不应尝试从 `LiSample` 复算；
    /// 本方法用于不经过 `sample_li` 的直接求值场景（如 BDPT 的某些路径）。
    fn emitted_radiance(
        &self,
        point: Vec3,
        normal: Vec3,
        uv: Vec2,
        wo: Vec3,
        wavelengths: &S::WavelengthContext,
    ) -> S;
}

/// 无限远光源能力（环境光 / 天空光 / 方向光）。
///
/// `emitted_radiance`、`sample_le`、`pdf_le` 必须来自同一发射分布。
/// 对任意 `s = sample_le(sample, wavelengths)`，应有：
///
/// ```text
/// emitted_radiance(Ray::new(_, -s.direction)) == s.emitted
/// pdf_le(s.direction, wavelengths) == s.pdf
/// ```
///
/// 其中 `-s.direction` 是因为 `emitted_radiance` 接收光线传播方向
/// （从着色点射向光源），而 `sample_le` 返回的 `direction` 是从光源
/// 射向场景的方向，两者反向。三者在同一 trait 上成对出现，就是为了
/// 让这条约束无法被实现方拆散。
///
/// **传输方向隐含 `Importance`**（从光源射向场景），见 [`LeSample`]
/// 的"传输方向"文档。`sample_li` 通过 [`Self::sample_li_via_le`] 复用
/// 本 trait 的采样，把方向取反后作为 `Radiance` 方向的入射光。
pub trait InfiniteLight<S: SpectralAlgebra>: Light<S> {
    /// 沿逃逸光线的辐射亮度。
    ///
    /// 命名说明：方法名来自渲染方程发射项 `L_e` 的语义读法
    /// "emitted radiance"，与 [`AreaLight::emitted_radiance`] 同名同义
    /// （都是"给定方向求发射辐射亮度"），仅参数形式不同（无限远光源
    /// 只有射线，没有表面点）。两者不会在同一类型上同时实现
    /// （见 [`Light::light_type`] 契约），因此同名不引入调用歧义。
    ///
    /// **传输方向隐含 `Importance`**：`ray` 从着色点出发，
    /// `ray.direction` 是视线方向（从着色点射向光源），与
    /// [`Self::sample_le`] 的 `direction` **反向**。这与
    /// [`LeSample::direction`] 的文档一致。
    ///
    /// 与 [`Light::light_type`] 的契约：实现本 trait 且非 delta ⇒
    /// `light_type()` 返回 [`LightType::Infinite`]；方向光（delta）⇒
    /// 返回 [`LightType::DeltaDirection`]。
    fn emitted_radiance(&self, ray: &Ray, wavelengths: &S::WavelengthContext) -> S;

    /// 光源侧方向采样。
    ///
    /// 命名说明：方法名 `sample_le` 中的 `le` 来自渲染方程发射项 `L_e`
    /// （light emitted），与返回类型 [`LeSample`] 对应，与
    /// [`Light::sample_li`] 的 `li`（light incident）形成对称。
    ///
    /// 返回 `None` 表示采样失败（如采样到零辐射方向）。方向为
    /// 从光源出发、射向场景的方向，与 [`InfiniteLight::emitted_radiance`]
    /// 求值时的 `ray.direction` **反向**；原因见 [`LeSample::direction`]。
    ///
    /// **传输方向隐含 `Importance`**，无 `TransportMode` 参数；原因见
    /// [`LeSample`] 的"传输方向"文档。
    ///
    /// 参数 `sample` 是调用方从 sampler 取出的 2D 均匀样本；本方法不管理
    /// sampler 状态、不消耗维度。
    ///
    /// 无限远光源（环境光 / 天空光）返回的 `geometry` 应为
    /// [`LeGeometry::Direction`]；方向光作为 delta 光源，`pdf` 无有限值，
    /// 调用方应依据 [`LightType::is_delta`] 决定是否参与 MIS。
    ///
    /// 契约：对任意 `s = sample_le(sample, wavelengths)`，应有
    /// `pdf_le(s.direction, wavelengths) == s.pdf`（delta 光源除外）。
    fn sample_le(&self, sample: Vec2, wavelengths: &S::WavelengthContext) -> Option<LeSample<S>>;

    /// 给定方向，返回 `sample_le` 采样该方向的 PDF（方向测度）。
    ///
    /// 命名说明：方法名 `pdf_le` 中的 `le` 来自渲染方程发射项 `L_e`，
    /// 与 [`InfiniteLight::sample_le`] 对应。
    ///
    /// 必须与 `sample_le` 使用同一分布；delta 光源返回 0。
    fn pdf_le(&self, direction: Vec3, wavelengths: &S::WavelengthContext) -> f32;

    /// 供实现方在 `impl Light::sample_li` 中复用：把 `sample_le` 的结果
    /// 转成从着色点看出去的 `LiSample`，保证两条路径 PDF 不分叉。
    ///
    /// 返回的 `geometry` 恒为 [`LiGeometry::Direction`]。
    fn sample_li_via_le(
        &self,
        sample: Vec2,
        wavelengths: &S::WavelengthContext,
    ) -> Option<LiSample<S>> {
        let s = self.sample_le(sample, wavelengths)?;
        Some(LiSample {
            geometry: LiGeometry::Direction,
            // sample_le 的 direction 从光源射向场景；LiSample::wi 从着色点
            // 射向光源，两者反向。
            wi: -s.direction,
            incident: s.emitted,
            pdf: s.pdf,
        })
    }
}

// ═══════════════════════════════════════════════════════════════
// 光源选择器
// ═══════════════════════════════════════════════════════════════

/// 光源选择器：只负责在多个光源之间选择。
///
/// `LightSampler` 的实例在构造层创建，采样策略、光源表、每个光源的
/// 选择概率在构造时固定；每次调用的 `sampler` 由调用方（积分器）传入。
///
/// 光源选择概率的标量转换（如亮度近似或参考波长）在构造阶段显式完成，
/// 不在本 trait 的方法上暴露。
///
/// # 维度管理
///
/// 与 [`Light::sample_li`] 不同，`LightSampler::sample` 接收
/// `&mut impl Sampler` 并**在内部消耗维度**（如 BVH 光源采样器先选
/// 节点、再选光源，消耗 > 1 维）。因此 [`LightSelection`] 必须携带
/// `consumed` 字段，报告本次采样实际消耗的维度数；调用方下一次采样用
/// `dimension + sel.consumed` 推进基址。
///
/// `sample` 返回 `None` 时，调用方用 [`Self::max_sample_dimensions`]
/// 估计已消耗维度以恢复基址。失败通常意味着路径终止，所以这只是保守
/// 恢复，不要求精确。
///
/// # PDF 一致性契约
///
/// 对任意 `sel = sample(ctx, sampler)`，应有：
///
/// ```text
/// pdf(ctx, sel.light_id) == sel.light_pdf
/// ```
///
/// 这是 MIS 的根基：光源采样路径用 `sel.light_pdf`，BSDF 采样路径用
/// `pdf(ctx, light_id)` 反查。两者不一致，MIS 权重就错。跨实现测试
/// 必须校验。
///
/// # `impl Sampler` 参数与 trait object
///
/// `sample` 用 `&mut impl Sampler` 而非 `&mut dyn Sampler`，是为了让
/// 调用点单态化。代价是 `LightSampler` **不能直接作为 trait object**
/// （`Box<dyn LightSampler>` 会因 `impl Sampler` 不 object-safe 而失败）。
///
/// 需要运行时切换光源选择策略时，有两种方案：
///
/// - **包装层**：定义一个 enum `LightSamplerKind`，内部持有具体
///   `LightSampler` 实现。包装层**不实现** `LightSampler` trait（因为
///   trait 的 `sample` 用 `&mut impl Sampler`，不是 object-safe）；
///   而是定义自己的方法
///   `fn sample(&self, ctx, sampler: &mut dyn Sampler) -> Option<LightSelection>`，
///   内部 `match` 分派到具体实现的 `sample`（具体实现仍用
///   `impl Sampler`，包装层用 `dyn` 转发）。虚分派限制在包装层。
/// - **改签名**：把 `sample` 改为 `&mut dyn Sampler`，牺牲单态化换
///   trait object。适合策略切换频繁、单次采样开销可接受的场景。
///
/// # 泛型 `S`
///
/// `sample` / `pdf` 泛型化在 `S: SpectralAlgebra` 上——`ctx` 携带
/// `&Library<S>`，`S` 必须传播。`LightSampler` trait 本身不带 `S`，
/// 方法级泛型化导致它不 object-safe；`Scene::light_sampler()` 返回
/// `Option<&LS>` 泛型引用，走单态化，无 `dyn` 需求。
pub trait LightSampler: Send + Sync {
    /// 按构造时固定的选择策略选一个光源。
    ///
    /// 参数：
    /// - `ctx`：着色点上下文，供依赖位置的选择策略使用；
    /// - `sampler`：采样器；
    /// - `dimension`：**维度基址**。采样器从此维开始消耗，按 `consumed`
    ///   推进。与 `BxDF::sample_f` / `BSDF::sample_f` 的 `dimension` 语义
    ///   一致。
    ///
    /// 返回的 `LightSelection::consumed` 报告本次采样实际消耗的维度数。
    fn sample<S: SpectralAlgebra, SM: Sampler>(
        &self,
        ctx: &LightSampleContext<'_, S>,
        sampler: &mut SM,
        dimension: u32,
    ) -> Option<LightSelection>;

    /// 给定光源 ID，返回选择该光源的概率。
    ///
    /// 用于 MIS 权重计算：`total_pdf = light_pdf * light.pdf_li(...)`。
    ///
    /// 契约：对任意 `sel = sample(ctx, sampler)`，应有
    /// `pdf(ctx, sel.light_id) == sel.light_pdf`。
    ///
    /// `ctx` 参数供依赖接收点位置的选择策略（如 BVH 光源采样器）使用；
    /// 不依赖位置的采样器可以忽略 `ctx`。
    fn pdf<S: SpectralAlgebra>(
        &self,
        ctx: &LightSampleContext<'_, S>,
        light_id: LightId,
    ) -> f32;

    /// 采样最多消耗的维度数。
    ///
    /// 用于两个场景：
    ///
    /// - `sample` 返回 `None` 时，调用方按此估计已消耗维度以恢复基址；
    /// - 调用方需要静态分配维度时（如分层采样），按此预留维度块。
    ///
    /// 是否使用本方法由调用方决定；默认路径用 [`LightSelection::consumed`]
    /// 追踪实际消耗，不需要读本方法。
    fn max_sample_dimensions(&self) -> u32;
}

/// 光源选择结果：选了哪个光源、选择概率是多少。
#[derive(Clone, Copy, Debug)]
pub struct LightSelection {
    /// 被选中的光源 ID。
    pub light_id: LightId,
    /// 选择该光源的概率。
    ///
    /// 调用方计算总 PDF 时：`total_pdf = light_pdf * li_sample.pdf`
    /// （非 delta 光源）；delta 光源不参与 MIS，不需要总 PDF。
    ///
    /// 契约：`light_pdf == LightSampler::pdf(ctx, light_id)`。
    pub light_pdf: f32,
    /// `sample` 从传入 sampler 消耗的维度数。
    ///
    /// 调用方下一次采样用 `dimension + consumed`。`sample` 返回 `None`
    /// 时用 [`LightSampler::max_sample_dimensions`] 估计。
    pub consumed: u32,
}


/// 面积光背面行为。求交层根据命中面的朝向决定返回哪种 `InstanceHit`。
///
/// **与 `light::ShapeAreaLight::two_sided` 的一致性约束**：
/// - `TwoSided` 要求 `light.two_sided == true`（否则背面命中求 emission
///   得到 0，视觉上等于背面不发光但遮挡）；
/// - `Transparent` / `Opaque` 要求 `light.two_sided == false`。
/// - 构建层需保证一致；`ShapeAreaLight::build()` 辅助方法自动推导。
#[derive(Clone, Copy, Debug)]
pub enum AreaLightBackface {
    /// 背面穿过：求交返回 `None`，光线继续找更远的交点。面积光从背面
    /// 看完全不可见（不遮挡、不发光）。
    Transparent,
    /// 背面也发光：求交返回 `AreaLight` 命中，积分器求 `emitted_radiance`。
    /// 要求 `light.two_sided == true`。
    TwoSided,
    /// 背面退化为普通实例：求交返回 `Shape` 命中（不带 `light_ref`）。
    /// 积分器当普通表面处理——无材质则 break（黑体，遮挡但不发光）。
    Opaque,
}

