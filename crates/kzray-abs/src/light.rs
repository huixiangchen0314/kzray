use glam::{Vec2, Vec3};
use kzray_data::LightId;
use crate::ray::Ray;
use crate::sampler::Sampler;
use crate::spectrum::SpectralAlgebra;

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

/// 着色点上下文，传给光源采样。
#[derive(Clone, Copy, Debug)]
pub struct LightSampleContext {
    /// 着色点世界坐标。
    pub position: Vec3,
    /// 着色点着色法线。
    pub normal: Vec3,
    // TODO(ray-or-context): 介质信息是否放这里，留给未来决断。
}

/// `LiSample` 使用的几何形态。
///
/// 区分“有具体光源点”和“只有方向”，避免无限远光源的采样结果被误用于
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
/// （无限远光源）。与 [`LiGeometry`] 的区别：光源侧没有“到着色点的
/// 距离”参照点，但需要表面法线与 UV，供 BDPT 继续 trace 或做端点连接。
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
/// - `pdf` 是光源内部采样该入射方向的概率密度（立体角测度），不包含
///   “选择该光源”的概率；总 PDF 由调用方把 `light_pdf * pdf` 得到。
/// - `geometry` 是光源一侧的采样几何，用于阴影射线、可见性判断，
///   以及面积测度到立体角测度的转换；无限远光源为
///   [`LiGeometry::Direction`]。
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
    ///
    /// 如果实现内部在面积测度上采样，返回前必须转换成
    /// 立体角测度：`pdf_omega = pdf_area * distance^2 / |cos(theta_light)|`。
    ///
    /// delta 光源的 `pdf` 无有限值，调用方不应读取此字段参与 MIS。
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
#[derive(Clone, Copy, Debug)]
pub struct LeSample<S> {
    /// 光源侧采样几何：`Surface` 有具体表面点，`Direction` 没有。
    pub geometry: LeGeometry,
    /// 从光源出发的方向，与 [`InfiniteLight::emitted_radiance`] 求值时的
    /// `ray.direction` 同向。
    ///
    /// 该方向无对应渲染方程符号（方程中 `ω_o` 是着色点出射方向），
    /// 因此使用工程化名称 `direction` 而非符号转写。
    pub direction: Vec3,
    /// 发射辐射亮度，对应渲染方程中的 `L_e`。
    ///
    /// 命名来源：方程符号 `L_e` 的语义读法 "emitted radiance"。
    pub emitted: S,
    /// 方向测度 PDF，必须与 `pdf_le` 来自同一采样分布。
    ///
    /// 如果实现内部在 uv / texel 测度上采样，返回前必须转换成方向测度。
    pub pdf: f32,
}

/// 所有光源共享的采样契约。
///
/// `LightSampler` 只负责“在多个光源之间选一个”；本 trait 负责
/// “给定一个光源，采样它的点/方向、计算 PDF、给出总功率”。
///
/// 面光源求值见 [`AreaLight::emitted_radiance`]，无限远光源求值与
/// 光源侧采样（`sample_le` / `pdf_le` / `emitted_radiance`）见
/// [`InfiniteLight`]。拆分 trait 的目的是让“不适用的能力”在类型层面
/// 不存在，而不是用 `S::zero()` 或 `unreachable!()` 兜底。具体某个
/// `Light` 实例具备哪些能力，由实现层自行枚举分派，抽象层不做类型枚举。
///
/// 波长上下文由 `S::WavelengthContext` 提供：
/// - `Rgb` 配 `NoWavelengths`，零大小；
/// - `Spectrum<N>` 配 `SampledWavelengths<N>`，N 强绑定。
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
    /// `pdf` 字段是光源内部采样 PDF，立体角测度，不包含“选择该光源”
    /// 的概率；delta 光源的 `pdf` 无意义，调用方应根据
    /// [`LightType::is_delta`] 决定是否使用。
    ///
    /// 命名说明：方法名 `sample_li` 中的 `li` 来自渲染方程入射项 `L_i`
    /// （light incident），与返回类型 [`LiSample`] 对应。
    ///
    /// 参数 `sample` 是调用方从 sampler 取出的 2D 均匀样本；本方法不管理
    /// sampler 状态，与 [`LightSampler::sample`] 的分工见该方法的文档。
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
    /// 跨实现测试应校验 `sample_li` 和 `sample_le` 的 PDF 一致。
    fn sample_li(
        &self,
        ctx: &LightSampleContext,
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
    /// 与 [`Light::sample_li`] 的一致性约束：对任意
    /// `s = sample_li(ctx, sample, wavelengths)`，应有
    /// `pdf_li(ctx, s.wi) == s.pdf`（delta 光源除外，两者均为 0 /
    /// 无有限值）。跨实现测试应校验此等式。
    fn pdf_li(&self, ctx: &LightSampleContext, wi: Vec3) -> f32;

    /// 光源的总辐射功率，逐波长（光谱值）。
    ///
    /// 用于 `LightSampler` 在构造阶段计算光源选择概率。选择概率需要
    /// 标量，标量转换（如亮度近似或参考波长）由 `LightSampler` 实现
    /// 显式完成，不在本接口上暴露模糊的“标量功率近似”。
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
/// “从 `dyn Light` 找回 `AreaLight`”的能力查询；持有具体实例的
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
    /// - `wo` 是从光源表面点出发、射向场景的方向，对应渲染方程
    ///   出射方向 `ω_o`；等于 [`LiSample::wi`] 取反（从着色点指向光源点
    ///   的 `-wi`）。
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
/// `emitted_radiance`、`sample_le`、`pdf_le` 必须来自同一发射分布：
/// 给定方向 `d`，`emitted_radiance(Ray::new(_, d))` 必须等于 `sample_le`
/// 在该方向上返回的 `emitted`，且 `pdf_le(d)` 必须等于 `sample_le`
/// 采样到 `d` 的方向测度 PDF。三者在同一 trait 上成对出现，就是为了
/// 让这条约束无法被实现方拆散。
pub trait InfiniteLight<S: SpectralAlgebra>: Light<S> {
    /// 沿逃逸光线的辐射亮度。
    ///
    /// 命名说明：方法名来自渲染方程发射项 `L_e` 的语义读法
    /// "emitted radiance"，与 [`AreaLight::emitted_radiance`] 同名同义
    /// （都是“给定方向求发射辐射亮度”），仅参数形式不同（无限远光源
    /// 只有射线，没有表面点）。两者不会在同一类型上同时实现
    /// （见 [`Light::light_type`] 契约），因此同名不引入调用歧义。
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
    /// 从光源出发的方向，与 [`InfiniteLight::emitted_radiance`] 求值时的
    /// `ray.direction` 同向。
    ///
    /// 参数 `sample` 是调用方从 sampler 取出的 2D 均匀样本；本方法不管理
    /// sampler 状态。
    ///
    /// 无限远光源（环境光 / 天空光）返回的 `geometry` 应为
    /// [`LeGeometry::Direction`]；方向光作为 delta 光源，`pdf` 无有限值，
    /// 调用方应依据 [`LightType::is_delta`] 决定是否参与 MIS。
    fn sample_le(
        &self,
        sample: Vec2,
        wavelengths: &S::WavelengthContext,
    ) -> Option<LeSample<S>>;

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
            wi: -s.direction,
            incident: s.emitted,
            pdf: s.pdf,
        })
    }
}

/// 光源选择器：只负责在多个光源之间选择。
///
/// `LightSampler` 的实例在构造层创建，采样策略、光源表、每个光源的
/// 选择概率在构造时固定；每次调用的 `sampler` 由调用方（积分器）传入，
/// 类型在调用点单态化。
///
/// 光源选择概率的标量转换（如亮度近似或参考波长）在构造阶段显式完成，
/// 不在本 trait 的方法上暴露。
pub trait LightSampler: Send + Sync {
    /// 按构造时固定的选择策略选一个光源。
    ///
    /// 返回 `None` 表示没有可用光源（如场景无光源）。
    ///
    /// 与 [`Light::sample_li`] 的分工：本方法从 `sampler` 取样本并
    /// 选择光源（可能取多个样本，如 BVH 光源采样器先选节点再选光源）；
    /// `Light::sample_li` 接受已取出的 2D 样本，只负责采样单个光源的
    /// 点/方向，不管理 sampler 状态。
    fn sample(
        &self,
        ctx: &LightSampleContext,
        sampler: &mut impl Sampler,
    ) -> Option<LightSelection>;

    /// 给定光源 ID，返回选择该光源的概率。
    ///
    /// 用于 MIS 权重计算：`total_pdf = light_pdf * light.pdf_li(...)`。
    ///
    /// `ctx` 参数供依赖接收点位置的选择策略（如 BVH 光源采样器）使用；
    /// 不依赖位置的采样器可以忽略 `ctx`。
    fn pdf(&self, ctx: &LightSampleContext, light_id: LightId) -> f32;
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
    pub light_pdf: f32,
}
