//! BSDF：单个 BxDF 分量 + 组合容器 + 一致的混合 PDF。
//!
//! 维度管理：**追踪实际消耗**。`BxDFSample::consumed` 返回本次采样从传入
//! `dimension` 开始实际消耗的维度数；调用方下一次采样用
//! `dimension + consumed`。`sample_f` 失败时返回 `None`，调用方用
//! [`BxDF::max_sample_dimensions`] 估计已消耗维度以恢复基址。
//!
//! 追踪的封闭性：只有 [`BSDF::sample_f`] / [`BSDF::rho`] 调用子 BxDF 时
//! 需要累加 `consumed`，以及积分器调用 [`BSDF::sample_f`] 后推进基址。
//! 其余代码只读 `consumed`。
//!
//! 分层 / 低差异采样若未来需要，用 [`BSDF::max_sample_dimensions`] 静态
//! 分配，不依赖追踪值。是否启用分层采样是调用方的配置决策，本模块不感知。
//!
//! PDF 精确性：`BxDFSample::pdf_is_exact` 与 [`BxDF::pdf_is_exact`] 标记
//! 该 PDF 是否是精确归一化的立体角测度。默认 `true`（原子 BxDF 均精确）；
//! 层状 / 涂层 BxDF 可能返回 `false`，此时 PDF 只是与 `value` 成比例，
//! 不能直接参与 MIS。组合 BSDF 的整体精确性是所有非零权重条目的 `AND`。
//!
//! Albedo 与构造期计算：半球-半球反射率（albedo）是材质的**静态属性**，
//! 由构造参数唯一确定，在构造时已知。它**不是** [`BxDF`] trait 的方法——
//! 每个具体 BxDF 类型按其自身语义提供固有方法（如 `Diffuse::albedo` 直接
//! 返回字段，`Conductor::albedo` 返回 Smith 模型近似），材质构造 [`BSDF`]
//! 时调用一次，结果存入 [`BxDFEntry::albedo`]，[`BSDF::albedo`] 只读缓存。
//! 这样避免 trait 默认返回零的静默错误，也避免每次查询走动态分派。

use std::sync::Arc;

use glam::Vec3;

use crate::sampler::Sampler;
use crate::spectrum::ScalarAlgebra;
use kzray_math::onb::Onb;
use kzray_math::sampling;

// ═══════════════════════════════════════════════════════════════
// 能力标签
// ═══════════════════════════════════════════════════════════════

/// BxDF 能力标签。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BxDFFlags(u8);

impl BxDFFlags {
    pub const NONE: BxDFFlags = BxDFFlags(0);
    pub const REFLECTION: BxDFFlags = BxDFFlags(1 << 0);
    pub const TRANSMISSION: BxDFFlags = BxDFFlags(1 << 1);
    pub const DIFFUSE: BxDFFlags = BxDFFlags(1 << 2);
    pub const GLOSSY: BxDFFlags = BxDFFlags(1 << 3);
    pub const SPECULAR: BxDFFlags = BxDFFlags(1 << 4);

    pub const fn empty() -> Self {
        BxDFFlags::NONE
    }
    pub const fn union(self, other: Self) -> Self {
        BxDFFlags(self.0 | other.0)
    }
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn is_reflective(self) -> bool {
        self.contains(BxDFFlags::REFLECTION)
    }
    pub const fn is_transmissive(self) -> bool {
        self.contains(BxDFFlags::TRANSMISSION)
    }
    pub const fn is_diffuse(self) -> bool {
        self.contains(BxDFFlags::DIFFUSE)
    }
    pub const fn is_glossy(self) -> bool {
        self.contains(BxDFFlags::GLOSSY)
    }
    pub const fn is_specular(self) -> bool {
        self.contains(BxDFFlags::SPECULAR)
    }

    pub const fn is_delta(self) -> bool {
        self.is_specular()
    }
    pub const fn is_non_delta(self) -> bool {
        self.is_diffuse() || self.is_glossy()
    }

    pub const fn is_pure_delta(self) -> bool {
        self.is_delta() && !self.is_non_delta()
    }

    pub const fn is_pure_non_delta(self) -> bool {
        self.is_non_delta() && !self.is_delta()
    }

    pub const fn is_valid(self) -> bool {
        self.is_pure_delta() != self.is_pure_non_delta()
    }
}

impl std::ops::BitOr for BxDFFlags {
    type Output = BxDFFlags;
    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

// ═══════════════════════════════════════════════════════════════
// 传输模式
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportMode {
    Radiance,
    Importance,
}

impl TransportMode {
    pub fn flip(self) -> Self {
        match self {
            TransportMode::Radiance => TransportMode::Importance,
            TransportMode::Importance => TransportMode::Radiance,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// 采样结果
// ═══════════════════════════════════════════════════════════════

/// 单次 BxDF 采样的结果。
#[derive(Clone, Copy, Debug)]
pub struct BxDFSample<S: ScalarAlgebra> {
    /// 采样出的入射方向（世界坐标系，已归一化）。
    pub wi: Vec3,
    /// 原始 BSDF 值（不含余弦项、不含 PDF）。
    pub value: S,
    /// 权重：`value × |cos| / pdf`（纯 delta：`value × |cos| / weight_selected`）。
    ///
    /// **纯 delta 路径的 `weight` 不可分解**：`weight_selected`（选择该
    /// 分量的概率）已合并进 `weight`，调用方无法从 `weight` 反推
    /// `value` 或选择概率。需要 delta 路径的 `value` 时读 `self.value`。
    pub weight: S,
    /// 采样该方向的混合 PDF（纯 non-delta 有效；纯 delta 恒为 0）。
    pub pdf: f32,
    /// `pdf` 是否是精确归一化的立体角测度 PDF。
    ///
    /// - `true`（默认）：`pdf` 是精确的、归一化的 PDF，可直接参与 MIS。
    /// - `false`：`pdf` 只是与 `value` 成比例，不是精确归一化 PDF。
    ///   此时 `value / pdf` 仍然无偏（比例因子抵消），但 `pdf` **不能**
    ///   直接参与 MIS 的 PDF 相加或比较。
    ///
    /// 典型场景：层状 / 涂层 BxDF 的内部采样分布是近似分布。pbrt 用
    /// `pdfIsProportional` 标记同一概念，此处使用语义正向的
    /// `pdf_is_exact` 避免 `Default` 实现时搞反。
    pub pdf_is_exact: bool,
    /// 本次采样命中的 BxDF 分量的能力标签。
    ///
    /// 用 `flags.is_pure_delta()` 判断本次是否命中纯 delta 分量。
    pub flags: BxDFFlags,
    /// 透射时的相对折射率；反射或非折射为 1.0。
    pub eta: f32,
    /// 本次采样消耗的维度数，从传入的 `dimension` 开始计。
    ///
    /// 调用方下一次采样用 `dimension + consumed`。`sample_f` 失败时
    /// 调用方应使用 [`BxDF::max_sample_dimensions`] 估计。
    pub consumed: u32,
}

// ═══════════════════════════════════════════════════════════════
// 单个 BxDF 分量
// ═══════════════════════════════════════════════════════════════

/// 单个散射分量。
///
/// 所有方向都在局部坐标系：`n = (0, 0, 1)`，`wo` 指向表面外侧且
/// `wo.z > 0`（由 [`BSDF::new`] 保证）。透射分量返回的 `wi` 指向
/// 表面内侧（`wi.z < 0`）。
///
/// **每个 BxDF 必须纯 delta 或纯 non-delta**（`flags().is_valid()`）。
///
/// # 维度追踪
///
/// `sample_f` 和 `rho` 从传入的 `dimension` 开始消耗维度，通过返回值的
/// `consumed` 字段报告实际消耗数。调用方负责累加。`sample_f` 失败时
/// 返回 `None`，调用方用 [`Self::max_sample_dimensions`] 估计。
///
/// # PDF 精确性
///
/// `pdf_is_exact()` 标记该 BxDF 的采样 PDF 是否是精确归一化的。原子
/// BxDF（漫反射、GGX、纯镜面）应返回 `true`（默认）；层状 / 涂层 BxDF
/// 的近似采样应覆盖为 `false`。组合 BSDF 整体状态是所有非零权重条目的
/// `AND`。
///
/// # Albedo 不在本 trait 上
///
/// 半球-半球反射率（albedo）是材质的静态属性，由构造参数确定。具体
/// BxDF 类型按自身语义提供固有方法（如 `Diffuse::albedo`），材质构造
/// [`BSDF`] 时调用一次，结果存入 [`BxDFEntry::albedo`]。本 trait 不提供
/// `albedo()` 方法，避免默认返回零的静默错误与每次查询的动态分派。
pub trait BxDF<S: ScalarAlgebra>: Send + Sync {
    /// 能力标签。必须满足 `flags().is_valid()`。
    fn flags(&self) -> BxDFFlags;

    /// 给定出射/入射方向，返回 BSDF 值（不含余弦项，不含 PDF）。
    fn eval(&self, wo: Vec3, wi: Vec3, mode: TransportMode) -> S;

    /// 采样一个入射方向（局部坐标系）。
    ///
    /// 返回 `None` 表示采样失败；调用方用 [`Self::max_sample_dimensions`]
    /// 估计已消耗维度。
    fn sample_f(
        &self,
        wo: Vec3,
        sampler: &mut dyn Sampler,
        dimension: u32,
        mode: TransportMode,
    ) -> Option<BxDFSample<S>>;

    /// 给定方向的 PDF（立体角测度）。
    ///
    /// 契约：对任意 `s = sample_f(wo, sampler, dim, mode)` 且
    /// `s.flags.is_pure_delta() == false`，应有 `pdf(wo, s.wi, mode) == s.pdf`。
    ///
    /// `pdf_is_exact() == false` 时，返回的 PDF 只是与 `value` 成比例，
    /// 不能直接参与 MIS。
    fn pdf(&self, wo: Vec3, wi: Vec3, mode: TransportMode) -> f32;

    /// 方向-半球反射率（或透射率）。
    ///
    /// 返回 `(rho, consumed)`：`consumed` 是从传入 `dimension` 开始实际
    /// 消耗的维度数，调用方下一次采样用 `dimension + consumed`。
    ///
    /// 默认返回 `(S::zero(), 0)`（解析估计，不消耗维度）。需要随机估计
    /// 的实现从 `dimension` 开始消耗维度，返回实际消耗数。
    ///
    /// 与 albedo 的区别：`rho(wo)` 依赖出射方向，是运行时量；albedo 是
    /// 方向无关的静态属性，在构造层计算，不在本 trait 上。
    fn rho(
        &self,
        _wo: Vec3,
        _sampler: &mut dyn Sampler,
        _dimension: u32,
        _mode: TransportMode,
    ) -> (S, u32) {
        (S::zero(), 0)
    }

    /// 采样最多消耗的维度数。
    ///
    /// 用于两个场景：
    ///
    /// - `sample_f` 返回 `None` 时，调用方按此估计已消耗维度以恢复基址；
    /// - 调用方需要静态分配维度时（如分层采样），按此预留维度块。
    ///
    /// 默认 2（方向采样）。需要额外样本的 BxDF 覆盖此方法。
    ///
    /// 是否使用本方法由调用方决定；默认路径用 `consumed` 追踪实际消耗，
    /// 不需要读本方法。分层采样是否启用是调用方的配置决策。
    fn max_sample_dimensions(&self) -> u32 {
        2
    }

    /// 该 BxDF 的采样 PDF 是否是精确归一化的。
    ///
    /// - `true`（默认）：`sample_f` 返回的 `pdf` 是精确的立体角测度 PDF，
    ///   可直接参与 MIS。原子 BxDF（漫反射、GGX、纯镜面）均返回 `true`。
    /// - `false`：`pdf` 只是与 `value` 成比例，不能直接参与 MIS。
    ///   典型场景是层状 / 涂层 BxDF 内部使用近似采样分布。
    ///
    /// `BSDF::new` 构造时用此方法缓存整体状态；组合 BSDF 的整体状态是
    /// 所有非零权重条目的 `AND`。`BSDF::sample_f` 在采样返回的
    /// `BxDFSample::pdf_is_exact` 上叠加整体状态，供调用方判断。
    fn pdf_is_exact(&self) -> bool {
        true
    }
}

// ═══════════════════════════════════════════════════════════════
// BSDF 组合容器
// ═══════════════════════════════════════════════════════════════

/// 组合 BSDF 的一个条目。
///
/// 由材质在构造层填充：`bxdf` 是具体 BxDF 的共享句柄，`weight` 是选择
/// 权重，`albedo` 是该分量的半球-半球反射率——由材质调用具体 BxDF 的
/// 固有方法（如 `Diffuse::albedo`）计算，构造时一次性完成。
///
/// `albedo` 不经过 [`BxDF`] trait：它是静态属性，不是运行时接口。
/// 具体 BxDF 类型按自身语义提供固有方法；材质知道具体类型，编译期解析。
pub struct BxDFEntry<S: ScalarAlgebra> {
    /// 共享的 BxDF 分量。
    pub bxdf: Arc<dyn BxDF<S>>,
    /// 选择该分量的概率权重（构造时归一化）。
    pub weight: f32,

    /// 该分量的半球-半球反射率：
    ///
    /// ```text
    /// R = (1/π) ∫∫ f(wi, wo) |cos θi| |cos θo| dωi dωo
    /// ```
    ///
    /// 由材质在构造时计算，通常是具体 BxDF 类型的固有方法返回值。
    /// 对无法解析计算的实验性 BxDF，材质可用固定的低差异采样估计后传入。
    pub albedo: S,
}

impl<S: ScalarAlgebra> BxDFEntry<S> {
    /// 便捷构造：调用 `bxdf` 的 `albedo()` 固有方法不可行（`Arc<dyn BxDF>`
    /// 擦除了具体类型），所以 albedo 必须由调用方显式传入。
    ///
    /// 典型用法（材质构造代码）：
    ///
    /// ```text
    /// let diffuse = Diffuse::new(albedo);
    /// let entry = BxDFEntry {
    ///     bxdf: Arc::new(diffuse),
    ///     weight: 1.0,
    ///     albedo: diffuse.albedo,  // 固有字段/方法，编译期解析
    /// };
    /// ```
    pub fn new(bxdf: Arc<dyn BxDF<S>>, weight: f32, albedo: S) -> Self {
        BxDFEntry { bxdf, weight, albedo }
    }
}

/// 多个 BxDF 的组合。
pub struct BSDF<S: ScalarAlgebra> {
    entries: Vec<BxDFEntry<S>>,
    wo: Vec3,
    onb: Onb,
    all_delta: bool,
    /// 所有非零权重条目的 `pdf_is_exact()` 都为 `true` 时，整体为 `true`。
    /// `false` 时 `BSDF::pdf` 的返回值不能直接用于 MIS。
    pdf_is_exact: bool,
    /// 组合的半球-半球反射率，构造时按权重加权求和后缓存。
    ///
    /// 独立叠加的 BxDF（无层间相互作用）该加权和是精确的；层状 / 涂层
    /// 等有相互作用的组合，精确 albedo 需具体材质覆盖（材质自行计算后
    /// 通过构造参数覆盖，或在外层包装类型上提供）。
    albedo: S,
}

impl<S: ScalarAlgebra> BSDF<S> {
    /// 构造 BSDF。
    ///
    /// `entries` 的 `albedo` 字段由材质在调用本方法前计算。`wo_world` 和
    /// `normal_world` 用于建立局部坐标系；`wo_world` 必须与 `normal_world`
    /// 同侧，否则返回 `Err`。
    pub fn new(
        entries: Vec<BxDFEntry<S>>,
        wo_world: Vec3,
        normal_world: Vec3,
    ) -> Result<Self, BSDFError> {
        if entries.is_empty() {
            return Err(BSDFError::Empty);
        }

        let total_weight: f32 = entries.iter().map(|e| e.weight).sum();
        if total_weight <= 0.0 {
            return Err(BSDFError::ZeroWeight);
        }

        for entry in &entries {
            if !entry.bxdf.flags().is_valid() {
                return Err(BSDFError::InvalidFlags);
            }
        }

        let onb = Onb::from_w(normal_world);
        let wo_local = onb.to_local(wo_world);
        let wo_len = wo_local.length();
        if wo_len <= f32::EPSILON {
            return Err(BSDFError::DegenerateWo);
        }
        let wo_local = wo_local / wo_len;
        if wo_local.z <= 0.0 {
            return Err(BSDFError::WoOnWrongSide);
        }

        // 归一化权重，同时按归一化后的权重复合 albedo
        let inv_total = 1.0 / total_weight;
        let mut albedo = S::zero();
        let mut normalized: Vec<BxDFEntry<S>> = Vec::with_capacity(entries.len());
        for mut entry in entries {
            entry.weight *= inv_total;
            albedo = albedo.add(&entry.albedo.scale(S::Scalar::from(entry.weight)));
            normalized.push(entry);
        }

        let all_delta = normalized.iter().all(|e| e.bxdf.flags().is_pure_delta());
        let pdf_is_exact = normalized.iter().all(|e| e.bxdf.pdf_is_exact());

        Ok(BSDF {
            entries: normalized,
            wo: wo_local,
            onb,
            all_delta,
            pdf_is_exact,
            albedo,
        })
    }

    pub fn has_reflection(&self) -> bool {
        self.entries.iter().any(|e| e.bxdf.flags().is_reflective())
    }

    pub fn has_transmission(&self) -> bool {
        self.entries
            .iter()
            .any(|e| e.bxdf.flags().is_transmissive())
    }

    pub fn is_delta(&self) -> bool {
        self.all_delta
    }

    /// 该 BSDF 的采样 PDF 是否精确归一化。
    ///
    /// 返回 `false` 时，`BSDF::pdf` 和 `BxDFSample::pdf` 的返回值不能
    /// 直接用于 MIS；调用方应走修正路径。
    pub fn is_pdf_exact(&self) -> bool {
        self.pdf_is_exact
    }

    /// 组合的半球-半球反射率，构造时缓存。
    ///
    /// 纯属性查询，无动态分派、无维度消耗。用于材质预计算、能量守恒
    /// 测试、光源采样重要性近似——**不是渲染主循环的热路径**。
    ///
    /// 独立叠加的 BxDF 该值是精确的；层状 / 涂层等有相互作用的组合，
    /// 精确 albedo 需材质在外层覆盖。
    pub fn albedo(&self) -> S {
        self.albedo
    }

    pub fn to_world(&self, wi_local: Vec3) -> Vec3 {
        self.onb.to_world(wi_local)
    }

    pub fn to_local(&self, wi_world: Vec3) -> Vec3 {
        self.onb.to_local(wi_world)
    }

    pub fn eval(&self, wi_world: Vec3, mode: TransportMode) -> S {
        let wi = self.to_local(wi_world);
        let mut result = S::zero();
        for entry in &self.entries {
            result = result.add(&entry.bxdf.eval(self.wo, wi, mode));
        }
        result
    }

    /// 给定世界方向，返回混合 PDF（立体角测度）。
    ///
    /// 只包含 non-delta 分量的加权贡献；delta 分量对连续测度贡献 0。
    /// 当 `is_delta()` 为 `true` 时返回 0。
    ///
    /// **注意**：`is_pdf_exact() == false` 时，返回值不是精确的混合 PDF，
    /// 不能直接参与 MIS。调用方必须先检查 `is_pdf_exact()`。
    ///
    /// 契约：对任意非 delta 采样 `s = sample_f(...)`，应有
    /// `pdf(s.wi) == s.pdf`。
    pub fn pdf(&self, wi_world: Vec3, mode: TransportMode) -> f32 {
        if self.all_delta {
            return 0.0;
        }
        let wi = self.to_local(wi_world);
        let mut result = 0.0;
        for entry in &self.entries {
            if entry.bxdf.flags().is_pure_delta() {
                continue;
            }
            result += entry.weight * entry.bxdf.pdf(self.wo, wi, mode);
        }
        result
    }

    /// 采样一个入射方向（世界坐标系）。
    ///
    /// 返回的 `consumed` 是从传入 `dimension` 开始的总消耗（选择维 1 +
    /// 子 BxDF 消耗）。调用方下一次采样用 `dimension + consumed`。
    ///
    /// 失败时返回 `None`；调用方用 [`Self::max_sample_dimensions`] 估计。
    ///
    /// 返回的 `pdf_is_exact` 是选中条目的 `pdf_is_exact` 与组合整体
    /// `is_pdf_exact()` 的 `AND`。
    pub fn sample_f(
        &self,
        sampler: &mut dyn Sampler,
        dimension: u32,
        mode: TransportMode,
    ) -> Option<BxDFSample<S>> {
        // 1. 按权重选一个 BxDF（1 维）
        let idx = self.pick_entry(sampler.next_f32(dimension))?;
        let entry = &self.entries[idx];

        // 2. 子 BxDF 从 dimension + 1 开始，返回自身消耗
        let mut bs = entry.bxdf.sample_f(self.wo, sampler, dimension + 1, mode)?;
        bs.consumed += 1; // 加上选择维

        let wi_local = bs.wi;
        if wi_local.z.abs() <= f32::EPSILON {
            return None;
        }

        // 组合后 pdf 的精确性：选中条目的 pdf_is_exact AND 整体 pdf_is_exact。
        // 只要有任何非零权重条目的 pdf 不是精确的，整个混合 PDF 都不是
        // 精确的（混合公式 Σ weight × pdf 依赖每个 pdf 都归一化）。
        bs.pdf_is_exact &= self.pdf_is_exact;

        // 3. 用子 BxDF 返回的实际命中 flags 判断纯 delta
        if bs.flags.is_pure_delta() {
            let cos = wi_local.z.abs();
            let weight = bs.value.scale(S::Scalar::from(cos / entry.weight));
            bs.wi = self.to_world(wi_local).normalize();
            bs.weight = weight;
            bs.pdf = 0.0;
            // delta 不参与 MIS，pdf_is_exact 无意义，但保持真实值
            return Some(bs);
        }

        // 4. 纯 non-delta 采样：混合 PDF + 总 BSDF 值
        let mut mixed_pdf = 0.0;
        for e in &self.entries {
            if e.bxdf.flags().is_pure_delta() {
                continue;
            }
            mixed_pdf += e.weight * e.bxdf.pdf(self.wo, wi_local, mode);
        }
        if mixed_pdf <= f32::EPSILON {
            return None;
        }

        let mut total_f = S::zero();
        for e in &self.entries {
            total_f = total_f.add(&e.bxdf.eval(self.wo, wi_local, mode));
        }

        let cos = wi_local.z.abs();
        let weight = total_f.scale(S::Scalar::from(cos / mixed_pdf));

        Some(BxDFSample {
            wi: self.to_world(wi_local).normalize(),
            value: total_f,
            weight,
            pdf: mixed_pdf,
            pdf_is_exact: bs.pdf_is_exact,
            flags: bs.flags,
            eta: bs.eta,
            consumed: bs.consumed,
        })
    }

    fn pick_entry(&self, u: f32) -> Option<usize> {
        let u = u.clamp(0.0, 1.0 - f32::EPSILON);
        let mut acc = 0.0;
        for (i, entry) in self.entries.iter().enumerate() {
            acc += entry.weight;
            if u < acc {
                return Some(i);
            }
        }
        self.entries.len().checked_sub(1)
    }

    /// 采样最多消耗的维度数（选择维 1 + 子 BxDF 最大消耗）。
    ///
    /// 用于 `sample_f` 失败后估计已消耗维度，以及调用方需要静态分配维度
    /// （如分层采样）时预留维度块。默认追踪路径不需要读本方法。
    pub fn max_sample_dimensions(&self) -> u32 {
        1 + self
            .entries
            .iter()
            .map(|e| e.bxdf.max_sample_dimensions())
            .max()
            .unwrap_or(0)
    }

    /// 聚合方向-半球反射率（或透射率），按条目权重加权。
    ///
    /// **追踪实际消耗**：每个 BxDF 的 `rho` 返回 `(value, consumed)`，
    /// 维度按实际 `consumed` 推进。返回 `(rho, consumed)`；调用方下一次
    /// 采样用 `dimension + consumed`。
    pub fn rho(&self, sampler: &mut dyn Sampler, dimension: u32, mode: TransportMode) -> (S, u32) {
        let mut result = S::zero();
        let mut dim = dimension;
        for entry in &self.entries {
            let (r, consumed) = entry.bxdf.rho(self.wo, sampler, dim, mode);
            result = result.add(&r.scale(S::Scalar::from(entry.weight)));
            // 追踪实际消耗，不用 max_sample_dimensions 保守推进
            dim += consumed;
        }
        (result, dim - dimension)
    }
}

/// 构造 BSDF 的错误。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BSDFError {
    Empty,
    ZeroWeight,
    InvalidFlags,
    DegenerateWo,
    WoOnWrongSide,
}

impl std::fmt::Display for BSDFError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BSDFError::Empty => write!(f, "BSDF has no BxDF entries"),
            BSDFError::ZeroWeight => write!(f, "BSDF weights sum to zero"),
            BSDFError::InvalidFlags => {
                write!(f, "BxDF flags are neither pure delta nor pure non-delta")
            }
            BSDFError::DegenerateWo => write!(f, "wo direction is degenerate"),
            BSDFError::WoOnWrongSide => {
                write!(f, "wo is on the wrong side of the shading normal")
            }
        }
    }
}

impl std::error::Error for BSDFError {}

// ═══════════════════════════════════════════════════════════════
// 示例：漫反射 BxDF
// ═══════════════════════════════════════════════════════════════

/// 理想漫反射。`eval` 返回 `albedo / π`，采样按余弦加权。
///
/// 采样和 PDF 复用 `crate::sampling`，不内联实现。
///
/// `albedo` 字段既是漫反射的反射率参数，也是半球-半球反射率——漫反射的
/// 方向无关性使二者相同。材质构造 `BxDFEntry` 时直接读该字段：
///
/// ```text
/// let d = Diffuse::new(albedo);
/// let entry = BxDFEntry { bxdf: Arc::new(d), weight: 1.0, albedo: d.albedo };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Diffuse<S: ScalarAlgebra> {
    pub albedo: S,
}

impl<S: ScalarAlgebra> Diffuse<S> {
    pub fn new(albedo: S) -> Self {
        Diffuse { albedo }
    }

    /// 半球-半球反射率。漫反射的方向无关性使 R 等于 albedo。
    ///
    /// 固有方法，不是 [`BxDF`] trait 方法；材质构造时调用一次。
    pub fn hemispherical_reflectance(&self) -> S {
        self.albedo
    }
}

impl<S: ScalarAlgebra> BxDF<S> for Diffuse<S> {
    fn flags(&self) -> BxDFFlags {
        BxDFFlags::REFLECTION | BxDFFlags::DIFFUSE
    }

    fn eval(&self, wo: Vec3, wi: Vec3, _mode: TransportMode) -> S {
        // 防御性检查：BSDF::new 已保证 wo.z > 0，但子 BxDF 独立使用时
        // 仍需要保护。开销很低，保留。
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return S::zero();
        }
        self.albedo
            .scale(S::Scalar::from(1.0 / std::f32::consts::PI))
    }

    fn sample_f(
        &self,
        wo: Vec3,
        sampler: &mut dyn Sampler,
        dimension: u32,
        mode: TransportMode,
    ) -> Option<BxDFSample<S>> {
        if wo.z <= 0.0 {
            return None;
        }
        let (u, v) = sampler.next_2d(dimension);
        let (wi, pdf) = sampling::cosine_hemisphere(u, v);
        if wi.z <= 0.0 || pdf <= f32::EPSILON {
            return None;
        }
        let value = self.eval(wo, wi, mode);
        let cos = wi.z.abs();
        let weight = value.scale(S::Scalar::from(cos / pdf));
        Some(BxDFSample {
            wi,
            value,
            weight,
            pdf,
            pdf_is_exact: true, // 余弦加权采样是精确 PDF
            flags: self.flags(),
            eta: 1.0,
            consumed: 2, // 2 维方向采样
        })
    }

    fn pdf(&self, wo: Vec3, wi: Vec3, _mode: TransportMode) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        sampling::cosine_hemisphere_pdf(wi.z)
    }

    fn rho(
        &self,
        _wo: Vec3,
        _sampler: &mut dyn Sampler,
        _dimension: u32,
        _mode: TransportMode,
    ) -> (S, u32) {
        // 解析估计，不消耗维度
        (self.albedo, 0)
    }

    fn max_sample_dimensions(&self) -> u32 {
        2
    }

    // pdf_is_exact() 用默认 true，无需覆盖

    // albedo 不在 trait 上；固有方法 hemispherical_reflectance()
    // 或直接读字段 self.albedo。
}
