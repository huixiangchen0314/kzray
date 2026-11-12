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
//! 分配，不依赖追踪值。

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
    pub weight: S,
    /// 采样该方向的混合 PDF（纯 non-delta 有效；纯 delta 恒为 0）。
    pub pdf: f32,
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
/// `sample_f` 从传入的 `dimension` 开始消耗维度，通过
/// [`BxDFSample::consumed`] 返回实际消耗数。调用方负责累加。失败时
/// 返回 `None`，调用方用 [`Self::max_sample_dimensions`] 估计。
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
    fn pdf(&self, wo: Vec3, wi: Vec3, mode: TransportMode) -> f32;

    /// 方向-半球反射率（或透射率）。
    ///
    /// 默认返回零（解析，不消耗维度）。需要随机估计的实现从 `dimension`
    /// 开始消耗维度，约定不超过 [`Self::max_sample_dimensions`]。
    fn rho(
        &self,
        _wo: Vec3,
        _sampler: &mut dyn Sampler,
        _dimension: u32,
        _mode: TransportMode,
    ) -> S {
        S::zero()
    }

    /// 采样最多消耗的维度数。
    ///
    /// 用于：
    ///
    /// - `sample_f` 返回 `None` 时，调用方按此估计已消耗维度以恢复基址；
    /// - 调用方需要静态分配维度时（如分层采样），按此预留维度块。
    ///
    /// 默认 2（方向采样）。需要额外样本的 BxDF 覆盖此方法。
    ///
    /// 是否使用本方法由调用方决定；默认路径用 [`BxDFSample::consumed`]
    /// 追踪实际消耗，不需要读本方法。
    fn max_sample_dimensions(&self) -> u32 {
        2
    }
}

// ═══════════════════════════════════════════════════════════════
// BSDF 组合容器
// ═══════════════════════════════════════════════════════════════

struct BxDFEntry<S: ScalarAlgebra> {
    bxdf: Arc<dyn BxDF<S>>,
    weight: f32,
}

/// 多个 BxDF 的组合。
pub struct BSDF<S: ScalarAlgebra> {
    entries: Vec<BxDFEntry<S>>,
    wo: Vec3,
    onb: Onb,
    all_delta: bool,
}

impl<S: ScalarAlgebra> BSDF<S> {
    pub fn new(
        entries: Vec<(Arc<dyn BxDF<S>>, f32)>,
        wo_world: Vec3,
        normal_world: Vec3,
    ) -> Result<Self, BSDFError> {
        if entries.is_empty() {
            return Err(BSDFError::Empty);
        }

        let total_weight: f32 = entries.iter().map(|(_, w)| *w).sum();
        if total_weight <= 0.0 {
            return Err(BSDFError::ZeroWeight);
        }

        for (bxdf, _) in &entries {
            if !bxdf.flags().is_valid() {
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

        let normalized: Vec<BxDFEntry<S>> = entries
            .into_iter()
            .map(|(bxdf, weight)| BxDFEntry {
                bxdf,
                weight: weight / total_weight,
            })
            .collect();

        let all_delta = normalized.iter().all(|e| e.bxdf.flags().is_pure_delta());

        Ok(BSDF {
            entries: normalized,
            wo: wo_local,
            onb,
            all_delta,
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

        // 3. 用子 BxDF 返回的实际命中 flags 判断纯 delta
        if bs.flags.is_pure_delta() {
            let cos = wi_local.z.abs();
            let weight = bs.value.scale(S::Scalar::from(cos / entry.weight));
            bs.wi = self.to_world(wi_local).normalize();
            bs.weight = weight;
            bs.pdf = 0.0;
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
    /// 用于 `sample_f` 失败后估计已消耗维度，以及分层采样静态分配。
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
    /// 每个 BxDF 的 `rho` 从当前维度基址开始，按实际消耗推进（追踪模式）。
    /// 返回 `(rho, consumed)`；调用方下一次采样用 `dimension + consumed`。
    pub fn rho(&self, sampler: &mut dyn Sampler, dimension: u32, mode: TransportMode) -> (S, u32) {
        let mut result = S::zero();
        let mut dim = dimension;
        for entry in &self.entries {
            let r = entry.bxdf.rho(self.wo, sampler, dim, mode);
            result = result.add(&r.scale(S::Scalar::from(entry.weight)));
            // 保守估计：每个 BxDF 最多消耗 max_sample_dimensions
            // （rho 的随机估计在契约上不超过这个上限）
            dim += entry.bxdf.max_sample_dimensions();
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

#[derive(Clone, Copy, Debug)]
pub struct Diffuse<S: ScalarAlgebra> {
    pub albedo: S,
}

impl<S: ScalarAlgebra> Diffuse<S> {
    pub fn new(albedo: S) -> Self {
        Diffuse { albedo }
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
    ) -> S {
        self.albedo
    }

    fn max_sample_dimensions(&self) -> u32 {
        2
    }
}
