use std::sync::Arc;

use glam::Vec3;
use kzray_abs::{BSDF, BSDFBuilder, BxDF, BxDFEntry, BxDFFlags, BxDFSample, Library, Sampler, ScalarAlgebra, SpectralAlgebra, SurfaceHit, TransportMode};
use kzray_data::TextureId;


/// 完美镜面反射。`eval` 返回零（delta 无连续值），采样按确定性反射方向。
///
/// # BRDF 定义
///
/// 镜面反射的 BRDF 是狄拉克 delta：
///
/// ```text
/// f(wo, wi) = albedo × δ(wi - reflect(wo)) / |cos θi|
/// ```
///
/// `eval` 对连续方向返回 0（delta 在非反射方向的积分贡献为 0）。
/// `sample_f` 返回反射方向，`value = albedo / cos`（隐含 `1/cos` 因子），
/// `BSDF::sample_f` 的 delta 分支乘回 `cos` 得到 `weight = albedo`——
/// 这是 specular BRDF 的标准约定（PBRT / Cycles 同）。
///
/// # 能力标签
///
/// `REFLECTION | SPECULAR`——纯 delta，`is_pure_delta() == true`。
/// `PathIntegrator` 的 `specular_bounce` 会识别它，NEE 时跳过 MIS。
///
/// # 维度
///
/// 确定性反射，**不消耗随机数**（`consumed = 0`）。`BSDF::sample_f` 仍
/// 消耗 1 维选择 BxDF 条目（与其它材质统一），总 `consumed = 1`。
#[derive(Clone, Copy, Debug)]
pub struct Mirror<S: ScalarAlgebra> {
    pub albedo: S,
}

impl<S: ScalarAlgebra> Mirror<S> {
    pub fn new(albedo: S) -> Self {
        Mirror { albedo }
    }

    /// 半球-半球反射率。完美镜面：R = albedo。
    ///
    /// 固有方法，不是 [`BxDF`] trait 方法；材质构造 `BxDFEntry` 时调用一次。
    pub fn hemispherical_reflectance(&self) -> S {
        self.albedo
    }
}

impl<S: ScalarAlgebra> BxDF<S> for Mirror<S> {
    fn flags(&self) -> BxDFFlags {
        BxDFFlags::REFLECTION | BxDFFlags::SPECULAR
    }

    fn eval(&self, _wo: Vec3, _wi: Vec3, _mode: TransportMode) -> S {
        // delta 在连续方向测度下密度为 0——非反射方向的积分贡献为零，
        // 反射方向的密度无穷大，无法数值表示。
        S::zero()
    }

    fn sample_f(
        &self,
        wo: Vec3,
        _sampler: &mut dyn Sampler,
        _dimension: u32,
        _mode: TransportMode,
    ) -> Option<BxDFSample<S>> {
        if wo.z <= 0.0 {
            return None;
        }
        // 反射：wi = -wo + 2*(wo·n)*n，局部 n = (0,0,1)
        // 即 wi = (-wo.x, -wo.y, wo.z)。反射保持 z 分量。
        let wi = Vec3::new(-wo.x, -wo.y, wo.z);
        let cos = wi.z; // = wo.z > 0
        if cos <= f32::EPSILON {
            // 掠射：反射方向退化，数值不稳定
            return None;
        }
        // value = albedo / cos（隐含 specular BRDF 的 1/cos 因子）
        //
        // BSDF::sample_f 的 delta 分支：
        //   weight = value * cos / weight_selected
        //          = (albedo/cos) * cos / w = albedo / w
        // 对单个 Mirror 条目，w = 1.0，weight = albedo。正确。
        let value = self.albedo.scale(S::Scalar::from(1.0 / cos));
        Some(BxDFSample {
            wi,
            value,
            // weight 会被 BSDF::sample_f 覆盖，这里给 value 占位
            weight: value,
            pdf: 0.0,
            // delta 无连续 pdf，pdf_is_exact 无意义，保持 true 不参与 MIS
            pdf_is_exact: true,
            flags: self.flags(),
            eta: 1.0,
            // 确定性反射，不消耗随机数
            consumed: 0,
        })
    }

    fn pdf(&self, _wo: Vec3, _wi: Vec3, _mode: TransportMode) -> f32 {
        // delta 对连续方向无 pdf
        0.0
    }

    fn rho(
        &self,
        _wo: Vec3,
        _sampler: &mut dyn Sampler,
        _dimension: u32,
        _mode: TransportMode,
    ) -> (S, u32) {
        (self.albedo, 0)
    }

    fn max_sample_dimensions(&self) -> u32 {
        // 确定性反射，`sample_f` 不消耗随机数
        0
    }
}


/// 完美镜面材质构建器。
///
/// 持有 `albedo: TextureId`，`build` 时从 `Library` 求值。构造单个
/// `Mirror` BxDF，包装成 `BSDF`。
///
/// # 双面语义
///
/// 与 Lambert 一致：`wo` 与 `shading_normal` 反向时翻转法线，双面镜面。
/// 单面镜面（如背面为黑）留作后续扩展。
///
/// # 泛型
///
/// `MirrorBSDFBuilder` 不带 `S`——`TextureId` 是裸 u32。同一实例在
/// `impl<S: SpectralAlgebra> BSDFBuilder<S>` 下可服务任意 `S`。
#[derive(Clone, Copy, Debug)]
pub struct MirrorBSDFBuilder {
    albedo: TextureId,
}

impl MirrorBSDFBuilder {
    pub fn new(albedo: TextureId) -> Self {
        MirrorBSDFBuilder { albedo }
    }

    pub fn albedo(&self) -> TextureId {
        self.albedo
    }
}

impl<S: SpectralAlgebra> BSDFBuilder<S> for MirrorBSDFBuilder {
    fn build(
        &self,
        hit: &SurfaceHit,
        wo: Vec3,
        _sampler: &mut dyn Sampler,
        _wavelengths: &S::WavelengthContext,
        lib: &Library<S>,
    ) -> BSDF<S> {
        let albedo = lib.albedo(self.albedo).eval(hit.uv, None);

        let normal = hit.shading_normal;
        // 双面：翻转法线使 wo 落在正向
        let normal = if wo.dot(normal) > 0.0 { normal } else { -normal };

        let mirror = Mirror::new(albedo);
        let entry = BxDFEntry::new(
            Arc::new(mirror),
            1.0,
            // 镜面的半球-半球反射率 = albedo（完美反射，无吸收）
            mirror.hemispherical_reflectance(),
        );

        BSDF::new(vec![entry], wo, normal)
            .expect("Mirror BSDF construction should succeed")
    }
}
