//! 跨模块约定总览。
//!
//! 本模块不定义 trait、不引入新类型。它是**跨模块约定的权威来源**：
//! 方向符号、测度约定、delta 判定、维度流程、测试宏集中在这里，避免
//! 散落在 `light` / `bsdf` / `camera` / `film` / `integrator` 的文档里
//! 漂移。
//!
//! # 为什么需要这个模块
//!
//! MIS 的正确性依赖以下约定：
//!
//! - **方向符号统一**：`wi` / `wo` 在两侧的定义必须一致，否则
//!   `LiSample::incident` 与 `BSDF::eval` 的 `wi` 方向相反会导致
//!   余弦项符号错；
//! - **测度统一**：所有 `pdf` 是立体角测度，不同参考点的 PDF 不能
//!   相加；面积测度返回前必须转；
//! - **delta 判定统一**：delta 不参与 MIS，判定方式必须一致；
//! - **维度流程统一**：调用方按固定的维度推进顺序，否则 Sobol 维度
//!   错位——这是最难查的 bug 之一。
//!
//! 这些约定**类型系统抓不住**：`f32` 的 `pdf` 可以是任何测度，`Vec3`
//! 的方向可以是任何符号。文档 + 测试是唯一保证。
//!
//! # 模块结构
//!
//! - [`direction`]：方向符号（`wi` / `wo` / `direction`）；
//! - [`measure`]：测度约定（立体角 / 面积 / 离散概率，参考点）；
//! - [`delta`]：delta 判定；
//! - [`dimensions`]：维度流程（单次反弹、失败恢复、相机采样）；
//! - 测试宏：`assert_pdf_consistent!` / `assert_measure_conversion!`。
//!
//! # 使用方式
//!
//! 实现层在跨模块测试中使用宏：
//!
//! ```text
//! // 验证 BSDF 的 sample 与 pdf 一致
//! let s = bsdf.sample_f(&mut sampler, dim, mode).unwrap();
//! let expected = bsdf.pdf(wo, s.wi, mode);
//! assert_pdf_consistent!(s, expected);
//!
//! // 验证面积→立体角的转换
//! assert_measure_conversion!(pdf_omega, pdf_area, distance, cos_theta);
//! ```

// ═══════════════════════════════════════════════════════════════
// direction：方向符号
// ═══════════════════════════════════════════════════════════════

/// 方向符号约定。
///
/// # 渲染方程的标准符号
///
/// ```text
/// L_o(p, wo) = L_e(p, wo) + ∫_Ω f(p, wi, wo) · L_i(p, wi) · cos θ_i dω_i
/// ```
///
/// - `wi`：着色点入射方向（指向光源）；
/// - `wo`：着色点出射方向（指向相机）。
///
/// # 各模块的 `wi` / `wo`
///
/// | 模块 | 字段 | 方向 | 语义 |
/// |---|---|---|---|
/// | `BSDF::eval(wo, wi)` | `wi` | 着色点 → 光源 | 与渲染方程一致 |
/// | `BSDF::eval(wo, wi)` | `wo` | 着色点 → 相机 | 与渲染方程一致 |
/// | `LiSample::wi` | `wi` | 着色点 → 光源 | 与 `BSDF::eval` 的 `wi` 同向 |
/// | `LeSample::direction` | — | 光源 → 场景 | 无渲染方程符号，工程名 |
/// | `CameraSample.ray.dir` | — | 相机 → 场景 | 即 `wo`（从相机出射） |
/// | `CameraWiSample.wi` | `wi` | 场景 → 相机 | 即 `wi`（指向设备） |
/// | `AreaLight::emitted_radiance(wo)` | `wo` | 光源表面 → 着色点 | 等于 `-LiSample::wi` |
/// | `InfiniteLight::emitted_radiance(ray)` | `ray.dir` | 着色点 → 无限远 | 光线传播方向 |
/// | `InfiniteLight::sample_le` 的 `direction` | — | 无限远 → 场景 | `emitted_radiance` 的反向 |
///
/// # 两个关键反向
///
/// 1. **`LiSample::wi` 与 `AreaLight::emitted_radiance` 的 `wo`**：
///    `wi` 从着色点指向光源，`wo` 从光源表面射向着色点，两者反向
///    （`wo = -wi`）。原因：`wi` 是着色点侧的入射方向，`wo` 是光源侧的
///    出射方向，同一几何关系的两端描述。
///
/// 2. **`LeSample::direction` 与 `InfiniteLight::emitted_radiance` 的
///    `ray.dir`**：`direction` 从光源射向场景，`ray.dir` 从着色点射向
///    光源，两者反向（`ray.dir = -direction`）。原因：`sample_le` 从
///    光源出发，`emitted_radiance` 从着色点出发，参考点不同。
///
/// # 实现层的责任
///
/// - `LiSample::wi` 必须与 `BSDF::eval` 的 `wi` 同向（都用着色点 →
///   光源），否则 MIS 的余弦项符号错；
/// - `AreaLight::emitted_radiance(wo)` 的 `wo` 必须等于 `-LiSample::wi`，
///   否则 `sample_li` 的 `incident` 与 `emitted_radiance` 求值不一致；
/// - `InfiniteLight::sample_li_via_le` 已处理 `LeSample::direction` 到
///   `LiSample::wi` 的反向，实现层不要重复取反。
///
/// 跨模块测试必须校验 `LiSample::wi` 与 `BSDF::eval` 的 `wi` 同向。
pub mod direction {}

// ═══════════════════════════════════════════════════════════════
// measure：测度约定
// ═══════════════════════════════════════════════════════════════

/// 测度约定。
///
/// # 所有 `pdf` 都是立体角测度
///
/// 除非字段文档明确标注其他测度，所有 `pdf` 字段是**从某个参考点看的
/// 立体角测度**。面积测度（面光源面积、镜头面积）在返回前必须转换成
/// 立体角测度。
///
/// # 测度与参考点
///
/// | 字段 | 测度 | 参考点 | 用途 |
/// |---|---|---|---|
/// | `LiSample::pdf` | 立体角 | 着色点 | MIS，与 `BxDFSample::pdf` 同参考点 |
/// | `BxDFSample::pdf` | 立体角 | 着色点 | MIS，与 `LiSample::pdf` 同参考点 |
/// | `LeSample::pdf` | 立体角 | 光源点 | 光源侧子路径，BDPT 连接需几何项 |
/// | `CameraSample::pdf` | 立体角 | 相机 | 相机出射方向 |
/// | `CameraWiSample::pdf` | 立体角 | 场景点 | 相机入射方向，与 `Light::pdf_le` 对称 |
/// | `LightSelection::light_pdf` | 离散概率 | — | 选择该光源的概率 |
///
/// # 同参考点才能相加
///
/// - **`LiSample::pdf` + `BxDFSample::pdf`**：同参考点（着色点），
///   可以在 MIS 权重中直接相加；
/// - **`LeSample::pdf` + `CameraWiSample::pdf`**：不同参考点（光源点
///    vs 场景点），不能相加——BDPT 端点连接用几何项转换；
/// - **`CameraSample::pdf` + `BxDFSample::pdf`**：不同参考点（相机
///    vs 着色点），不能相加——MIS 只在同一着色点的光源采样与 BSDF
///   采样之间做。
///
/// # 面积 ↔ 立体角
///
/// 面光源的采样通常在面积上完成，`pdf_area` 必须转成立体角：
///
/// ```text
/// pdf_omega = pdf_area × d² / |cos θ_light|
/// ```
///
/// 其中：
///
/// - `d`：着色点到光源点的距离；
/// - `θ_light`：光源表面法线与 `-wi` 的夹角（光源法线朝向着色点）。
///
/// 用 [`area_to_solid_angle_factor`] 计算转换因子。
///
/// # 总 PDF 的组合
///
/// 光源采样路径的总 PDF：
///
/// ```text
/// total_pdf = light_pdf × LiSample::pdf
/// ```
///
/// - `light_pdf`：离散概率，选择该光源的概率；
/// - `LiSample::pdf`：立体角测度，光源内部采样方向的 PDF；
/// - 两者**相乘**得到"从着色点看，采样到 `wi` 方向"的总立体角测度 PDF；
/// - MIS 计算时与 `BSDF::pdf(wi)` 相加，两者都是着色点参考的立体角。
///
/// 离散概率与立体角测度是**相乘**关系，不是转换关系。面积与立体角
/// 才是转换关系。
///
/// # delta 例外
///
/// delta 光源（`LightType::is_delta()`）的 `LiSample::pdf` 无有限值，
/// 不参与 MIS。总 PDF 无意义，调用方应直接以"采样命中"的权重贡献。
pub mod measure {}

/// 面积测度 → 立体角测度的转换因子。
///
/// `pdf_omega = pdf_area × area_to_solid_angle_factor(distance, cos_theta)`
///
/// 参数：
/// - `distance`：着色点到光源点的距离；
/// - `cos_theta`：光源表面法线与 `-wi` 的夹角余弦（光源法线朝向着色点）。
///
/// 返回 `d² / |cos θ|`。调用方乘以 `pdf_area` 得到 `pdf_omega`。
///
/// # 退化
///
/// `cos_theta` 接近 0 时转换因子趋于无穷。调用方应在采样时拒绝
/// 退化方向（`|cos_theta| < epsilon`），而不是在转换时处理。
///
/// 本函数不检查退化——它是纯数学转换，退化处理是采样层的责任。
#[inline]
pub fn area_to_solid_angle_factor(distance: f32, cos_theta: f32) -> f32 {
    distance * distance / cos_theta.abs()
}

// ═══════════════════════════════════════════════════════════════
// delta：delta 判定
// ═══════════════════════════════════════════════════════════════

/// delta 判定约定。
///
/// # 什么是 delta
///
/// delta 分布是"集中在单一方向或点上的概率分布"。物理上对应：
///
/// - **点光源**：位置成 delta，方向连续；
/// - **方向光**：方向成 delta，位置连续；
/// - **镜面 BSDF**：反射/透射方向成 delta；
/// - **针孔相机**：方向由像素唯一确定，成 delta。
///
/// # 为什么不参与 MIS
///
/// delta 分布的 `pdf` 无有限值（在某点趋于无穷）。MIS 需要比较两个
/// 连续分布的 PDF，delta 无法参与。所以：
///
/// - **delta 光源**：不参与"光源采样 vs BSDF 采样"的 MIS；
/// - **delta BSDF**：不参与"BSDF 采样 vs 光源采样"的 MIS；
/// - **delta 相机**：`pdf_wo` 返回 0，不参与"相机采样 vs 其他"的 MIS。
///
/// # 判定方式
///
/// | 模块 | 判定方法 |
/// |---|---|
/// | 光源 | `LightType::is_delta()` |
/// | BSDF | `BxDFFlags::is_pure_delta()`（命中分量的 flags） |
/// | 相机 | `pdf_wo` 返回 0 |
/// | 无限远光源 | `LightType::is_delta()`（方向光是 delta） |
///
/// # 不要用 `pdf == 0.0` 反推
///
/// 非 delta 分布在某些退化方向也可能返回接近 0 的 PDF。判定是否
/// delta 必须用显式的类型标签（`is_delta` / `is_pure_delta`），不能
/// 靠 `pdf == 0.0`。两者的语义不同：
///
/// - `pdf == 0.0`：该方向不可采到（连续分布的正常情况）；
/// - `is_delta() == true`：整个分布是 delta，`pdf` 无有限值。
///
/// # 双重计数避免
///
/// delta 光源 / BSDF 的贡献**必须**通过采样命中路径获取（NEE 覆盖
/// 不到 delta 方向）。非 delta 的贡献需要 MIS 或显式的双重计数避免
/// （如 PT 的 `specular_bounce` 标志）。实现层负责这个逻辑，契约层
/// 只提供判定接口。
pub mod delta {}

// ═══════════════════════════════════════════════════════════════
// dimensions：维度流程
// ═══════════════════════════════════════════════════════════════

/// 维度流程约定。
///
/// # 核心原则：追踪实际消耗
///
/// 所有需要 `Sampler` 的组件，从传入的 `dimension` 开始消耗维度，通过
/// 返回值（`consumed` 字段或 `(value, consumed)` 元组）报告实际消耗。
/// 调用方用 `dimension + consumed` 推进基址。
///
/// 这个原则保证 Sobol / Halton 等低差异采样器的维度不错位——每个
/// 组件精确知道消耗多少，调用方按实际推进，不依赖预分配。
///
/// # 各组件的消耗
///
/// | 组件 | 方法 | 消耗 | 报告方式 |
/// |---|---|---|---|
/// | 光源选择 | `LightSampler::sample` | 动态（BVH 采样器 > 1） | `LightSelection::consumed` |
/// | 光源采样 | `Light::sample_li` | 0（调用方预取 `Vec2`） | — |
/// | BSDF 采样 | `BSDF::sample_f` | 动态（组合 BSDF） | `BxDFSample::consumed` |
/// | 单 BxDF 采样 | `BxDF::sample_f` | 通常 2（方向采样） | `BxDFSample::consumed` |
/// | 相机采样 | `Camera::gen_ray` | 0（调用方预取像素 + 透镜 + 时间） | — |
///
/// # 单次反弹模板（PT）
///
/// 假设进入时维度基址为 `dim`：
///
/// ```text
/// 1. 光源选择：
///    let sel = scene.light_sampler().sample(ctx, sampler)?
///    dim1 = dim + sel.consumed
///
/// 2. 取光源采样样本：
///    let sample_2d = sampler.next_2d(dim1)
///    let li = light.sample_li(ctx, sample_2d, wl)?
///    dim2 = dim1 + 2
///
/// 3. BSDF 采样：
///    let bs = bsdf.sample_f(sampler, dim2, mode)?
///    dim3 = dim2 + bs.consumed
///
/// 4. 俄罗斯轮盘赌：
///    let rr_sample = sampler.next_f32(dim3)
///    dim4 = dim3 + 1
/// ```
///
/// 单次反弹消耗 `sel.consumed + 2 + bs.consumed + 1` 维。路径追踪的
/// 完整维度是 `dim4` 作为下一次反弹的起点，重复上述流程。
///
/// # 失败恢复
///
/// 采样失败时，维度基址需要用 `max_sample_dimensions()` 估计：
///
/// - `LightSampler::sample` 返回 `None`：
///   `dim1 = dim + light_sampler.max_sample_dimensions()`；
/// - `BSDF::sample_f` 返回 `None`：
///   `dim3 = dim2 + bsdf.max_sample_dimensions()`；
/// - `Light::sample_li` 返回 `None`：调用方已预取 `Vec2`，`dim2 = dim1 + 2`
///   不变，跳过该光源贡献。
///
/// 失败恢复是**保守估计**，不要求精确。精确追踪只在成功路径上。
/// 路径通常失败后终止，保守恢复的维度错位不影响最终结果。
///
/// # 相机采样（在积分器之外）
///
/// 相机采样通常在渲染循环中完成，独立于积分器的维度流程：
///
/// ```text
/// // 渲染循环（伪代码）
/// let dim0 = 0;
/// let pixel_sample = sampler.next_2d(dim0);         // dim1 = dim0 + 2
/// let lens_sample = sampler.next_2d(dim1);          // dim2 = dim1 + 2
/// let time_sample = sampler.next_f32(dim2);         // dim3 = dim2 + 1
/// let time = camera.sample_time(time_sample);
/// let cam_sample = camera.gen_ray(pixel_sample, lens_sample, time, wl)?;
/// let radiance = integrator.li(scene, &cam_sample.ray, sampler, wl);
/// film.add_sample(cam_sample.film_position, radiance * cam_sample.weight, 1.0 / spp);
/// ```
///
/// 相机采样消耗 5 维（像素 2D + 透镜 2D + 时间 1D）。积分器从 `dim3`
/// 开始按单次反弹模板消耗。
///
/// # BDPT 的维度流程
///
/// BDPT 比 PT 复杂（eye subpath + light subpath + 连接），但原则相同：
/// 每个组件报告 `consumed`，调用方按实际推进。BDPT 的详细流程由
/// 实现层定义，不在本契约层约束。
///
/// # 分层采样
///
/// 如果调用方需要静态分配维度块（如分层采样），用
/// `max_sample_dimensions()` 预留上界。是否启用分层是调用方的配置
/// 决策，契约层不感知。
pub mod dimensions {}

// ═══════════════════════════════════════════════════════════════
// 测试宏
// ═══════════════════════════════════════════════════════════════

/// 断言采样的 `pdf` 与期望值一致。
///
/// 用法：
///
/// ```text
/// let s = bsdf.sample_f(sampler, dim, mode).unwrap();
/// let expected = bsdf.pdf(wo, s.wi, mode);
/// assert_pdf_consistent!(s, expected);
/// ```
///
/// 用相对误差（`1e-5 × max(|expected|, 1.0)`），容忍极小的 `pdf`
/// （接近 0 但非零）和极大的 `pdf`（如接近光源时）。
///
/// 跨实现测试必须校验 `sample.pdf == pdf(wi)`，这是 MIS 正确性的
/// 根基。
#[macro_export]
macro_rules! assert_pdf_consistent {
    ($sample:expr, $expected:expr) => {{
        let s = $sample;
        let expected = $expected;
        let tolerance = 1e-5 * expected.abs().max(1.0);
        assert!(
            (s.pdf - expected).abs() < tolerance,
            "pdf mismatch: sample.pdf={}, expected={}, diff={}",
            s.pdf,
            expected,
            (s.pdf - expected).abs()
        );
    }};
}

/// 断言面积测度到立体角测度的转换正确。
///
/// 用法：
///
/// ```text
/// // 面光源在面积上采样，返回 pdf_omega（已转换）
/// assert_measure_conversion!(pdf_omega, pdf_area, distance, cos_theta);
/// ```
///
/// 校验 `pdf_omega == pdf_area × d² / |cos θ|`，用相对误差
/// （`1e-5 × max(|expected|, 1.0)`）。
///
/// 面光源的 `sample_li` 返回的 `pdf` 必须是立体角测度；如果内部在
/// 面积上采样，返回前必须转换。跨实现测试用本宏校验。
#[macro_export]
macro_rules! assert_measure_conversion {
    ($pdf_omega:expr, $pdf_area:expr, $distance:expr, $cos_theta:expr) => {{
        let actual = $pdf_omega;
        let pdf_area = $pdf_area;
        let distance = $distance;
        let cos_theta = $cos_theta;
        let expected =
            pdf_area * $crate::conventions::area_to_solid_angle_factor(distance, cos_theta);
        let tolerance = 1e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() < tolerance,
            "measure conversion mismatch: pdf_omega={}, expected={} \
             (pdf_area={}, distance={}, cos_theta={}, diff={})",
            actual,
            expected,
            pdf_area,
            distance,
            cos_theta,
            (actual - expected).abs()
        );
    }};
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────── area_to_solid_angle_factor ───────────

    #[test]
    fn factor_unit_distance_perpendicular() {
        // d = 1, cos = 1 → 因子 = 1
        assert_eq!(area_to_solid_angle_factor(1.0, 1.0), 1.0);
    }

    #[test]
    fn factor_distance_squared() {
        // d = 2, cos = 1 → 因子 = 4
        assert_eq!(area_to_solid_angle_factor(2.0, 1.0), 4.0);
    }

    #[test]
    fn factor_inverse_cosine() {
        // d = 1, cos = 0.5 → 因子 = 2
        assert_eq!(area_to_solid_angle_factor(1.0, 0.5), 2.0);
    }

    #[test]
    fn factor_uses_abs_cosine() {
        // cos 为负（光源法线反向）时用绝对值
        assert_eq!(
            area_to_solid_angle_factor(1.0, -0.5),
            area_to_solid_angle_factor(1.0, 0.5)
        );
    }

    // ─────────── assert_pdf_consistent! ───────────

    struct MockSample {
        pdf: f32,
        wi: glam::Vec3,
    }

    #[test]
    fn pdf_consistent_passes_on_exact_match() {
        let s = MockSample {
            pdf: 0.5,
            wi: glam::Vec3::ZERO,
        };
        assert_pdf_consistent!(s, 0.5);
    }

    #[test]
    fn pdf_consistent_passes_within_tolerance() {
        let s = MockSample {
            pdf: 0.5,
            wi: glam::Vec3::ZERO,
        };
        // 相对误差 1e-5 × max(0.5, 1.0) = 1e-5，0.5 + 1e-7 在容差内
        assert_pdf_consistent!(s, 0.5 + 1e-7);
    }

    #[test]
    #[should_panic(expected = "pdf mismatch")]
    fn pdf_consistent_fails_on_mismatch() {
        let s = MockSample {
            pdf: 0.5,
            wi: glam::Vec3::ZERO,
        };
        assert_pdf_consistent!(s, 0.6);
    }

    #[test]
    fn pdf_consistent_handles_large_values() {
        // 大 pdf（接近光源），相对误差容忍
        let s = MockSample {
            pdf: 1000.0,
            wi: glam::Vec3::ZERO,
        };
        // 1e-5 × 1000 = 0.01 的绝对容差
        assert_pdf_consistent!(s, 1000.005);
    }

    #[test]
    fn pdf_consistent_handles_small_values() {
        // 极小 pdf，绝对容差 1e-5 × max(0.001, 1.0) = 1e-5
        let s = MockSample {
            pdf: 0.001,
            wi: glam::Vec3::ZERO,
        };
        assert_pdf_consistent!(s, 0.001 + 1e-7);
    }

    // ─────────── assert_measure_conversion! ───────────

    #[test]
    fn measure_conversion_passes_on_exact_match() {
        // d = 2, cos = 1 → 因子 4，pdf_area = 0.25 → pdf_omega = 1.0
        assert_measure_conversion!(1.0, 0.25, 2.0, 1.0);
    }

    #[test]
    fn measure_conversion_passes_within_tolerance() {
        assert_measure_conversion!(1.0 + 1e-7, 0.25, 2.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "measure conversion mismatch")]
    fn measure_conversion_fails_on_mismatch() {
        // 期望 1.0，实际 0.5
        assert_measure_conversion!(0.5, 0.25, 2.0, 1.0);
    }

    #[test]
    fn measure_conversion_oblique_angle() {
        // d = 1, cos = 0.5 → 因子 2，pdf_area = 0.5 → pdf_omega = 1.0
        assert_measure_conversion!(1.0, 0.5, 1.0, 0.5);
    }

    // ─────────── 宏的跨模块可用性 ───────────

    #[test]
    fn macros_expand_in_this_module() {
        // 宏在本模块内可用（#[macro_export] 导出到 crate root）
        let s = MockSample {
            pdf: 1.0,
            wi: glam::Vec3::ZERO,
        };
        assert_pdf_consistent!(s, 1.0);
        assert_measure_conversion!(1.0, 1.0, 1.0, 1.0);
    }
}
