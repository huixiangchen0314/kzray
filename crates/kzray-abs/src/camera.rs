//! 相机采样契约。
//!
//! 相机是"逆向的光源"：`Light` 从着色点采样入射方向 `wi` 并求 `L_i`，
//! `Camera` 从胶片采样出射方向 `wo` 并求重要性。两者在 BDPT 的两端
//! 对称：光源侧用 `Importance` 传输方向，相机侧用 `Radiance`。
//!
//! # 方向符号
//!
//! 相机侧沿用渲染方程的方向符号：
//!
//! - **`wo`**：相机出射方向，从相机指向场景。对应 `CameraSample` 的
//!   `ray.dir`。与着色点侧的 `wo`（着色点指向相机）是**同一方向的反向**
//!   ——从着色点看是 `wo`，从相机看是 `-wo`。这里 `wo` 表示"设备出射"，
//!   与 `Light::LeSample::direction`（光源出射）对称。
//! - **`wi`**：相机入射方向，从场景点指向相机。对应
//!   `CameraWiSample::wi`。与 `LiSample::wi`（着色点指向光源）是同一
//!   语义——"指向设备"。
//!
//! 不用 `we` / `wi` 的原因：`we`（`W_e` 的缩写）是**函数名**，不是
//! 方向名。把方向命名为 `wo`，把 `W_e` 的求值放在 `CameraSample::weight`
//! 字段上，语义更清晰——方向是方向，函数值是函数值。
//!
//! # 职责分层
//!
//! 相机能力按"谁需要"拆成独立 trait，不在基础接口上堆可选方法：
//!
//! - [`Camera`]：基础契约。`gen_ray` / `pdf_wo` / `sample_time` /
//!   `film_size`。**PT 只需要这一层**——从相机出射追踪 primary ray，
//!   方向 PDF 是唯一的（像素采样决定方向，镜头采样决定起点）。
//! - [`SampleWiCamera`]：相机入射方向采样。`sample_wi` / `pdf_wi`。
//!   **BDPT 需要**——从场景点采样指向相机的方向，需要从场景点看相机的
//!   方向立体角 PDF。这是不同于 `pdf_wo` 的另一个测度，参考点不同。
//! - [`CameraRayDifferential`]：微分光线生成。`gen_ray_differential`。
//!   **纹理过滤 / AOV 需要**——相机才知道自己的微分形式（透视相机的
//!   解析微分、真实透镜相机的数值微分），调用方不该耦合。
//!
//! 拆成独立 trait 的原因：Rust 里"可选能力"用 trait 分离比"默认实现
//! 或 `unreachable!()`"更干净——PT 不会碰 `sample_wi`，BDPT 不会碰
//! 微分，每个 trait 的职责单一。
//!
//! # PDF 测度
//!
//! 所有 `pdf` 字段是**立体角测度**，与 `Light` 一致：
//!
//! - `Camera::pdf_wo(ray)` 是从**相机**看的方向立体角密度；
//! - `SampleWiCamera::pdf_wi(reference, wi)` 是从**场景点** `reference`
//!   看相机的方向立体角密度。
//!
//! 两者参考点不同，不能直接相加。BDPT 端点连接时用 `pdf_wi`（从场景点
//! 看相机），与 `Light::pdf_le`（从光源看场景）对称。
//!
//! # 不采样波长
//!
//! `gen_ray` / `sample_wi` 接收 `&S::WavelengthContext`，**不修改**。
//! 波长采样策略属于积分器或独立模块，不属于相机的几何采样。pbrt 让
//! 相机写入 `SampledWavelengths`，那是 C++ 的架构选择；我们的相机只
//! 消费波长上下文，不生产它。
//!
//! # 与 pbrt 的差异
//!
//! | 点 | pbrt | 我们 |
//! |---|---|---|
//! | 波长采样 | 相机写入 `SampledWavelengths` | 相机只读 `&S::WavelengthContext` |
//! | 双 PDF | `PDF_We` 返回 `(pdfPos, pdfDir)` | 基础 trait 只返回方向 PDF；入射侧分离到 `SampleWiCamera` |
//! | 微分光线 | `GenerateRayDifferential` 在基础接口 | 分离到 `CameraRayDifferential` |
//! | 持有 Film | `GetFilm()` | 解耦：只 `film_size()` 查询分辨率 |
//! | `SampleTime` | 基础接口 | 保留（快门区间是相机属性，采样纯函数） |
//! | `Approximate_dp_dxy` | 基础接口 | 不做（相机特定的微分优化，不属于抽象层） |

use glam::{UVec2, Vec2, Vec3};

use crate::ray::{Ray, RayDifferential};
use crate::spectrum::SpectralAlgebra;

// ═══════════════════════════════════════════════════════════════
// 采样结果
// ═══════════════════════════════════════════════════════════════

/// 单次相机出射采样的结果。
///
/// 对应渲染方程中的相机侧重要性 `W_e(p, wo)`。PT 的 primary ray 路径
/// 只需要 `ray` / `film_position`；BDPT 需要 `pdf` / `weight` 做端点
/// 连接与 MIS。
///
/// `ray.dir` 就是相机出射方向 `wo`——从相机指向场景。字段不再单独
/// 存 `wo`，因为 `ray.dir` 已经是它；`pdf` 是 `wo` 的立体角密度。
#[derive(Clone, Copy, Debug)]
pub struct CameraSample<S: SpectralAlgebra> {
    /// 生成的光线。`ray.dir` 即相机出射方向 `wo`（从相机指向场景）；
    /// `ray.time` 已由调用方传入的时刻填充。
    pub ray: Ray,
    /// 胶片上的连续位置（像素坐标，含亚像素）。
    ///
    /// 范围 `[0, width] × [0, height]`。供 `Film::add_sample` 使用；
    /// 整数化和 filter 重建由 `Film` 内部处理。
    pub film_position: Vec2,
    /// 出射方向 `wo` 被相机采样到的概率密度，**立体角测度**。
    ///
    /// 从**相机**看的方向立体角密度。与 `BxDFSample::pdf` 同测度、
    /// 不同参考点。针孔相机为 delta 分布（无有限值），此时 `pdf` 无意义，
    /// 不参与 MIS。
    pub pdf: f32,
    /// 相机侧光谱重要性权重 `W_e(p, wo)`。
    ///
    /// PT 的 primary ray 路径通常不需要；BDPT 的相机子路径需要。
    /// 针孔相机在胶片内通常为 `S::one()`，有限孔径相机可能随方向变化。
    pub weight: S,
}

/// 单次相机入射方向采样的结果（BDPT 用）。
///
/// 与 [`CameraSample`] 的区别：`CameraSample` 从相机出射（`wo`），方向
/// 由像素决定；`CameraWiSample` 从场景点入射（`wi`），方向指向相机。
/// 两者是 BDPT 两端的对偶。
///
/// 与 `Light::LeSample` 对称：`LeSample` 从光源出射，`CameraWiSample`
/// 从场景点入射指向相机。
#[derive(Clone, Copy, Debug)]
pub struct CameraWiSample<S: SpectralAlgebra> {
    /// 从场景点 `reference` 指向相机的单位入射方向 `wi`。
    ///
    /// 与 `CameraSample::ray.dir`（`wo`，相机出射方向）**反向**：
    /// `wo` 从相机指向场景，`wi` 从场景指向相机。这是相机两侧的
    /// 出入方向，与渲染方程 `L_o(p, wo)` / `L_i(p, wi)` 的用法一致。
    pub wi: Vec3,
    /// 沿 `wi` 方向到达相机的重要性 `W_e`。
    pub weight: S,
    /// 采样到 `wi` 的 PDF，**立体角测度**（从 `reference` 看相机）。
    ///
    /// 与 `Camera::pdf_wo` 参考点不同：`pdf_wo` 从相机看，`pdf_wi` 从
    /// `reference` 看。BDPT 端点连接用这个值。
    pub pdf: f32,
    /// `reference` 到相机的距离。
    ///
    /// 应等于 `(camera_position - reference).length()`；实现时必须保持
    /// 一致，否则 BDPT 几何项 `G = cos_s × cos_c / distance²` 会出错。
    /// 跨实现测试应校验此等式。
    pub distance: f32,
}

// ═══════════════════════════════════════════════════════════════
// Camera：基础契约（PT 必需）
// ═══════════════════════════════════════════════════════════════

/// 相机基础采样契约。PT 必需。
///
/// 提供从相机出射方向的采样（`gen_ray`）和反查（`pdf_wo`），以及快门
/// 时间采样（`sample_time`）和胶片分辨率查询（`film_size`）。
///
/// # 与 `Light` 的对称
///
/// | Light | Camera |
/// |---|---|
/// | `sample_le(u)`（光源 → 场景） | `gen_ray(...)`（相机 → 场景） |
/// | `pdf_le(dir)` | `pdf_wo(ray)` |
///
/// 相机入射方向采样见 [`SampleWiCamera`]（BDPT 用）。
///
/// # 传输方向
///
/// 本 trait 的方法隐含 `Radiance` 传输方向（从相机出发，追踪辐射亮度），
/// 无 `TransportMode` 参数。与 `Light` 侧的隐含方向对应：
///
/// - `Light::sample_li` / `pdf_li` 隐含 `Radiance`；
/// - `InfiniteLight::sample_le` / `pdf_le` 隐含 `Importance`；
/// - `Camera::gen_ray` / `pdf_wo` 隐含 `Radiance`。
///
/// # PDF 一致性契约
///
/// 对任意 `s = gen_ray(pixel_sample, lens_sample, time, wl)`：
///
/// ```text
/// pdf_wo(&s.ray) == s.pdf
/// ```
///
/// 针孔相机（delta）除外：`pdf` 无有限值，`pdf_wo` 返回 0。
/// 跨实现测试必须校验。
///
/// # Object-safety
///
/// `Camera` 是 object-safe 的：`gen_ray` / `pdf_wo` 都不是泛型方法，
/// `Box<dyn Camera<S>>` 可用。
pub trait Camera<S: SpectralAlgebra>: Send + Sync {
    /// 生成 primary ray（相机出射方向 `wo`）。
    ///
    /// 参数：
    /// - `pixel_sample`：像素内 2D 均匀样本 `[0, 1)²`，决定亚像素位置；
    /// - `lens_sample`：透镜 2D 均匀样本 `[0, 1)²`，决定孔径采样位置
    ///   （针孔相机忽略）；
    /// - `time`：快门区间内的归一化时刻 `[0, 1]`，填充 `Ray::time`；
    ///   调用方用 [`Camera::sample_time`] 从均匀样本生成；
    /// - `wavelengths`：波长上下文。**本方法不修改它**，只读取用于
    ///   光谱相关的相机模型（如色散相机）。
    ///
    /// 返回 `None` 表示采样失败（如样本落在胶片有效区域外、镜头样本
    /// 退化）。失败时调用方跳过该路径。
    ///
    /// `pdf` 是**立体角测度**（从相机看的方向 `wo`）。针孔相机的方向
    /// 由像素唯一确定，`pdf` 为 delta，调用方不应读取此字段参与 MIS。
    ///
    /// 与 `Light::sample_le` 的分工：本方法从调用方预取的样本生成光线，
    /// 不管理 `Sampler` 状态、不消耗维度。相机采样是固定维度预算
    /// （像素 2D + 透镜 2D + 时间 1D），不需要 `consumed`。
    fn gen_ray(
        &self,
        pixel_sample: Vec2,
        lens_sample: Vec2,
        time: f32,
        wavelengths: &S::WavelengthContext,
    ) -> Option<CameraSample<S>>;

    /// 给定光线，返回相机采样其出射方向 `wo` 的 PDF（立体角测度）。
    ///
    /// 从**相机**看的方向立体角密度。用于 MIS：光源采样路径用
    /// `light_pdf × pdf_li`，BSDF 采样路径用 `BSDF::pdf`，相机路径用
    /// `pdf_wo` 反查。
    ///
    /// 针孔相机返回 0：连续方向采样采不到 delta 方向。
    ///
    /// 契约：对任意 `s = gen_ray(...)`，应有
    /// `pdf_wo(&s.ray) == s.pdf`（delta 相机除外）。
    ///
    /// 与 [`SampleWiCamera::pdf_wi`] 的区别：本方法从相机看（`wo`），
    /// 后者从场景点看（`wi`）。参考点不同，不能混用。
    fn pdf_wo(&self, ray: &Ray) -> f32;

    /// 快门区间内的时刻采样。
    ///
    /// `u` 是 1D 均匀样本 `[0, 1)`；返回 `[0, 1]` 内的归一化时刻。
    /// 无运动模糊的相机应返回 `0.0`（快门瞬间开启关闭）。
    ///
    /// 快门区间（如 `[t_open, t_close]`）是相机属性，`sample_time` 是
    /// 纯函数：`time = t_open + u × (t_close - t_open)`。调用方用
    /// `camera.sample_time(u)` 生成时间，再传给 `gen_ray`。
    fn sample_time(&self, u: f32) -> f32;

    /// 胶片分辨率。
    ///
    /// 与 `Film::resolution` 应一致；调用方可以用两者校验。
    fn film_size(&self) -> UVec2;
}

// ═══════════════════════════════════════════════════════════════
// SampleWiCamera：相机入射方向采样（BDPT 用）
// ═══════════════════════════════════════════════════════════════

/// 相机入射方向采样能力。BDPT 用。
///
/// 从场景点 `reference` 采样指向相机的方向 `wi`，与 `Light::sample_le`
/// 对称：光源侧从光源出射，相机侧从场景点入射。BDPT 连接两端时，光源
/// 侧用 `sample_le` / `pdf_le`，相机侧用 `sample_wi` / `pdf_wi`。
///
/// # 与 `Camera` 分离的理由
///
/// `pdf_wo`（从相机看）和 `pdf_wi`（从场景点看）是**不同参考点的
/// 立体角测度**，不能相加。PT 只用 `pdf_wo`，BDPT 才需要 `pdf_wi`。
/// 分离 trait 让 PT 不感知 `sample_wi` 的存在，BDPT 显式要求能力。
///
/// # PDF 一致性契约
///
/// 对任意 `s = sample_wi(reference, sample, wl)`：
///
/// ```text
/// pdf_wi(reference, s.wi) == s.pdf
/// ```
///
/// 针孔相机除外：`pdf` 无有限值，`pdf_wi` 返回 0。
/// 跨实现测试必须校验。
///
/// # Object-safety
///
/// `SampleWiCamera` 是 object-safe 的。因为 `SampleWiCamera: Camera`，
/// trait upcasting 可以把 `Box<dyn SampleWiCamera<S>>` 转成
/// `Box<dyn Camera<S>>`（Rust 1.86+ 稳定）。
pub trait SampleWiCamera<S: SpectralAlgebra>: Camera<S> {
    /// 从场景点 `reference` 采样指向相机的方向 `wi`。
    ///
    /// 参数：
    /// - `reference`：场景点世界坐标，采样方向从这里出发；
    /// - `sample`：2D 均匀样本 `[0, 1)²`；
    /// - `wavelengths`：波长上下文，只读取。
    ///
    /// 返回 `None` 表示采样失败（如方向不经过胶片、样本退化）。
    ///
    /// `pdf` 是**立体角测度**（从 `reference` 看相机的方向 `wi`）。
    ///
    /// 与 `Light::sample_le` 对称：
    ///
    /// - `Light::sample_le` 从光源出射，采样射向场景的方向；
    /// - `SampleWiCamera::sample_wi` 从场景点入射，采样射向相机的方向。
    fn sample_wi(
        &self,
        reference: Vec3,
        sample: Vec2,
        wavelengths: &S::WavelengthContext,
    ) -> Option<CameraWiSample<S>>;

    /// 给定方向，返回 `sample_wi` 采样该方向 `wi` 的 PDF（立体角测度）。
    ///
    /// 从**场景点** `reference` 看的方向立体角密度。必须与 `sample_wi`
    /// 使用同一分布；针孔相机返回 0。
    ///
    /// 契约：对任意 `s = sample_wi(reference, sample, wl)`，应有
    /// `pdf_wi(reference, s.wi) == s.pdf`（delta 相机除外）。
    fn pdf_wi(&self, reference: Vec3, wi: Vec3) -> f32;
}

// ═══════════════════════════════════════════════════════════════
// CameraRayDifferential：微分光线生成
// ═══════════════════════════════════════════════════════════════

/// 相机微分光线生成能力。
///
/// 微分光线用于纹理过滤：屏幕空间步进差打到表面后转成纹理坐标偏导，
/// 组成雅可比矩阵，决定 mipmap 层级和各向异性。
///
/// # 与 `Camera` 分离的理由
///
/// 微分光线的生成依赖相机模型——透视相机的微分有解析解，真实透镜
/// 相机的微分需要数值扰动。调用方不该耦合这些差异；需要微分的路径
/// 显式要求本 trait，不需要的路径（PT 的普通追踪、阴影射线）用基础
/// `Camera` 即可。
///
/// # 与 `Ray::spawn_ray` 的关系
///
/// `gen_ray_differential` 生成**初始**微分光线（从相机出发）；
/// 光线打到表面后的**传播**由 `RayDifferential::spawn_ray` 负责。
/// 两者是接力的：相机负责初始微分，`spawn_ray` 负责传播。
///
/// # Object-safety
///
/// `CameraRayDifferential` 是 object-safe 的。因为
/// `CameraRayDifferential: Camera`，trait upcasting 可以把
/// `Box<dyn CameraRayDifferential<S>>` 转成 `Box<dyn Camera<S>>`。
pub trait CameraRayDifferential<S: SpectralAlgebra>: Camera<S> {
    /// 生成 primary 微分光线（相机出射方向 `wo` 的微分）。
    ///
    /// 参数与 [`Camera::gen_ray`] 相同；返回的 `RayDifferential` 的
    /// `ray` 与 `gen_ray` 在相同参数下的结果一致（base ray），
    /// `differentials` 是屏幕空间 x/y 方向的初始步进差。
    ///
    /// 返回 `None` 表示采样失败。即使 `differentials` 为 `None` 也可以
    /// 返回（退化为普通光线），调用方应检查 `differentials.is_some()`
    /// 决定是否做纹理过滤。
    ///
    /// 一致性契约：对相同参数，`gen_ray_differential(...)` 的 `.ray`
    /// 应等于 `gen_ray(...)` 的 `.ray`（浮点误差允许范围）。跨实现测试
    /// 必须校验。
    fn gen_ray_differential(
        &self,
        pixel_sample: Vec2,
        lens_sample: Vec2,
        time: f32,
        wavelengths: &S::WavelengthContext,
    ) -> Option<RayDifferential>;
}

// ═══════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::{NoWavelengths, Rgb as SpectrumRgb};
    use glam::Vec3;

    // ─────────── 测试用相机 ───────────

    /// 极简相机：固定在原点，朝 -z 看，分辨率 100×100。
    /// 方向由 pixel_sample 唯一决定（针孔），pdf 为 delta（返回 0）。
    struct PinholeCamera {
        resolution: UVec2,
    }

    impl PinholeCamera {
        fn new() -> Self {
            PinholeCamera {
                resolution: UVec2::new(100, 100),
            }
        }
    }

    impl Camera<SpectrumRgb> for PinholeCamera {
        fn gen_ray(
            &self,
            pixel_sample: Vec2,
            _lens_sample: Vec2,
            time: f32,
            _wavelengths: &NoWavelengths,
        ) -> Option<CameraSample<SpectrumRgb>> {
            let x = (pixel_sample.x - 0.5) * 0.1;
            let y = (pixel_sample.y - 0.5) * 0.1;
            let dir = Vec3::new(x, y, -1.0).normalize();
            let ray = Ray::new(Vec3::ZERO, dir, time);
            Some(CameraSample {
                ray,
                film_position: pixel_sample * 100.0,
                pdf: 0.0, // 针孔相机：delta
                weight: SpectrumRgb::ONE,
            })
        }

        fn pdf_wo(&self, _ray: &Ray) -> f32 {
            0.0 // 针孔相机：delta
        }

        fn sample_time(&self, u: f32) -> f32 {
            u
        }

        fn film_size(&self) -> UVec2 {
            self.resolution
        }
    }

    /// 有限孔径相机：pdf 非零（模拟薄透镜）。
    struct ThinLensCamera {
        resolution: UVec2,
        lens_radius: f32,
    }

    impl ThinLensCamera {
        fn new() -> Self {
            ThinLensCamera {
                resolution: UVec2::new(100, 100),
                lens_radius: 0.1,
            }
        }
    }

    impl Camera<SpectrumRgb> for ThinLensCamera {
        fn gen_ray(
            &self,
            pixel_sample: Vec2,
            lens_sample: Vec2,
            time: f32,
            _wavelengths: &NoWavelengths,
        ) -> Option<CameraSample<SpectrumRgb>> {
            let x = (pixel_sample.x - 0.5) * 0.1;
            let y = (pixel_sample.y - 0.5) * 0.1;
            let dir = Vec3::new(x, y, -1.0).normalize();
            let origin = Vec3::new(
                (lens_sample.x - 0.5) * 2.0 * self.lens_radius,
                (lens_sample.y - 0.5) * 2.0 * self.lens_radius,
                0.0,
            );
            let ray = Ray::new(origin, dir, time);
            Some(CameraSample {
                ray,
                film_position: pixel_sample * 100.0,
                pdf: 1.0 / (std::f32::consts::PI * self.lens_radius * self.lens_radius),
                weight: SpectrumRgb::ONE,
            })
        }

        fn pdf_wo(&self, _ray: &Ray) -> f32 {
            1.0 / (std::f32::consts::PI * self.lens_radius * self.lens_radius)
        }

        fn sample_time(&self, u: f32) -> f32 {
            u
        }

        fn film_size(&self) -> UVec2 {
            self.resolution
        }
    }

    /// BDPT 用的相机：实现 SampleWiCamera。
    struct BDPTCamera {
        inner: PinholeCamera,
        camera_position: Vec3,
    }

    impl BDPTCamera {
        fn new() -> Self {
            BDPTCamera {
                inner: PinholeCamera::new(),
                camera_position: Vec3::ZERO,
            }
        }
    }

    impl Camera<SpectrumRgb> for BDPTCamera {
        fn gen_ray(
            &self,
            pixel_sample: Vec2,
            lens_sample: Vec2,
            time: f32,
            wl: &NoWavelengths,
        ) -> Option<CameraSample<SpectrumRgb>> {
            self.inner.gen_ray(pixel_sample, lens_sample, time, wl)
        }
        fn pdf_wo(&self, ray: &Ray) -> f32 {
            self.inner.pdf_wo(ray)
        }
        fn sample_time(&self, u: f32) -> f32 {
            self.inner.sample_time(u)
        }
        fn film_size(&self) -> UVec2 {
            self.inner.film_size()
        }
    }

    impl SampleWiCamera<SpectrumRgb> for BDPTCamera {
        fn sample_wi(
            &self,
            reference: Vec3,
            _sample: Vec2,
            _wavelengths: &NoWavelengths,
        ) -> Option<CameraWiSample<SpectrumRgb>> {
            let to_camera = self.camera_position - reference;
            let distance = to_camera.length();
            if distance < 1e-6 {
                return None;
            }
            let wi = to_camera / distance;
            Some(CameraWiSample {
                wi,
                weight: SpectrumRgb::ONE,
                pdf: 0.0, // 针孔相机：delta
                distance,
            })
        }

        fn pdf_wi(&self, _reference: Vec3, _wi: Vec3) -> f32 {
            0.0 // 针孔相机：delta
        }
    }

    /// 微分相机：实现 CameraRayDifferential。
    struct DiffCamera {
        inner: PinholeCamera,
    }

    impl DiffCamera {
        fn new() -> Self {
            DiffCamera {
                inner: PinholeCamera::new(),
            }
        }
    }

    impl Camera<SpectrumRgb> for DiffCamera {
        fn gen_ray(
            &self,
            pixel_sample: Vec2,
            lens_sample: Vec2,
            time: f32,
            wl: &NoWavelengths,
        ) -> Option<CameraSample<SpectrumRgb>> {
            self.inner.gen_ray(pixel_sample, lens_sample, time, wl)
        }
        fn pdf_wo(&self, ray: &Ray) -> f32 {
            self.inner.pdf_wo(ray)
        }
        fn sample_time(&self, u: f32) -> f32 {
            self.inner.sample_time(u)
        }
        fn film_size(&self) -> UVec2 {
            self.inner.film_size()
        }
    }

    impl CameraRayDifferential<SpectrumRgb> for DiffCamera {
        fn gen_ray_differential(
            &self,
            pixel_sample: Vec2,
            lens_sample: Vec2,
            time: f32,
            wavelengths: &NoWavelengths,
        ) -> Option<RayDifferential> {
            let base = self.gen_ray(pixel_sample, lens_sample, time, wavelengths)?;
            let dx = self.gen_ray(
                pixel_sample + Vec2::new(1.0 / 100.0, 0.0),
                lens_sample,
                time,
                wavelengths,
            )?;
            let dy = self.gen_ray(
                pixel_sample + Vec2::new(0.0, 1.0 / 100.0),
                lens_sample,
                time,
                wavelengths,
            )?;
            Some(RayDifferential::with_differentials(
                base.ray,
                crate::ray::RayDifferentials {
                    dx_origin: dx.ray.origin - base.ray.origin,
                    dx_dir: dx.ray.dir - base.ray.dir,
                    dy_origin: dy.ray.origin - base.ray.origin,
                    dy_dir: dy.ray.dir - base.ray.dir,
                },
            ))
        }
    }

    // ─────────── 基础 Camera 契约 ───────────

    #[test]
    fn pinhole_gen_ray_works() {
        let camera = PinholeCamera::new();
        let sample = camera
            .gen_ray(Vec2::splat(0.5), Vec2::ZERO, 0.0, &NoWavelengths)
            .expect("sample should succeed");
        // 中心像素应该朝 -z（相机出射方向 wo）
        assert!((sample.ray.dir.z + 1.0).abs() < 0.05);
        // 针孔相机 pdf 为 0（delta）
        assert_eq!(sample.pdf, 0.0);
        assert_eq!(sample.weight, SpectrumRgb::ONE);
    }

    #[test]
    fn pinhole_pdf_wo_zero() {
        let camera = PinholeCamera::new();
        let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), 0.0);
        assert_eq!(camera.pdf_wo(&ray), 0.0);
    }

    #[test]
    fn thin_lens_pdf_consistent() {
        let camera = ThinLensCamera::new();
        let sample = camera
            .gen_ray(Vec2::splat(0.5), Vec2::splat(0.5), 0.0, &NoWavelengths)
            .expect("sample should succeed");
        let pdf_wo = camera.pdf_wo(&sample.ray);
        assert!(
            (sample.pdf - pdf_wo).abs() < 1e-5,
            "pdf mismatch: sample.pdf={}, pdf_wo={}",
            sample.pdf,
            pdf_wo
        );
    }

    #[test]
    fn sample_time_passthrough() {
        let camera = PinholeCamera::new();
        assert!((camera.sample_time(0.0) - 0.0).abs() < 1e-6);
        assert!((camera.sample_time(0.5) - 0.5).abs() < 1e-6);
        assert!((camera.sample_time(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn film_size_query() {
        let camera = PinholeCamera::new();
        assert_eq!(camera.film_size(), UVec2::new(100, 100));
    }

    #[test]
    fn ray_time_filled() {
        let camera = PinholeCamera::new();
        let sample = camera
            .gen_ray(Vec2::splat(0.5), Vec2::ZERO, 0.7, &NoWavelengths)
            .expect("sample should succeed");
        assert!((sample.ray.time - 0.7).abs() < 1e-6);
    }

    // ─────────── Camera 是 object-safe ───────────

    #[test]
    fn camera_is_object_safe() {
        let _: Box<dyn Camera<SpectrumRgb>> = Box::new(PinholeCamera::new());
        let _: Box<dyn Camera<SpectrumRgb>> = Box::new(ThinLensCamera::new());
    }

    #[test]
    fn camera_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PinholeCamera>();
        assert_send_sync::<ThinLensCamera>();
    }

    // ─────────── SampleWiCamera 契约 ───────────

    #[test]
    fn sample_wi_direction_points_to_camera() {
        let camera = BDPTCamera::new();
        let reference = Vec3::new(0.0, 0.0, 5.0);
        let sample = camera
            .sample_wi(reference, Vec2::splat(0.5), &NoWavelengths)
            .expect("sample should succeed");
        // wi 从 reference 指向原点（相机位置），应为 -z 方向
        assert!(sample.wi.z < 0.0);
        assert!((sample.distance - 5.0).abs() < 1e-5);
    }

    #[test]
    fn sample_wi_degenerate_reference() {
        let camera = BDPTCamera::new();
        let sample = camera.sample_wi(Vec3::ZERO, Vec2::splat(0.5), &NoWavelengths);
        assert!(sample.is_none());
    }

    #[test]
    fn sample_wi_camera_is_object_safe() {
        let _: Box<dyn SampleWiCamera<SpectrumRgb>> = Box::new(BDPTCamera::new());
    }

    #[test]
    fn sample_wi_camera_upcasts_to_camera() {
        let s: Box<dyn SampleWiCamera<SpectrumRgb>> = Box::new(BDPTCamera::new());
        let _: Box<dyn Camera<SpectrumRgb>> = s;
    }

    // ─────────── CameraRayDifferential 契约 ───────────

    #[test]
    fn gen_ray_differential_matches_base() {
        let camera = DiffCamera::new();
        let pixel = Vec2::splat(0.5);
        let lens = Vec2::splat(0.5);
        let base = camera.gen_ray(pixel, lens, 0.0, &NoWavelengths).unwrap();
        let diff = camera
            .gen_ray_differential(pixel, lens, 0.0, &NoWavelengths)
            .unwrap();
        assert!((base.ray.origin - diff.ray.origin).length() < 1e-6);
        assert!((base.ray.dir - diff.ray.dir).length() < 1e-6);
        assert!(diff.differentials.is_some());
    }

    #[test]
    fn ray_differential_camera_is_object_safe() {
        let _: Box<dyn CameraRayDifferential<SpectrumRgb>> = Box::new(DiffCamera::new());
    }

    #[test]
    fn ray_differential_camera_upcasts_to_camera() {
        let d: Box<dyn CameraRayDifferential<SpectrumRgb>> = Box::new(DiffCamera::new());
        let _: Box<dyn Camera<SpectrumRgb>> = d;
    }

    // ─────────── 能力 trait 相互独立 ───────────

    #[test]
    fn pinhole_does_not_need_sample_wi() {
        fn takes_camera<C: Camera<SpectrumRgb>>(_c: &C) {}
        takes_camera(&PinholeCamera::new());
    }

    #[test]
    fn bdpt_camera_needs_both() {
        fn takes_bdpt<C: SampleWiCamera<SpectrumRgb>>(_c: &C) {}
        takes_bdpt(&BDPTCamera::new());
        fn takes_camera<C: Camera<SpectrumRgb>>(_c: &C) {}
        takes_camera(&BDPTCamera::new());
    }

    // ─────────── 方向符号语义 ───────────

    #[test]
    fn wo_and_wi_are_opposite() {
        // 相机出射 wo（CameraSample.ray.dir）与相机入射 wi
        // （CameraWiSample.wi）方向相反。
        let camera = BDPTCamera::new();
        let sample = camera.gen_ray(Vec2::splat(0.5), Vec2::ZERO, 0.0, &NoWavelengths).unwrap();
        let wo = sample.ray.dir;
        // 从相机原点看场景点（沿 wo 走 5 单位）
        let scene_point = sample.ray.origin + wo * 5.0;
        let wi_sample = camera
            .sample_wi(scene_point, Vec2::splat(0.5), &NoWavelengths)
            .unwrap();
        // wi 应该与 wo 大致反向（夹角接近 π）
        let dot = wo.normalize().dot(wi_sample.wi.normalize());
        assert!(dot < -0.99, "wo·wi = {dot}, expected ≈ -1");
    }
}
