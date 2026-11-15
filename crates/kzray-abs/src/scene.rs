//! 场景只读视图。
//!
//! # 全泛型化（`Material` 除外）
//!
//! `Scene<S, LS, L, IL>` 携带 4 个泛型参数：
//!
//! - `S`：光谱类型；
//! - `LS: LightSampler`：光源采样器；
//! - `L: Light<S>`：光源；
//! - `IL: InfiniteLight<S>`：无限远光源。
//!
//! `material` 返回 `&Material<S>`——`Material` 是契约层的具体 enum 类型，
//! 不是泛型参数。材质种类封闭（传输方程结构），由 enum 表达；光源种类
//! 开放（场景内容），由泛型 `L` 表达。
//!
//! 所有查询方法返回具体类型引用，零虚分派。`Scene` 本身 object-safe
//! （方法都不是泛型），但通过泛型 `SC: Scene<...>` 传播。
//!
//! # `Library<S>` 的定位
//!
//! [`Scene::library`] 返回 `&Library<S>`——统一资源容器，持有网格、
//! 各通道纹理（alpha / albedo / roughness 等）。求交层通过它查 alpha
//! 纹理做 cutout；着色层通过它查 albedo / roughness 等纹理。`Library`
//! 定义在 `kzray-abs`，具体纹理实现由 `kzray-impl` 通过
//! `Box<dyn Texture<C>>` 注入。

use std::ops::Range;

use kzray_data::common::LightId;
use kzray_data::geometry::Aabb;

use crate::library::Library;
use crate::light::{InfiniteLight, Light, LightSampler};
use crate::material::Material;
use crate::ray::{Ray, SurfaceHit};
use crate::spectrum::SpectralAlgebra;

pub trait Scene<S, LS, L, IL>: Send + Sync
where
    S: SpectralAlgebra,
    LS: LightSampler,
    L: Light<S>,
    IL: InfiniteLight<S>,
{
    /// 场景整体 AABB。
    ///
    /// 无限图元（无限平面）存在时返回 [`Aabb::empty`]，表示"场景没有
    /// 有限包围盒"。上层用 `is_empty()` 判断。
    fn bounds(&self) -> Aabb;

    /// 最近命中。`t_range` 限定搜索区间。
    fn intersect(&self, ray: &Ray, t_range: Range<f32>) -> Option<SurfaceHit>;

    /// 遮挡查询：`t_range` 内任意可接受命中存在即 `true`。
    ///
    /// 与 [`Self::intersect`] 的区别：只回答"有没有"，不求最近、不构造
    /// `SurfaceHit`。阴影射线、AO 射线专用。alpha 透明命中被自动跳过。
    fn intersect_p(&self, ray: &Ray, t_range: Range<f32>) -> bool;

    /// 查询运行时材质。返回 `&Material<S>`（具体 enum）。
    ///
    /// `hit.instance_id` 索引到材质表。材质与实例一一对应（每个实例
    /// 一个材质），与几何形状无关。
    fn material(&self, hit: &SurfaceHit) -> Option<&Material<S>>;

    /// 查光源。
    fn light(&self, id: LightId) -> Option<&L>;

    /// 光源采样器。无光源时为 `None`。
    fn light_sampler(&self) -> Option<&LS>;

    /// 无限远光源（环境光）。无环境光时为 `None`。
    fn infinite_light(&self) -> Option<&IL>;

    /// 统一资源容器。
    ///
    /// 着色层通过它查纹理（albedo / roughness / normal 等），求交层
    /// 内部已用它的 alpha 纹理做 cutout。`Scene` 暴露它以让 shading
    /// 流程可以在 `scene.intersect(...)` 之后继续 `scene.library()`，
    /// 两次 `&self` 共享借用，无冲突。
    fn library(&self) -> &Library<S>;
}
