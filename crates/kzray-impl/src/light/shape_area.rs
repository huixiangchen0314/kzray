//! kzray-impl/src/light/shape_area.rs

use glam::{Affine3A, Vec2, Vec3, Vec3A};

use kzray_abs::light::{
    LiGeometry, LiSample, Light, LightSampleContext, LightType,
};
use kzray_abs::spectrum::{ScalarAlgebra, SpectralAlgebra};

use crate::geometry::{AnyShapeA, ShapeA, SphereA, TriangleA};

/// 解析形状面积光。自包含几何——采样时零查表、零 `Library` 依赖。
///
/// # 世界空间几何
///
/// 构造时把局部几何经 `transform` 变换成**世界空间几何**存储。面积
/// 采样在世界空间做——避免变换对采样密度的影响，`pdf_area` 直接是
/// `1 / 世界面积`。这是与"存局部几何 + 每次采样时变换"的关键区别：
/// 后者在非均匀缩放下采样密度不均匀，pdf 需计入雅可比，复杂且易错。
///
/// # 支持的形状
///
/// - `Triangle`：任意仿射变换（三角形在仿射下还是三角形）；
/// - `Sphere`：仅刚体 + 均匀缩放（否则变椭球，采样困难）。构造时
///   校验，非均匀缩放 panic；
/// - `Plane`：不支持（无面积）。
///
/// # 未来
///
/// mesh 面积光不在本类型——届时新增 `MeshAreaLight`，从
/// [`LightSampleContext::library`] 取 mesh 顶点做世界空间采样。
pub struct ShapeAreaLight<S: SpectralAlgebra> {
    /// 世界空间几何。`Triangle` 或 `Sphere`。
    world_shape: AnyShapeA,
    /// 辐射亮度。
    radiance: S,
    /// 双面自发光。`false`（默认）仅法线正向发光。
    two_sided: bool,
}

impl<S: SpectralAlgebra> ShapeAreaLight<S> {
    /// 从局部几何 + 变换构造。几何被变换到世界空间。
    ///
    /// # Panics
    ///
    /// - 几何是 `Plane`：无面积，不能做面积光；
    /// - 几何是 `Sphere` 且 `transform` 含非均匀缩放。
    pub fn new(shape: AnyShapeA, transform: Affine3A, radiance: S) -> Self {
        let world_shape = transform_to_world(shape, &transform);
        Self { world_shape, radiance, two_sided: false }
    }

    /// 双面自发光（两面都发光）。默认单面（仅法线正向）。
    pub fn two_sided(mut self, two_sided: bool) -> Self {
        self.two_sided = two_sided;
        self
    }

    /// 世界空间几何。
    pub fn shape(&self) -> &AnyShapeA {
        &self.world_shape
    }
}

impl<S: SpectralAlgebra> Light<S> for ShapeAreaLight<S> {
    #[inline]
    fn light_type(&self) -> LightType {
        LightType::Area
    }

    fn sample_li(
        &self,
        ctx: &LightSampleContext<'_, S>,
        sample: Vec2,
        _wavelengths: &S::WavelengthContext,
    ) -> Option<LiSample<S>> {
        // 世界空间面积采样。忽略 ctx.library——几何自包含。
        let area_sample = self.world_shape.sample_area(sample)?;
        let world_pos = Vec3::from(area_sample.position);
        let world_normal = Vec3::from(area_sample.normal);

        let to_light = world_pos - ctx.position;
        let distance = to_light.length();
        if distance <= 1e-6 {
            return None;
        }
        let wi = to_light / distance;

        // cos_light：光源法线与指向着色点的方向的点积（取正）。
        // 双面取绝对值。
        let cos_light = world_normal.dot(-wi);
        let cos_light = if self.two_sided {
            cos_light.abs()
        } else {
            cos_light
        };
        if cos_light <= 1e-6 {
            return None; // 背面或掠射
        }

        // 面积 pdf → 立体角 pdf。
        let pdf_solid = area_sample.pdf_area * distance * distance / cos_light;

        Some(LiSample {
            geometry: LiGeometry::Surface {
                position: world_pos,
                distance,
            },
            wi,
            incident: self.radiance,
            pdf: pdf_solid,
        })
    }

    fn pdf_li(&self, _ctx: &LightSampleContext<'_, S>, _wi: Vec3) -> f32 {
        // 反算：光线与几何求交得交点后算 pdf。MIS 阶段实现。
        // 当前 NEE 不调用——NEE 用 sample_li 返回的 pdf。
        0.0
    }

    fn phi(&self, _wavelengths: &S::WavelengthContext) -> S {
        // 总辐射功率 = 辐射亮度 × 面积 × π（半球余弦积分）。
        // 双面自发光时两面都发光，功率翻倍。
        let area = self.world_shape.surface_area().unwrap_or(0.0);
        let two_sided_factor = if self.two_sided { 2.0 } else { 1.0 };
        let pi_area = std::f32::consts::PI * area * two_sided_factor;
        self.radiance.scale(S::Scalar::from(pi_area))
    }
}

/// 把局部几何变换到世界空间。
fn transform_to_world(shape: AnyShapeA, transform: &Affine3A) -> AnyShapeA {
    match shape {
        AnyShapeA::Triangle(tri) => {
            let v0 = transform.transform_point3a(tri.v0);
            let v1 = transform.transform_point3a(tri.v1);
            let v2 = transform.transform_point3a(tri.v2);
            AnyShapeA::Triangle(TriangleA::new(v0, v1, v2))
        }
        AnyShapeA::Sphere(s) => {
            // 校验均匀缩放：三个轴向量长度相等。
            let m = &transform.matrix3;
            let sx = m.x_axis.length();
            let sy = m.y_axis.length();
            let sz = m.z_axis.length();
            let eps = 1e-4;
            assert!(
                (sx - sy).abs() < eps * sx.max(sy)
                    && (sy - sz).abs() < eps * sy.max(sz),
                "ShapeAreaLight with sphere requires uniform scale \
                 (got axes |x|={sx}, |y|={sy}, |z|={sz})"
            );
            let center = transform.transform_point3a(s.center);
            AnyShapeA::Sphere(SphereA::new(center, s.radius * sx))
        }
        AnyShapeA::Plane(_) => {
            panic!("ShapeAreaLight cannot use Plane (infinite surface has no area)")
        }
    }
}
