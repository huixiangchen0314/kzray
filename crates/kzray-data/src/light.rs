//! 灯光。**封闭域**——灯光种类是物理枚举，用 enum 保证类型安全。
//!
//! 核心字段强类型（`position: [f32; 3]`、`intensity: Param`），扩展数据走
//! `meta`（`HasMeta` trait）——核心集封闭但允许携带动态拓展参数。

use serde::{Deserialize, Serialize};

use crate::common::meta::HasMeta;
use crate::common::{DataError, Meta, Param, Transform};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub struct LightFlags {
    pub camera_visible: bool,
    pub diffuse_visible: bool,
    pub glossy_visible: bool,
    pub cast_shadows: bool,
}

impl Default for LightFlags {
    fn default() -> Self {
        LightFlags {
            camera_visible: false,
            diffuse_visible: true,
            glossy_visible: true,
            cast_shadows: true,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct LightData {
    pub kind: LightType,
    pub flags: LightFlags,
    /// 动态拓展参数。
    pub meta: Meta,
}

impl HasMeta for LightData {
    fn meta(&self) -> &Meta { &self.meta }
    fn meta_mut(&mut self) -> &mut Meta { &mut self.meta }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum LightType {
    Point(PointLight),
    Spot(SpotLight),
    Area(AreaLight),
    Sun(SunLight),
    Background(BackgroundLight),
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct PointLight {
    pub position: [f32; 3],
    pub intensity: Param,
    pub color: Param,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct SpotLight {
    pub position: [f32; 3],
    /// 光线方向（单位向量，世界空间）。
    pub direction: [f32; 3],
    /// 光锥半角（弧度）。
    pub cone_angle: Param,
    /// 软边比例 `[0, 1]`。
    pub penumbra: Param,
    pub intensity: Param,
    pub color: Param,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct AreaLight {
    /// 灯光世界变换。局部 +Z 为发光面法线。
    pub transform: Transform,
    pub shape: AreaShape,
    pub intensity: Param,
    pub color: Param,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum AreaShape {
    Rectangle { size_u: Param, size_v: Param },
    Disk { radius: Param },
    Sphere { radius: Param },
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct SunLight {
    /// 光线方向（从太阳射向场景，单位向量）。
    pub direction: [f32; 3],
    /// 太阳角直径（弧度）。
    pub angle: Param,
    pub intensity: Param,
    pub color: Param,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct BackgroundLight {
    /// 环境颜色（常量 `Vec3` 或指向环境贴图的 Field）。
    pub color: Param,
    pub intensity: Param,
}

impl LightData {
    pub fn new(kind: LightType) -> Self {
        LightData { kind, flags: LightFlags::default(), meta: Meta::new() }
    }

    pub fn point(position: [f32; 3], intensity: Param, color: Param) -> Self {
        LightData::new(LightType::Point(PointLight { position, intensity, color }))
    }

    pub fn validate(&self) -> Result<(), DataError> {
        match &self.kind {
            LightType::Point(l) => {
                l.intensity.validate("light.point.intensity")?;
                l.color.validate("light.point.color")?;
            }
            LightType::Spot(l) => {
                l.cone_angle.validate("light.spot.cone_angle")?;
                l.penumbra.validate("light.spot.penumbra")?;
                l.intensity.validate("light.spot.intensity")?;
                l.color.validate("light.spot.color")?;
            }
            LightType::Area(l) => {
                match &l.shape {
                    AreaShape::Rectangle { size_u, size_v } => {
                        size_u.validate("light.area.size_u")?;
                        size_v.validate("light.area.size_v")?;
                    }
                    AreaShape::Disk { radius } => radius.validate("light.area.radius")?,
                    AreaShape::Sphere { radius } => radius.validate("light.area.radius")?,
                }
                l.intensity.validate("light.area.intensity")?;
                l.color.validate("light.area.color")?;
            }
            LightType::Sun(l) => {
                l.angle.validate("light.sun.angle")?;
                l.intensity.validate("light.sun.intensity")?;
                l.color.validate("light.sun.color")?;
            }
            LightType::Background(l) => {
                l.color.validate("light.background.color")?;
                l.intensity.validate("light.background.intensity")?;
            }
        }
        self.meta.validate("light.meta")?;
        Ok(())
    }
}
