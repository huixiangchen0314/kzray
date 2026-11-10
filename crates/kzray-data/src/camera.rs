//! 相机。**封闭域**——投影模型有限，用 enum 保证类型安全。
//! 扩展数据走 `meta`（`HasMeta`）。

use serde::{Deserialize, Serialize};

use crate::common::meta::HasMeta;
use crate::common::{DataError, Meta, Param, Transform};

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct CameraData {
    /// camera-to-world。局部空间：+Z 前方，+Y 上，+X 右。
    pub transform: Transform,
    pub projection: CameraProjection,
    pub shutter: Option<ShutterInterval>,
    /// 动态拓展参数。
    pub meta: Meta,
}

impl HasMeta for CameraData {
    fn meta(&self) -> &Meta { &self.meta }
    fn meta_mut(&mut self) -> &mut Meta { &mut self.meta }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct ShutterInterval {
    pub open: f32,
    pub close: f32,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum CameraProjection {
    Perspective(PerspectiveCamera),
    Orthographic(OrthographicCamera),
    Panorama(PanoramaCamera),
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct PerspectiveCamera {
    /// 垂直视场角（弧度）。
    pub fov_y: Param,
    pub near: f32,
    pub far: f32,
    /// 光圈直径（世界单位）。0 = 无景深。
    pub aperture: Param,
    pub focus_distance: Param,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct OrthographicCamera {
    /// 视口高度（世界单位）。
    pub ortho_scale: Param,
    pub near: f32,
    pub far: f32,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct PanoramaCamera {
    pub panorama_type: PanoramaType,
    pub fov_y: Param,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum PanoramaType {
    Equirectangular,
    FisheyeEquidistant,
    FisheyeEquidistantRectangular,
}

impl CameraData {
    pub fn new(transform: Transform, projection: CameraProjection) -> Self {
        CameraData { transform, projection, shutter: None, meta: Meta::new() }
    }

    pub fn validate(&self) -> Result<(), DataError> {
        if !self.transform.validate() {
            return Err(DataError::InvalidParam {
                path: "camera.transform".into(),
                reason: "变换矩阵含非有限值".into(),
            });
        }
        match &self.projection {
            CameraProjection::Perspective(c) => {
                c.fov_y.validate("camera.perspective.fov_y")?;
                c.aperture.validate("camera.perspective.aperture")?;
                c.focus_distance.validate("camera.perspective.focus_distance")?;
            }
            CameraProjection::Orthographic(c) => {
                c.ortho_scale.validate("camera.orthographic.ortho_scale")?;
            }
            CameraProjection::Panorama(c) => {
                c.fov_y.validate("camera.panorama.fov_y")?;
            }
        }
        self.meta.validate("camera.meta")?;
        Ok(())
    }
}
