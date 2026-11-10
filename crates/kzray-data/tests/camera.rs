//! 对应 `src/camera.rs`。

use kzray_data::camera::{
    CameraData, CameraProjection, OrthographicCamera, PanoramaCamera, PanoramaType,
    PerspectiveCamera, ShutterInterval,
};
use kzray_data::common::{HasMeta, Meta, Param, Transform};

fn roundtrip<T>(v: &T) -> Vec<u8>
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let bytes = bincode::serialize(v).expect("serialize");
    let back: T = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(&back, v);
    assert_eq!(bytes, bincode::serialize(&back).unwrap());
    bytes
}

fn f32_param(v: f32) -> Param { Param::from_f32(v) }

#[test]
fn perspective_camera() {
    let cam = CameraData::new(
        Transform::IDENTITY,
        CameraProjection::Perspective(PerspectiveCamera {
            fov_y: f32_param(0.7),
            near: 0.01,
            far: 1000.0,
            aperture: f32_param(0.0),
            focus_distance: f32_param(5.0),
        }),
    );
    assert!(cam.validate().is_ok());
    roundtrip(&cam);
}

#[test]
fn orthographic_camera() {
    let cam = CameraData::new(
        Transform::IDENTITY,
        CameraProjection::Orthographic(OrthographicCamera {
            ortho_scale: f32_param(2.0),
            near: 0.01,
            far: 1000.0,
        }),
    );
    assert!(cam.validate().is_ok());
    roundtrip(&cam);
}

#[test]
fn panorama_equirectangular() {
    let cam = CameraData::new(
        Transform::IDENTITY,
        CameraProjection::Panorama(PanoramaCamera {
            panorama_type: PanoramaType::Equirectangular,
            fov_y: f32_param(std::f32::consts::PI),
        }),
    );
    assert!(cam.validate().is_ok());
    roundtrip(&cam);
}

#[test]
fn shutter_roundtrip() {
    roundtrip(&ShutterInterval { open: 0.0, close: 0.5 });
}

#[test]
fn camera_with_motion_blur() {
    let mut cam = CameraData::new(
        Transform::IDENTITY,
        CameraProjection::Perspective(PerspectiveCamera {
            fov_y: f32_param(0.7),
            near: 0.01,
            far: 1000.0,
            aperture: f32_param(0.05),
            focus_distance: f32_param(3.0),
        }),
    );
    cam.shutter = Some(ShutterInterval { open: 0.0, close: 0.5 });
    assert!(cam.validate().is_ok());
    roundtrip(&cam);
}

#[test]
fn validate_rejects_bad_transform() {
    let cam = CameraData::new(
        Transform([
            [f32::NAN, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        CameraProjection::Perspective(PerspectiveCamera {
            fov_y: f32_param(0.7),
            near: 0.01,
            far: 1000.0,
            aperture: f32_param(0.0),
            focus_distance: f32_param(5.0),
        }),
    );
    assert!(cam.validate().is_err());
}

#[test]
fn camera_carries_meta() {
    let mut meta = Meta::new();
    meta.insert("shot_id", Param::from_i64(101));
    let cam = CameraData::new(
        Transform::IDENTITY,
        CameraProjection::Perspective(PerspectiveCamera {
            fov_y: f32_param(0.7),
            near: 0.01,
            far: 1000.0,
            aperture: f32_param(0.0),
            focus_distance: f32_param(5.0),
        }),
    )
    .with_meta(meta);

    assert_eq!(cam.meta().len(), 1);
    assert!(cam.validate().is_ok());
    roundtrip(&cam);
}
