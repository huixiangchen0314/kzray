use glam::Vec2;
use crate::ray::Ray;

pub struct CameraRay {
    pub ray: Ray,
    pub pdf: f32,
    pub wavelength: Option<f32>,
}

pub trait Camera: Send + Sync {
    fn generate_ray(&self, pixel: Vec2, u: &mut [f32]) -> CameraRay;
    fn film_bounds(&self) -> (u32, u32);
}
