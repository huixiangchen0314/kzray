use std::collections::BTreeMap;
use glam::Vec3;
use kzray_data::volume::VolumeData;
use crate::error::VolumeError;
use crate::ray::Ray;
use crate::runtime::Runtime;
use crate::sampler::Sampler;
use crate::scalar_algebra::ScalarAlgebra;

#[derive(Clone, Copy, Debug)]
pub struct VolumeInteraction {
    pub t: f32,
    pub position: Vec3,
}

pub trait Volume<S: ScalarAlgebra>: Send + Sync {
    fn intersect(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<(f32, f32)>;
    fn sample_interaction(
        &self,
        ray: &Ray,
        t_enter: f32,
        t_exit: f32,
        sampler: &mut dyn Sampler,
    ) -> Option<VolumeInteraction>;
    fn transmittance(&self, ray: &Ray, t_enter: f32, t_exit: f32, sampler: &mut dyn Sampler) -> S;
    fn density(&self, p: Vec3) -> f32;
}

pub trait PhaseFunction<S: ScalarAlgebra>: Send + Sync {
    fn eval(&self, wo: Vec3, wi: Vec3) -> S;
    fn sample(&self, wo: Vec3, u: &mut [f32]) -> Option<(Vec3, f32)>;
    fn pdf(&self, wo: Vec3, wi: Vec3) -> f32;
}

pub type VolumeBuilderFn<S> = Box<
    dyn Fn(&VolumeData, &Runtime<S>) -> Result<Box<dyn Volume<S>>, VolumeError>
        + Send + Sync,
>;

pub struct VolumeRegistry<S: ScalarAlgebra> {
    builders: BTreeMap<String, VolumeBuilderFn<S>>,
}

impl<S: ScalarAlgebra> VolumeRegistry<S> {
    pub fn new() -> Self { VolumeRegistry { builders: BTreeMap::new() } }
    pub fn register(&mut self, kind: &str, builder: VolumeBuilderFn<S>) {
        self.builders.insert(kind.to_string(), builder);
    }
    pub fn build(&self, data: &VolumeData, runtime: &Runtime<S>) -> Result<Box<dyn Volume<S>>, VolumeError> {
        let b = self.builders.get(&data.kind)
            .ok_or_else(|| VolumeError::UnknownKind(data.kind.clone()))?;
        b(data, runtime)
    }
}

impl<S: ScalarAlgebra> Default for VolumeRegistry<S> { fn default() -> Self { Self::new() } }
