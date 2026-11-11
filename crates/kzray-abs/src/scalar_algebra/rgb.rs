use glam::Vec3;
use super::ScalarAlgebra;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb(pub [f32; 3]);

impl Rgb {
    pub const ZERO: Rgb = Rgb([0.0; 3]);
    pub const ONE: Rgb = Rgb([1.0; 3]);

    pub const fn new(r: f32, g: f32, b: f32) -> Self { Rgb([r, g, b]) }
    pub const fn from_array(a: [f32; 3]) -> Self { Rgb(a) }
    pub const fn to_array(self) -> [f32; 3] { self.0 }
    pub fn from_vec3(v: Vec3) -> Self { Rgb([v.x, v.y, v.z]) }
    pub fn to_vec3(self) -> Vec3 { Vec3::from_array(self.0) }
}

impl ScalarAlgebra for Rgb {
    type Scalar = f32;

    #[inline]
    fn zero() -> Self { Rgb::ZERO }

    #[inline]
    fn one() -> Self { Rgb::ONE }

    #[inline]
    fn add(&self, o: &Self) -> Self {
        Rgb([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]])
    }

    #[inline]
    fn neg(&self) -> Self {
        Rgb([-self.0[0], -self.0[1], -self.0[2]])
    }

    #[inline]
    fn mul(&self, o: &Self) -> Self {
        Rgb([self.0[0] * o.0[0], self.0[1] * o.0[1], self.0[2] * o.0[2]])
    }

    #[inline]
    fn scale(&self, k: f32) -> Self {
        Rgb([self.0[0] * k, self.0[1] * k, self.0[2] * k])
    }

    #[inline]
    fn max_abs(&self) -> f32 {
        self.0[0].abs().max(self.0[1].abs()).max(self.0[2].abs())
    }
}
