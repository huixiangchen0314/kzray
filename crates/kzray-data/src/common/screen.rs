//! 像素坐标。屏幕空间 (x, y)。

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PixelCoord {
    pub x: u32,
    pub y: u32,
}

impl PixelCoord {
    pub const fn new(x: u32, y: u32) -> Self { PixelCoord { x, y } }
}
