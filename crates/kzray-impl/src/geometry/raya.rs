use glam::Vec3A;
use kzray_abs::Ray;

/// `Ray` 的 SIMD 内部表示，**仅用于求交热点路径**。
///
/// 语义与 [`Ray`] 完全一致（`dir` 不归一化、`t` 为仿射参数、`time`
/// 继承规则相同），唯一的区别是 `origin`/`dir` 用 `Vec3A` 存储。
/// 不要在此类型上定义新语义——它只是一个表示层的镜像。
///
/// 不实现 `Serialize`/`Deserialize`：纯内部加速类型，序列化一律用 `Ray`。
#[derive(Clone, Copy, Debug)]
pub struct RayA {
    pub origin: Vec3A,
    pub dir: Vec3A,
    /// 与 `Ray::time` 同义，归一化到 `[0, 1]` 快门区间。
    pub time: f32,
}

impl RayA {
    /// 注意：**不是 `const fn`**。`Vec3A` 的构造在 SIMD 后端下不保证
    /// const（SSE2 的 `_mm_set_ps` 不是 const intrinsic），所以不要照抄
    /// `Ray::new` 的 `const`。需要 const 场景请先构造 `Ray` 再 `.into()`。
    #[inline]
    pub fn new(origin: Vec3A, dir: Vec3A, time: f32) -> Self {
        Self { origin, dir, time }
    }

    /// 与 `Ray::spawn` 语义相同：`time` 自动继承。
    #[inline]
    pub fn spawn(&self, origin: Vec3A, dir: Vec3A) -> Self {
        Self { origin, dir, time: self.time }
    }

    /// 参数化位置。返回 `Vec3A`——热点里不要在这里转回 `Vec3`。
    #[inline]
    pub fn at(&self, t: f32) -> Vec3A {
        self.origin + self.dir * t
    }
}

impl From<Ray> for RayA {
    #[inline]
    fn from(r: Ray) -> Self {
        Self { origin: r.origin.into(), dir: r.dir.into(), time: r.time }
    }
}

impl From<RayA> for Ray {
    #[inline]
    fn from(r: RayA) -> Self {
        Self { origin: r.origin.into(), dir: r.dir.into(), time: r.time }
    }
}
