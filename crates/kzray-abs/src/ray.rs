use glam::{Vec2, Vec3};
use kzray_data::{LightId, common::InstanceId};
use kzray_math::onb::Onb;

// ═══════════════════════════════════════════════════════════════
// Ray
// ═══════════════════════════════════════════════════════════════

/// 光线。`origin + t * dir`，`t ∈ [t_min, t_max]` 由求交参数约束，不存于结构体。
///
/// # 携带的信息
///
/// - 几何：`origin` / `dir`；
/// - 时间：`time`，用于运动模糊的时间插值。
///
/// # 不携带的信息
///
/// - 介质（`medium`）：随光线传播切换，放求交/积分上下文，不在这里。
///   放 `Ray` 意味着每次切换要构造新 `Ray`（破坏 `Copy`）或让 `Ray`
///   可变；体积路径本来就需要维护介质栈，放上下文是自然的。pbrt 把
///   `medium` 放 `Ray` 是因为 C++ 的 `Ray` 可变；Rust 里 `Ray` 是
///   `Copy` 的，代价更高。
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,

    /// 方向向量。
    ///
    /// 不要求归一化；`t` 是仿射参数，实际距离见 `SurfaceHit::distance`。
    /// 光线变换不强制归一化，以保留 `t` 的仿射可比性。
    pub dir: Vec3,

    /// 光线时刻，归一化到快门区间 `[0, 1]`。
    ///
    /// - `0.0`：快门开启；
    /// - `1.0`：快门关闭。
    ///
    /// 相机生成 primary ray 时由相机样本决定；次级光线（反射/透射/
    /// 阴影）通过 [`Ray::spawn`] 派生，`time` 自动继承主光线。同一路径
    /// 上的所有光线共享同一时刻，这是运动模糊物理正确性的前提。
    ///
    /// 无运动模糊的场景所有光线 `time = 0.0`，求交时不做时间插值。
    pub time: f32,
}

impl Ray {
    /// 构造光线。
    ///
    /// `time` 归一化到 `[0, 1]`，对应快门区间。相机生成 primary ray 时
    /// 传相机样本的时间；固定场景传 `0.0`。
    pub const fn new(origin: Vec3, dir: Vec3, time: f32) -> Self {
        Ray { origin, dir, time }
    }

    /// 从当前光线派生一条新光线。
    ///
    /// `time` 自动继承——次级光线（反射/透射/阴影/偏移后的 primary ray）
    /// 必须与主光线同一时刻。这是运动模糊物理正确性的前提，也是用
    /// `spawn` 而非 `Ray::new` 的原因。
    ///
    /// 典型用法：
    ///
    /// ```text
    /// // 反射光线
    /// let reflected = ray.spawn(hit.offset_origin(1.0, eps), reflect_dir);
    /// // 阴影射线
    /// let shadow = ray.spawn(hit.offset_origin(1.0, eps), to_light);
    /// ```
    #[inline]
    pub const fn spawn(&self, origin: Vec3, dir: Vec3) -> Self {
        Ray { origin, dir, time: self.time }
    }

    /// 参数化位置。
    #[inline]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// 各类求交结果的统一枚举。
///
/// 仅在求交上下文不明确、或不关心性能时使用（场景遍历、调试、
/// 统计、跨类型统一处理）。热路径应直接使用具体类型，避免枚举
/// 分支和取最大变体体积。
#[derive(Clone, Copy, Debug)]
pub enum Hit {
    Surface(SurfaceHit),
    SurfaceWithDifferentials(SurfaceHitWithDifferentials),
    /// 面积光表面命中。
    AreaLight(AreaLightHit),
    // TODO: Curve(CurveHit), Volume(VolumeEvent), ...
}

impl Hit {
    /// 如果是表面命中（`Surface` / `SurfaceWithDifferentials` / `AreaLight`），
    /// 返回 `&SurfaceHit`；未来 `Curve` / `Volume` 变体返回 `None`。
    ///
    /// 调用方需要区分"是表面"和"不是表面"：直接 `match` 变体拿具体信息，
    /// 或 `surface().expect("此处只处理表面命中")` 在明确上下文中取值。
    #[inline]
    pub fn surface(&self) -> Option<&SurfaceHit> {
        match self {
            Self::Surface(h) => Some(h),
            Self::SurfaceWithDifferentials(h) => Some(&h.hit),
            Self::AreaLight(h) => Some(&h.hit),
            // TODO：
            // Self::Curve(_) => None,
            // Self::Volume(_) => None,
        }
    }

    /// 命中的仿射参数 `t`。
    ///
    /// 所有变体都有 `t`——它是光线的参数，与命中类型无关。`Curve` /
    /// `Volume` 落地时加对应分支，签名不变。
    #[inline]
    pub fn t(&self) -> f32 {
        match self {
            Self::Surface(h) => h.t,
            Self::SurfaceWithDifferentials(h) => h.hit.t,
            Self::AreaLight(h) => h.hit.t,
            // TODO：
            // Self::Curve(h) => h.t,
            // Self::Volume(h) => h.t,
        }
    }

    /// 如果是面积光命中，返回 `LightId`。
    #[inline]
    pub fn light_id(&self) -> Option<LightId> {
        match self {
            Self::Surface(_) | Self::SurfaceWithDifferentials(_) => None,
            Self::AreaLight(h) => Some(h.light_id),
            // TODO：Curve / Volume 默认 None
        }
    }

    /// 屏幕空间微分。`Surface` / `AreaLight` 变体无微分，返回 `None`；
    /// `SurfaceWithDifferentials` 返回 `Some`。
    ///
    /// 面积光若未来带 emission texture，需要微分做 MIP——届时加
    /// `AreaLightWithDifferentials` 变体，本方法补对应分支。
    #[inline]
    pub fn differentials(&self) -> Option<&SurfaceDifferentials> {
        match self {
            Self::Surface(_) | Self::AreaLight(_) => None,
            Self::SurfaceWithDifferentials(h) => Some(&h.differentials),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// SurfaceHit
// ═══════════════════════════════════════════════════════════════

/// 光线与几何体的表面击中结果。
///
/// 由加速结构的求交产生，携带着色所需的最小信息。
/// 材质、BSDF 等不在其中——通过 `instance_id` / `prim_id` 在 `Scene` 中查得。
/// 求交返回统一的 `SurfaceHit`，调用方拆字段使用；可选/昂贵信息（微分、AOV）
/// 通过不同的求交入口或旁路输出控制。
///
/// 不携带 `time`：`time` 是光线属性，从对应的 `Ray` 读取。求交不改变
/// 时刻。
#[derive(Clone, Copy, Debug)]
pub struct SurfaceHit {
    /// 光线参数。击中点 = `origin + t * dir`。
    pub t: f32,

    /// 从光线起点到命中点的实际距离，等于 `t * |dir|`。
    ///
    /// 当 `dir` 不归一化时，`t` 不是距离；偏移、微分距离缩放、
    /// epsilon 缩放都用这个。求交时计算一次，避免热路径重复开方。
    pub distance: f32,

    /// 击中点世界坐标。
    pub position: Vec3,

    /// 着色法线（可能经法线贴图或插值扰动）。
    pub shading_normal: Vec3,

    /// 几何法线（面法线，用于偏移和可见性判断）。
    pub geometric_normal: Vec3,

    /// 表面 UV。用于计算采样，纹理空间雅可比。
    pub uv: Vec2,

    /// 实例 ID。用于在 `Scene` 中查找变换、材质槽等。
    pub instance_id: InstanceId,

    /// 实例内的图元索引。材质槽通过 `Scene::material_slot(instance_id, prim_id)`
    /// 查询；图元按材质槽排序，该查询为 O(1)。
    pub prim_id: u32,
}

/// 带微分的表面命中，旁路输出。
///
/// 由 `intersect_with_differentials` 返回；不需要微分的路径走
/// `intersect`，不承担这个体积。
#[derive(Clone, Copy, Debug)]
pub struct SurfaceHitWithDifferentials {
    pub hit: SurfaceHit,

    /// 表面微分信息。
    ///
    /// 由求交/表面交互模块在传播微分时填充；需要 AOV 或纹理过滤的
    /// 路径应在求交时传入 `RayDifferential` 并填充此字段。
    /// - 几何法线做偏移和可见性，用着色法线做 BSDF 相关传播
    /// - 传播后的微分要 clamp/scale，避免数值爆炸或退化
    /// - 新光线起点偏移后，origin 微分仍归零，因为所有微分光线从同一偏移点出发
    pub differentials: SurfaceDifferentials,

    /// 表面参数化对 u 的偏导（世界坐标）。
    ///
    /// 由求交填充（三角形从顶点插值，解析形状从参数化算）。与
    /// [`SurfaceDifferentials::dpdx`] / [`SurfaceDifferentials::dpdy`] 不同：
    /// 后两者是**屏幕空间**微分，这里是**表面参数化（uv）**的偏导。
    ///
    /// 用途：构造精确切线空间（各向异性 BRDF、法线贴图）。各向同性
    /// BSDF 用 [`SurfaceHit::frame`] 的 `Onb::from_w(shading_normal)` 即可，
    /// 不需要这两个字段。
    ///
    /// 不要求与 `shading_normal` 正交（法线贴图扰动后通常不正交）；
    /// 实现层用 Gram-Schmidt 正交化，退化时回退到 `Onb::from_w`。
    pub dpdu: Vec3,

    /// 表面参数化对 v 的偏导（世界坐标）。
    ///
    /// 与 [`Self::dpdu`] 配对；用途与退化处理见该字段文档。
    pub dpdv: Vec3,
}

impl SurfaceHit {
    /// 从击中点沿几何法线偏移，避免自交。
    ///
    /// `sign` 为 `1.0` 时沿法线正向偏移，`-1.0` 时沿反向偏移。
    /// 偏移量基于实际距离的尺度，避免远处偏移不足、近处偏移过度。
    #[inline]
    pub fn offset_origin(&self, sign: f32, epsilon: f32) -> Vec3 {
        let scale = self.distance.max(1.0);
        self.position + self.geometric_normal * (sign * epsilon * scale)
    }

    /// 出射方向 `wo`：从着色点指向相机/上一跳。
    ///
    /// 等于 `-ray.dir` 归一化。`Ray::dir` 不要求归一化，此处显式归一化
    /// 后再取反，保证返回单位向量。
    ///
    /// 与 `BSDF::new(wo_world, ...)` 的 `wo_world` 一致。
    ///
    /// 求交成功的 `SurfaceHit` 对应的 `ray.dir` 一定非零（求交本身要求
    /// 非零方向），`normalize()` 不会产生 NaN。
    #[inline]
    pub fn wo(&self, ray: &Ray) -> Vec3 {
        -ray.dir.normalize()
    }

    /// 着色局部坐标系。
    ///
    /// 用 `shading_normal` 构造正交基，与 `BSDF::new` 的 `normal_world`
    /// 一致。各向同性 BSDF 使用此坐标系即可。
    ///
    /// 各向异性 BSDF 或法线贴图需要精确切线（`dpdu` / `dpdv`）时，应从
    /// [`SurfaceHitWithDifferentials`] 获取，本方法不承担该信息。
    #[inline]
    pub fn frame(&self) -> Onb {
        Onb::from_w(self.shading_normal)
    }
}

// ═══════════════════════════════════════════════════════════════
// RayDifferential(微分射线)
// ═══════════════════════════════════════════════════════════════

/// 屏幕空间 x/y 方向的步进差。
///
/// 存完整的光线偏移（origin + dir），而不是只存方向差：
/// 景深/非针孔相机下，相邻样本的 origin 也可能不同。
#[derive(Clone, Copy, Debug)]
pub struct RayDifferentials {
    pub dx_origin: Vec3,
    pub dx_dir: Vec3,
    pub dy_origin: Vec3,
    pub dy_dir: Vec3,
}

/// 微分光线。`differentials` 为 `None` 时退化为普通光线。
///
/// 微分信息用于纹理过滤：屏幕空间步进差打到表面后转成
/// 纹理坐标偏导，组成雅可比矩阵，决定 mipmap 层级和各向异性。
#[derive(Clone, Copy, Debug)]
pub struct RayDifferential {
    /// 主射线。`time` 由主射线携带，微分偏移不改变时刻。
    pub ray: Ray,

    /// 可选的微分偏移。
    ///
    /// `None` 表示这条光线没有微分信息（例如内部反弹、阴影射线、
    /// 或相机模型未生成微分）。`None` 不是错误，调用方应退化为
    /// 普通光线处理，不应用纹理过滤的微分路径。
    pub differentials: Option<RayDifferentials>,
}

impl RayDifferential {
    pub const fn new(ray: Ray) -> Self {
        RayDifferential {
            ray,
            differentials: None,
        }
    }

    pub const fn with_differentials(ray: Ray, differentials: RayDifferentials) -> Self {
        RayDifferential {
            ray,
            differentials: Some(differentials),
        }
    }

    /// 按采样率缩放微分，避免多采样时纹理过度模糊。
    ///
    /// `scale` 通常是 `1 / sqrt(spp)`，带下限。
    /// pbrt 用 max(0.125, 1/sqrt(spp))，外部钳制，不写在这里。
    #[inline]
    pub fn scale_differentials(&mut self, scale: f32) {
        if let Some(d) = &mut self.differentials {
            d.dx_origin *= scale;
            d.dx_dir *= scale;
            d.dy_origin *= scale;
            d.dy_dir *= scale;
        }
    }

    /// 从表面点生成一条新的微分光线。
    ///
    /// 语义：
    /// - 新光线的 origin 微分归零：所有微分光线从同一个表面点出发；
    /// - 新光线的 `time` 继承主光线的 `time`；
    /// - 方向微分根据距离、表面几何、事件类型（反射/透射）和折射率
    ///   更新，保证后续表面的纹理过滤使用正确的屏幕空间导数；
    /// - 若 `self.differentials` 为 `None`，返回的微分光线也为 `None`。
    ///
    /// 方向微分的更新依赖表面局部几何和 BSDF 采样方向，完整实现需要
    /// 参考光线微分传播公式（距离缩放、法线投影、反射/折射方向对
    /// 微分的线性变换）。当前只定义接口，不提供近似实现。
    /// 数值退化（零长度微分、法线异常等）时，实现应将 `differentials`
    /// 置 `None`，退化为普通光线，而不是引入 `Result` 到热路径。
    pub fn spawn_ray(&self, _params: SpawnRayParams) -> RayDifferential {
        todo!("微分光线传播：origin 微分归零，方向微分按距离/表面几何/eta 更新")
    }
}

/// 屏幕空间 x/y 方向步进差在表面上的结果。
///
/// 由微分光线打到表面后显式传播得到，描述“相邻像素对应的表面点
/// 在位置和纹理坐标上差了多少”。它是屏幕空间到纹理空间的局部
/// 线性近似（雅可比矩阵）的来源，用于 mipmap 层级选择和各向异性过滤。
#[derive(Clone, Copy, Debug)]
pub struct SurfaceDifferentials {
    /// 位置对屏幕 x 的偏导。
    pub dpdx: Vec3,
    /// 位置对屏幕 y 的偏导。
    pub dpdy: Vec3,
    /// 纹理坐标 u 对屏幕 x 的偏导。
    pub dudx: f32,
    /// 纹理坐标 v 对屏幕 x 的偏导。
    pub dvdx: f32,
    /// 纹理坐标 u 对屏幕 y 的偏导。
    pub dudy: f32,
    /// 纹理坐标 v 对屏幕 y 的偏导。
    pub dvdy: f32,
}

/// 生成微分光线的参数。
///
/// 微分传播需要两类信息：
/// - 新光线的几何：起点、方向；
/// - 表面交互信息：用于更新方向微分，使其反映距离、表面朝向和
///   反射/折射对屏幕空间步进的影响。
///
/// 参数显式传入，避免把表面交互状态塞进 `RayDifferential` 或
/// 依赖隐式上下文。`time` 不在这里——微分光线继承主光线的 `time`，
/// 与 `SpawnRayParams` 无关。
#[derive(Clone, Copy, Debug)]
pub struct SpawnRayParams {
    /// 新光线起点，通常是偏移后的表面交点。
    pub origin: Vec3,
    /// 新光线方向，单位向量。
    pub dir: Vec3,
    /// 原光线从起点到交点的实际传播距离，用于方向微分的距离缩放。
    /// 不依赖 `dir` 是否归一化。
    pub distance: f32,
    /// 交点处的几何法线。
    pub geometric_normal: Vec3,
    /// 交点处的着色法线。
    pub shading_normal: Vec3,
    /// 透射时的相对折射率；反射或非折射表面为 1.0。
    pub eta: f32,
    /// 是否为透射（折射）事件。反射事件为 false。
    pub is_transmission: bool,
}


#[derive(Clone, Copy, Debug)]
pub struct AreaLightHit {
    /// 面积光表面的几何命中。字段与 [`SurfaceHit`] 完全同构——面积光
    /// 也需要几何信息（位置、法线、UV、instance_id），用于 emission 求值、
    /// 偏移、可选材质查询。
    pub hit: SurfaceHit,
    /// 面积光对应的光源 ID。积分器用它查 `Light` 求 `emitted_radiance`。
    pub light_id: LightId,
}
