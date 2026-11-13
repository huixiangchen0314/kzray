# 单步采样抽象

## Light

本模块建模渲染方程中与光源相关的两项——入射项 `L_i` 与发射项 `L_e`——
在**单步采样**层面的接口契约，以及多光源场景下的选择概率分解。它不建模
光源的几何、发光分布数据或场景光源集合，这些属于实现层。

## 分层结构

| 类型 / trait                      | 职责                                                             |
|-----------------------------------|------------------------------------------------------------------|
| `LightType`                       | 光源类型标签，用于 `is_delta()` 判定与能力 trait 的契约锚点      |
| `LiSample` / `LiGeometry`         | 着色点看向光源的一次采样结果（对应 `L_i`）                       |
| `LeSample` / `LeGeometry`         | 光源看向场景的一次采样结果（对应 `L_e`，BDPT 用）                |
| `Light`                           | 所有光源共享的采样契约（`sample_li` / `pdf_li` / `phi`）         |
| `AreaLight: Light`                | 面光源能力，增加 `emitted_radiance`                              |
| `InfiniteLight: Light`            | 无限远光源能力，增加 `emitted_radiance` / `sample_le` / `pdf_le` |
| `LightSampler` / `LightSelection` | 多光源选择策略                                                   |

拆分 `AreaLight` / `InfiniteLight` 的目的是让“不适用的能力”在类型层面
不存在，而不是用 `S::zero()` / `unreachable!()` 兜底。具体某个 `Light`
实例具备哪些能力，由实现层自行枚举分派，抽象层不做类型枚举。

## 维度管理

### 谁消耗维度

| 方法                       | 接收                          | 消耗维度 | 报告方式                   |
|----------------------------|-------------------------------|----------|----------------------------|
| `Light::sample_li`         | 调用方预取 `Vec2`             | 不消耗   | —                          |
| `Light::pdf_li`            | `&LightSampleContext`, `Vec3` | 不消耗   | —                          |
| `InfiniteLight::sample_le` | 调用方预取 `Vec2`             | 不消耗   | —                          |
| `InfiniteLight::pdf_le`    | `Vec3`                        | 不消耗   | —                          |
| `LightSampler::sample`     | `&mut impl Sampler`           | 内部消耗 | `LightSelection::consumed` |

`Light` 侧全部接收预取样本、不管理 `Sampler` 状态；`LightSampler::sample`
内部取样本、必须报告 `consumed`。

### 积分器完整维度流程

从像素样本到 BSDF 采样：

```text
dim0 = pixel_sample_dimension

→ LightSampler::sample(ctx, sampler)
  消耗 c1 维，返回 LightSelection { consumed: c1 }
  dim1 = dim0 + c1

→ sampler.next_2d(dim1) 取 2D 样本
  Light::sample_li(ctx, sample_2d, wavelengths)
  不消耗维度
  dim2 = dim1 + 2

→ BSDF::sample_f(sampler, dim2, mode)
  消耗 c2 维，返回 BxDFSample { consumed: c2 }
  dim3 = dim2 + c2

→ sampler.next_f32(dim3) 取 RR 样本
  dim4 = dim3 + 1
```

### 失败恢复

- `LightSampler::sample` 返回 `None` →
  `dim1 = dim0 + sampler.max_sample_dimensions()`
- `BSDF::sample_f` 返回 `None` →
  `dim3 = dim2 + bsdf.max_sample_dimensions()`
- `Light::sample_li` 返回 `None` → 调用方已预取 `Vec2`，`dim2 = dim1 + 2`
  不变（失败通常意味着路径终止）

失败恢复是保守估计，不要求精确；精确追踪只在成功路径上。

## 测度约定

| 字段 | 测度 | 用途 |
|---|---|---|
| `LiSample::pdf` | 立体角 | 着色点看向光源，MIS 中与 BSDF 立体角测度直接相加 |
| `LightSelection::light_pdf` | 离散概率 | 乘以 `LiSample::pdf` 得总 PDF |
| `LeSample::pdf` | 方向测度（从光源点看的立体角） | 光源侧方向测度，BDPT 连接需几何项转换 |
| `BxDFSample::pdf` | 立体角 | BSDF 路径，与 `LiSample::pdf` 同测度 |

所有 MIS 权重计算前，必须确认参与比较的 PDF 在同一测度上。立体角测度
之间可以直接相加；立体角与面积/离散概率之间需要先转换。

### 总 PDF 的组合

光源采样路径的总 PDF：

```text
total_pdf = LightSelection::light_pdf × LiSample::pdf
```

- `light_pdf`：离散概率，选择该光源的概率；
- `LiSample::pdf`：光源内部采样的立体角测度 PDF；
- 两者相乘得到“从着色点看，采样到 `wi` 方向”的总立体角测度 PDF；
- MIS 计算时与 `BSDF::pdf(wi)` 相加，两者都是立体角测度。

delta 光源（`LightType::is_delta()`）的 `LiSample::pdf` 无有限值，
不参与 MIS，总 PDF 无意义，调用方应直接以“采样命中”的权重贡献。

### 面积测度到立体角测度的转换

`LiSample::pdf` 要求立体角测度。如果实现内部在光源表面面积上采样：

```text
pdf_omega = pdf_area × distance² / |cos(θ_light)|
```

- `distance`：着色点到光源点的距离，即 `LiGeometry::Surface::distance`；
- `cos(θ_light)`：光源表面法线与 `-wi` 的夹角余弦（光源法线朝向着色点）；
- 调用方不应自行做此转换——`LiSample::pdf` 已转换完毕。

## PDF 一致性契约

MIS 权重的正确性依赖以下等式。**跨实现测试必须校验**：

### `Light::sample_li` / `Light::pdf_li` 一致性

对任意 `s = sample_li(ctx, sample, wavelengths)`：

```text
pdf_li(ctx, s.wi) == s.pdf
```

delta 光源除外：`sample_li` 的 `pdf` 无有限值，`pdf_li` 返回 `0`。

### `Light::sample_li` / `InfiniteLight::sample_le` 一致性

仅 `InfiniteLight` 实现。对同一方向 `d`：

```text
sample_li(...).incident  == sample_le(...).emitted
sample_li(...).pdf       == sample_le(...).pdf
```

其中 `sample_li` 的 `wi = -sample_le(...).direction`。推荐复用
`InfiniteLight::sample_li_via_le` 保证两条路径不分叉。

### `InfiniteLight::sample_le` / `pdf_le` 一致性

对任意 `s = sample_le(sample, wavelengths)`：

```text
pdf_le(s.direction, wavelengths) == s.pdf
```

delta 光源除外。

### `LightSampler::sample` / `pdf` 一致性

对任意 `sel = sample(ctx, sampler)`：

```text
pdf(ctx, sel.light_id) == sel.light_pdf
```

这是光源采样路径 MIS 权重的根基。BSDF 采样路径用 `pdf(ctx, light_id)`
反查同一概率，两者不一致则 MIS 权重错。

### `Light::sample_li` / `AreaLight::emitted_radiance` 一致性

仅 `AreaLight` 实现。`sample_li` 返回的 `incident` 应等于在采样点处
调用 `emitted_radiance(light_position, light_normal, uv, -wi, wavelengths)`
的结果。`LiSample` 不携带 `light_normal` / `uv`，调用方不应尝试从
`LiSample` 复算；该求值用于不经过 `sample_li` 的直接场景（如 BDPT）。

### `InfiniteLight::emitted_radiance` / `sample_le` 一致性

对任意 `s = sample_le(sample, wavelengths)`：

```text
emitted_radiance(Ray::new(_, s.direction)) == s.emitted
```

三者（`emitted_radiance` / `sample_le` / `pdf_le`）必须来自同一发射分布，
在同一 trait 上成对出现就是为了让这条约束无法被实现方拆散。

## 传输方向

光源侧方法隐含传输方向，无 `TransportMode` 参数：

| 方法 | 隐含传输方向 | 含义 |
|---|---|---|
| `Light::sample_li` / `pdf_li` | `Radiance` | 从着色点看向光源 |
| `AreaLight::emitted_radiance` | `Radiance` | 给定 `wo`（光源射向场景）求 `L_e` |
| `InfiniteLight::sample_le` / `pdf_le` / `emitted_radiance` | `Importance` | 从光源射向场景 |

光源侧不需要 `mode`，因为“光源发射什么”是物理量，与“从哪边看”无关。
BDPT 连接两端时，光源侧用 `Importance`，相机侧用 `Radiance`，中间通过
`BxDF` 的 `mode` 参数切换。

## 与 BSDF 模块的对称性

`light.rs` 与 `bsdf.rs` 是渲染方程两侧的镜像：

| 概念       | `light.rs`  | `bsdf.rs`  |
| 采样入口   | `Light::sample_li(ctx, sample_2d)`   | `BSDF::sample_f(sampler, dimension, mode)`               |
| 采样结果   | `LiSample { wi, incident, pdf, geometry }`   | `BxDFSample { wi, value, weight, pdf, flags, consumed }` |
| PDF 反查   | `pdf_li(ctx, wi)`     | `BSDF::pdf(wi, mode)`  |
| delta 判定 | `LightType::is_delta()`   | `BxDFFlags::is_pure_delta()` |
| 一致性契约 | `pdf_li(ctx, s.wi) == s.pdf`   | `pdf(s.wi) == s.pdf` |
| 维度管理   | `sample_li` 不消耗；`LightSampler::sample` 报告 `consumed` | `sample_f` 报告 `consumed`    |

差异是本质的：

- **几何形态**：`LiSample` 携带 `LiGeometry`（`Surface` / `Direction`），
  `BxDFSample` 不需要——BSDF 作用在同一表面点，没有“光源点”概念；
- **权重封装**：`BxDFSample` 直接返回 `weight = value × |cos| / pdf`；
  `LiSample` 只返回 `incident` 和 `pdf`，`cos` 由调用方根据着色点法线计算。
  原因：`LiSample::incident` 是辐射亮度，乘 `cos` 是着色点几何项，
  不属于光源采样；`BxDFSample` 的 `cos` 是 BSDF 半球积分的固有部分；
- **维度管理**：`Light::sample_li` 不消耗维度（调用方预取 `Vec2`）；
  `BSDF::sample_f` 内部消耗维度（组合采样需要动态维度）。

## 实现层注意事项

### 能力 trait 与 `LightType` 的映射（实现层测试应校验）

- 实现 `AreaLight` ⇒ `light_type()` 必须返回 `LightType::Area`；
- 实现 `InfiniteLight` 且非 delta ⇒ `light_type()` 返回 `LightType::Infinite`；
- 实现 `InfiniteLight` 且 delta（方向光）⇒ `light_type()` 返回
  `LightType::DeltaDirection`；
- 点光源 ⇒ `LightType::DeltaPosition`；
- 一个类型不应同时实现 `AreaLight` 和 `InfiniteLight`。

### 不枚举实现

抽象层不提供 `as_area()` / `as_infinite()` 之类的能力查询。持有具体
实例的实现层（场景构建 / 积分器分发）自行枚举并 match。`Box<dyn Light>`
只用于 `LightSampler` 的光源表存储，不用于向下转型。

### `LightSampler` 与 trait object

`LightSampler::sample` 接受 `&mut impl Sampler`，因此
`LightSampler` 本身不是 object-safe 的（不能用 `Box<dyn LightSampler>`）。
需要运行时切换策略时，在实现层定义一个 enum 包装具体实现，
`sample` 内部 `match` 分派，统一用 `&mut dyn Sampler` 转发。虚分派
限制在包装层，具体实现仍然单态化。

### `LiGeometry::Surface::distance` 一致性

`distance` 应等于 `(position - ctx.position).length()`。实现时必须保持
一致，否则阴影射线 `t_max` 和面积↔立体角测度转换会出错。跨实现测试
应校验此等式。

## BSDF

