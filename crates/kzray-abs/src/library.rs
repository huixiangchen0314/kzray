//! kzray-impl/src/library.rs
//!
//! 统一数据资产表。
//!
//! 所有资源**扁平排列**——每个通道一个独立 `BTreeMap`，没有
//! `TextureLibrary` 中间层、没有 `TypeId` 分发、没有 `downcast`。
//!
//! # id 由外部分配
//!
//! `TextureId` 不由 `Library` 内部生成，而是由场景构建方/加载方分配
//! （可能来自 DCC、序列化文件或场景层的 id 分配器）。`Library` 只负责
//! 存储和**冲突检测**：`add_*` 接受 `(id, texture)`，若 `id` 已在任何
//! 通道中被使用，panic。id 全局唯一——一个 `TextureId` 至多命中一个
//! 通道的 map。
//!
//! # 关于 `S` 泛型
//!
//! 光谱纹理（`AlbedoTexture<S>` 等）**必须**绑定光谱代数 `S`——这是
//! 抽象层定义的，不可回避。`Library<S>` 因此带 `S`。这是**类型参数
//! 传播**，不是运行时成本——整个渲染器在实例化时已确定 `S`，所有 `S`
//! 相同的调用共享同一份单态化代码。

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    AlbedoTexture, AlphaTexture, ConductorF0Texture, EmissiveTexture, HeightTexture, IorTexture,
    MetallicTexture, NormalTexture, RoughnessTexture, SpectralAlgebra, TransmissionTexture,
};
use kzray_data::{Mesh, MeshId, TextureId};

// ═══════════════════════════════════════════════════════════════
// Library
// ═══════════════════════════════════════════════════════════════

/// 统一数据资产表。所有跨层共享的资源在此唯一持有，通过 `&Library<S>`
/// 被 `accel`、`material`、`scene` 共享借用。
///
/// 构建期填充，构建后只读。不持有着色资源（`Material<S>`、`Light<S>`）——
/// 那些与具体材质/光源实现绑定，由 `Scene<S>` 持有。
pub struct Library<S: SpectralAlgebra> {
    // ── 网格 ──────────────────────────────────────────────
    /// 网格数据。求交层做三角形求交，着色层插值顶点属性。
    pub meshes: BTreeMap<MeshId, Mesh>,

    // ── 标量通道纹理 ──────────────────────────────────────
    pub alpha_textures: BTreeMap<TextureId, Box<AlphaTexture>>,
    pub roughness_textures: BTreeMap<TextureId, Box<RoughnessTexture>>,
    pub metallic_textures: BTreeMap<TextureId, Box<MetallicTexture>>,
    pub ior_textures: BTreeMap<TextureId, Box<IorTexture>>,
    pub height_textures: BTreeMap<TextureId, Box<HeightTexture>>,

    // ── 向量通道纹理 ──────────────────────────────────────
    pub normal_textures: BTreeMap<TextureId, Box<NormalTexture>>,

    // ── 光谱通道纹理 ──────────────────────────────────────
    pub albedo_textures: BTreeMap<TextureId, Box<AlbedoTexture<S>>>,
    pub emissive_textures: BTreeMap<TextureId, Box<EmissiveTexture<S>>>,
    pub transmission_textures: BTreeMap<TextureId, Box<TransmissionTexture<S>>>,
    pub conductor_f0_textures: BTreeMap<TextureId, Box<ConductorF0Texture<S>>>,

    /// 所有已使用的 `TextureId`。`add_*` 前检查冲突——id 全局唯一，
    /// 一个 id 至多在一个通道的 map 中存在。
    used_texture_ids: BTreeSet<TextureId>,
}

impl<S: SpectralAlgebra> Library<S> {
    pub fn new() -> Self {
        Self {
            meshes: BTreeMap::new(),
            alpha_textures: BTreeMap::new(),
            roughness_textures: BTreeMap::new(),
            metallic_textures: BTreeMap::new(),
            ior_textures: BTreeMap::new(),
            height_textures: BTreeMap::new(),
            normal_textures: BTreeMap::new(),
            albedo_textures: BTreeMap::new(),
            emissive_textures: BTreeMap::new(),
            transmission_textures: BTreeMap::new(),
            conductor_f0_textures: BTreeMap::new(),
            used_texture_ids: BTreeSet::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
            && self.used_texture_ids.is_empty()
    }

    // ── 网格 ──────────────────────────────────────────────

    /// 添加网格。
    ///
    /// # Panics
    ///
    /// `id` 已存在时 panic——网格 id 由外部保证唯一。
    #[track_caller]
    pub fn add_mesh(&mut self, id: MeshId, mesh: Mesh) {
        let old = self.meshes.insert(id, mesh);
        assert!(
            old.is_none(),
            "duplicate MeshId {id:?}: already exists in Library"
        );
    }

    /// 查网格，不存在则 panic。编译期已校验 id 的热点路径用。
    #[track_caller]
    pub fn expect_mesh(&self, id: MeshId) -> &Mesh {
        self.meshes
            .get(&id)
            .unwrap_or_else(|| panic!("mesh {id:?} not found in Library"))
    }

    // ── 纹理：每个通道一对 add/get ────────────────────────
    //
    // `add_*` 接受外部分配的 `TextureId`，冲突则 panic。
    // `get_*` 按 id 在对应通道的 `BTreeMap` 中查，返回 `&dyn Texture<C>`。

    // ── 标量通道 ──────────────────────────────────────────

    #[track_caller]
    pub fn add_alpha(&mut self, id: TextureId, tex: Box<AlphaTexture>) {
        self.reserve_texture_id(id);
        self.alpha_textures.insert(id, tex);
    }

    #[track_caller]
    pub fn alpha(&self, id: TextureId) -> &AlphaTexture {
        self.alpha_textures
            .get(&id)
            .map(|b| b.as_ref())
            .unwrap_or_else(|| panic!("alpha texture {id:?} not found in Library"))
    }

    #[track_caller]
    pub fn add_roughness(&mut self, id: TextureId, tex: Box<RoughnessTexture>) {
        self.reserve_texture_id(id);
        self.roughness_textures.insert(id, tex);
    }

    #[track_caller]
    pub fn roughness(&self, id: TextureId) -> &RoughnessTexture {
        self.roughness_textures
            .get(&id)
            .map(|b| b.as_ref())
            .unwrap_or_else(|| panic!("roughness texture {id:?} not found in Library"))
    }

    #[track_caller]
    pub fn add_metallic(&mut self, id: TextureId, tex: Box<MetallicTexture>) {
        self.reserve_texture_id(id);
        self.metallic_textures.insert(id, tex);
    }

    #[track_caller]
    pub fn metallic(&self, id: TextureId) -> &MetallicTexture {
        self.metallic_textures
            .get(&id)
            .map(|b| b.as_ref())
            .unwrap_or_else(|| panic!("metallic texture {id:?} not found in Library"))
    }

    #[track_caller]
    pub fn add_ior(&mut self, id: TextureId, tex: Box<IorTexture>) {
        self.reserve_texture_id(id);
        self.ior_textures.insert(id, tex);
    }

    #[track_caller]
    pub fn ior(&self, id: TextureId) -> &IorTexture {
        self.ior_textures
            .get(&id)
            .map(|b| b.as_ref())
            .unwrap_or_else(|| panic!("ior texture {id:?} not found in Library"))
    }

    #[track_caller]
    pub fn add_height(&mut self, id: TextureId, tex: Box<HeightTexture>) {
        self.reserve_texture_id(id);
        self.height_textures.insert(id, tex);
    }

    #[track_caller]
    pub fn height(&self, id: TextureId) -> &HeightTexture {
        self.height_textures
            .get(&id)
            .map(|b| b.as_ref())
            .unwrap_or_else(|| panic!("height texture {id:?} not found in Library"))
    }

    // ── 向量通道 ──────────────────────────────────────────

    #[track_caller]
    pub fn add_normal(&mut self, id: TextureId, tex: Box<NormalTexture>) {
        self.reserve_texture_id(id);
        self.normal_textures.insert(id, tex);
    }

    #[track_caller]
    pub fn normal(&self, id: TextureId) -> &NormalTexture {
        self.normal_textures
            .get(&id)
            .map(|b| b.as_ref())
            .unwrap_or_else(|| panic!("normal texture {id:?} not found in Library"))
    }

    // ── 光谱通道 ──────────────────────────────────────────

    #[track_caller]
    pub fn add_albedo(&mut self, id: TextureId, tex: Box<AlbedoTexture<S>>) {
        self.reserve_texture_id(id);
        self.albedo_textures.insert(id, tex);
    }

    #[track_caller]
    pub fn albedo(&self, id: TextureId) -> &AlbedoTexture<S> {
        self.albedo_textures
            .get(&id)
            .map(|b| b.as_ref())
            .unwrap_or_else(|| panic!("albedo texture {id:?} not found in Library"))
    }

    #[track_caller]
    pub fn add_emissive(&mut self, id: TextureId, tex: Box<EmissiveTexture<S>>) {
        self.reserve_texture_id(id);
        self.emissive_textures.insert(id, tex);
    }

    #[track_caller]
    pub fn emissive(&self, id: TextureId) -> &EmissiveTexture<S> {
        self.emissive_textures
            .get(&id)
            .map(|b| b.as_ref())
            .unwrap_or_else(|| panic!("emissive texture {id:?} not found in Library"))
    }

    #[track_caller]
    pub fn add_transmission(&mut self, id: TextureId, tex: Box<TransmissionTexture<S>>) {
        self.reserve_texture_id(id);
        self.transmission_textures.insert(id, tex);
    }

    #[track_caller]
    pub fn transmission(&self, id: TextureId) -> &TransmissionTexture<S> {
        self.transmission_textures
            .get(&id)
            .map(|b| b.as_ref())
            .unwrap_or_else(|| panic!("transmission texture {id:?} not found in Library"))
    }

    #[track_caller]
    pub fn add_conductor_f0(&mut self, id: TextureId, tex: Box<ConductorF0Texture<S>>) {
        self.reserve_texture_id(id);
        self.conductor_f0_textures.insert(id, tex);
    }

    #[track_caller]
    pub fn conductor_f0(&self, id: TextureId) -> &ConductorF0Texture<S> {
        self.conductor_f0_textures
            .get(&id)
            .map(|b| b.as_ref())
            .unwrap_or_else(|| panic!("conductor F0 texture {id:?} not found in Library"))
    }

    // ── 内部 ──────────────────────────────────────────────

    /// 登记 `id`。已存在则 panic——id 由外部保证全局唯一，
    /// 冲突是构建期 bug。
    ///
    /// `BTreeSet::insert` 返回 `true` 表示 id 之前不存在。
    #[track_caller]
    #[inline]
    fn reserve_texture_id(&mut self, id: TextureId) {
        assert!(
            self.used_texture_ids.insert(id),
            "duplicate TextureId {id:?}: already used by another texture in Library"
        );
    }
}

impl<S: SpectralAlgebra> Default for Library<S> {
    fn default() -> Self {
        Self::new()
    }
}
