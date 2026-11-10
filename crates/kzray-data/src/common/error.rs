//! 数据层错误。只描述"数据不合法"，不描述"渲染失败"。

use thiserror::Error;

use super::id::EntityId;

#[derive(Debug, Error)]
pub enum DataError {
    /// 字符串名字哈希后与已有 id 冲突（两个不同名字映射到同一 u64）
    #[error("id 冲突：名字 `{new_name}` 与已有 `{existing_name}` 都哈希到 {id:?}")]
    IdCollision {
        new_name: String,
        existing_name: String,
        id: EntityId,
    },

    /// 同一 id 被插入两次（名字相同，值不同）
    #[error("id {id:?} 已存在，不允许重复插入")]
    DuplicateId { id: EntityId },

    /// 数据里引用了不存在的实体
    #[error("引用了不存在的实体：{0:?}")]
    MissingRef(EntityId),

    /// 参数值不合法（NaN / Inf / 负半径等，具体校验在域代码里做）
    #[error("参数不合法（{path}）：{reason}")]
    InvalidParam { path: String, reason: String },
}
