//! 库提供的便利错误类型。用户装配时可用，也可自定义。

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BsdfError {
    #[error("节点 `{kind}` 的参数不合法：{reason}")]
    InvalidParam { kind: String, reason: String },
}

#[derive(Debug, Error)]
pub enum VolumeError {
    #[error("体积 `{kind}` 的参数不合法：{reason}")]
    InvalidParam { kind: String, reason: String },
}
