use thiserror::Error;

#[derive(Debug, Error)]
pub enum BsdfError {
    #[error("未知 BSDF kind: `{0}`")]
    UnknownKind(String),
    #[error("节点 `{kind}` 的参数不合法：{reason}")]
    InvalidParam { kind: String, reason: String },
    #[error("节点 `{kind}` 引用的 Field 无法解析：{reason}")]
    FieldRef { kind: String, reason: String },
}

#[derive(Debug, Error)]
pub enum FieldError {
    #[error("未知 Field kind: `{0}`")]
    UnknownKind(String),
    #[error("Field `{kind}` 的参数不合法：{reason}")]
    InvalidParam { kind: String, reason: String },
}

#[derive(Debug, Error)]
pub enum VolumeError {
    #[error("未知 Volume kind: `{0}`")]
    UnknownKind(String),
    #[error("Volume `{kind}` 的参数不合法：{reason}")]
    InvalidParam { kind: String, reason: String },
}
