use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

/// Every error is serialized as the Ultralytics Platform `ErrorResponse`:
/// `{"error": "<message>"}` with an appropriate status code.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    BadRequest(String),

    #[error("Missing or invalid API key")]
    Unauthorized,

    #[error("{0}")]
    NotFound(String),

    #[error("{0}")]
    TooLarge(String),

    #[error("{0}")]
    Upstream(String),

    #[error("{0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, ApiError>;

impl ApiError {
    pub fn status(&self) -> StatusCode {
        match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::Unauthorized => StatusCode::UNAUTHORIZED,
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::TooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            ApiError::Upstream(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({ "error": self.to_string() });
        (self.status(), Json(body)).into_response()
    }
}
