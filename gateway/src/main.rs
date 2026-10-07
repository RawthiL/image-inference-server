use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use gateway::api::{self, AppState};
use gateway::config::Config;
use gateway::error::ApiError;
use gateway::sources;
use gateway::triton::TritonClient;
use tracing_subscriber::EnvFilter;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    if let Err(e) = rt.block_on(run()) {
        eprintln!("fatal: {e}");
        std::process::exit(1);
    }
}

fn parse_args() -> PathBuf {
    let mut config = std::env::var("GATEWAY_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("config.yaml"));

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" => {
                if let Some(p) = args.get(i + 1) {
                    config = PathBuf::from(p);
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!(
                    "usage: gateway [--config path/to/config.yaml]\n\n\
                     Ultralytics-compatible /predict gateway in front of Triton Inference Server.\n\
                     Model-agnostic: the backend is picked from the Triton model metadata."
                );
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    config
}

async fn run() -> Result<(), ApiError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config_path = parse_args();
    let mut cfg = Config::load(&config_path)?;
    // Deployment override (docker compose sets it to the Triton service), so
    // a dev config.yaml pointing at localhost still works in containers.
    if let Some(url) = std::env::var("TRITON_URL").ok().filter(|u| !u.is_empty()) {
        cfg.triton.url = url;
    }
    let cfg = Arc::new(cfg);
    tracing::info!(
        "loaded {} (model='{}' -> triton '{}' at {}, {} api key(s))",
        config_path.display(),
        cfg.default_model,
        cfg.triton_model(),
        cfg.triton.url,
        cfg.api_keys.len()
    );

    let triton = Arc::new(TritonClient::new(
        &cfg.triton.url,
        cfg.triton.api_key.as_deref(),
        Duration::from_millis(cfg.triton.timeout_ms),
    ));
    let fetch =
        sources::fetch_client(&cfg.limits).map_err(|e| ApiError::Internal(e.to_string()))?;

    // Serve immediately; the model loads in the background so clients get a
    // JSON 503 ("model ... is not ready: <reason>") rather than a dead port.
    let state = Arc::new(AppState::new(cfg.clone(), triton, fetch));
    api::spawn_model_loader(state.clone());

    let app = api::router(state);
    let listener = tokio::net::TcpListener::bind(&cfg.server.listen)
        .await
        .map_err(|e| ApiError::Internal(format!("cannot bind {}: {e}", cfg.server.listen)))?;
    tracing::info!("listening on http://{}", cfg.server.listen);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.ok();
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received");
}
