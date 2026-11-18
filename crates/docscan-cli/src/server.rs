//! HTTP API.
//!
//! - `POST /v1/documents/scans` (multipart): `file` (JPEG, PNG, WebP, TIFF,
//!   BMP; required), optional `mode` (color | gray | bw | original),
//!   `geometry` (dewarp | perspective | none), `corners` (four photo points,
//!   `x,y,x,y,x,y,x,y` or JSON: skips detection), `format` (png | jpeg |
//!   pdf), `dpi`, `paper` (auto | a4 | letter), `quality`, `max_side`;
//!   `model` is ignored (one model per process). Returns the file, with
//!   `X-Doc-Corners` (JSON, photo pixels TL TR BR BL), `X-Doc-Score`,
//!   `X-Doc-Size` (WxH) and `X-Doc-Warning` when the geometry is doubtful.
//! - `POST /v1/documents/detect` (multipart `file`): JSON corners, score,
//!   coverage, size, warnings, and the photo's width / height (for a live
//!   camera overlay; no rendering).
//! - `GET /health`, `GET /v1/models`.
//!
//! Requests run one at a time (a mutex): one image already uses every core.

use anyhow::{Context, Result};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use docscan::{Format, Options, Paper, Scanner};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

pub struct AppState {
    scanner: Arc<Mutex<Scanner>>,
    model_id: String,
}

type Shared = Arc<AppState>;

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = if self.0.is_client_error() {
            "invalid_request_error"
        } else {
            "server_error"
        };
        let body = json!({"error": {"message": self.1, "type": kind}});
        (self.0, Json(body)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))
    }
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

pub async fn serve(scanner: Scanner, model_id: String, host: &str, port: u16) -> Result<()> {
    let state = Arc::new(AppState {
        scanner: Arc::new(Mutex::new(scanner)),
        model_id,
    });
    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models))
        .route("/v1/documents/scans", post(scans))
        .route("/v1/documents/detect", post(detect))
        .layer(DefaultBodyLimit::max(256 << 20))
        .with_state(state);
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    tracing::info!("listening on http://{addr}");
    eprintln!("docscan server on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn models(State(st): State<Shared>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": st.model_id,
            "object": "model",
            "owned_by": "docscan",
            "modes": ["color", "gray", "bw", "original"],
            "formats": ["png", "jpeg", "pdf"],
        }]
    }))
}

struct Request {
    file: Vec<u8>,
    filename: Option<String>,
    opts: Options,
    format: Format,
    dpi: u16,
    paper: Paper,
    quality: u8,
}

fn parse<T: std::str::FromStr>(field: &str, v: &str) -> Result<T, ApiError>
where
    T::Err: std::fmt::Display,
{
    v.trim().parse().map_err(|e| bad(format!("{field}: {e}")))
}

async fn parse_multipart(headers: &HeaderMap, body: Bytes) -> Result<Request, ApiError> {
    let ct = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let boundary = multer::parse_boundary(ct)
        .map_err(|_| bad("expected multipart/form-data with a `file` field"))?;
    let stream = futures_util::stream::once(async move { Ok::<_, std::io::Error>(body) });
    let mut mp = multer::Multipart::new(stream, boundary);
    let mut r = Request {
        file: Vec::new(),
        filename: None,
        opts: Options::default(),
        format: Format::Png,
        dpi: 300,
        paper: Paper::Auto,
        quality: 90,
    };
    while let Some(field) = mp
        .next_field()
        .await
        .map_err(|e| bad(format!("multipart: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" || name == "image" {
            r.filename = field.file_name().map(String::from);
            r.file = field
                .bytes()
                .await
                .map_err(|e| bad(format!("file: {e}")))?
                .to_vec();
            continue;
        }
        let v = field.text().await.map_err(|e| bad(e.to_string()))?;
        if v.trim().is_empty() {
            continue;
        }
        match name.as_str() {
            "mode" => r.opts.mode = parse("mode", &v)?,
            "geometry" => r.opts.geometry = parse("geometry", &v)?,
            "corners" => {
                r.opts.corners =
                    Some(docscan::scan::parse_corners(&v).map_err(|e| bad(format!("{e:#}")))?)
            }
            "max_side" => r.opts.max_side = parse("max_side", &v)?,
            "format" | "response_format" => r.format = parse("format", &v)?,
            "dpi" => {
                r.dpi = parse("dpi", &v)?;
                if !(50..=1200).contains(&r.dpi) {
                    return Err(bad("dpi: 50 to 1200"));
                }
            }
            "paper" => r.paper = parse("paper", &v)?,
            "quality" => r.quality = parse("quality", &v)?,
            _ => {} // model…: ignored
        }
    }
    if r.file.is_empty() {
        return Err(bad("missing `file`"));
    }
    Ok(r)
}

fn decode(bytes: &[u8]) -> Result<docscan::image::Image, ApiError> {
    docscan::image::decode(bytes).map_err(|e| bad(format!("{e:#}")))
}

async fn scans(
    State(st): State<Shared>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let req = parse_multipart(&headers, body).await?;
    let base = req
        .filename
        .as_deref()
        .map(|f| f.rsplit_once('.').map(|(b, _)| b).unwrap_or(f).to_string())
        .filter(|b| !b.is_empty())
        .unwrap_or_else(|| "document".into());
    let guard = st.scanner.clone().lock_owned().await;
    let format = req.format;
    let (bytes, scan_meta, elapsed) =
        tokio::task::spawn_blocking(move || -> Result<_, ApiError> {
            let t = Instant::now();
            let photo = decode(&req.file)?;
            let s = guard.scan(&photo, &req.opts)?;
            let bytes = docscan::encode(&s.image, req.format, req.quality, req.dpi, req.paper)
                .map_err(|e| bad(format!("{e:#}")))?;
            Ok((
                bytes,
                (s.image.width, s.image.height, s.detection, s.geometry),
                t.elapsed().as_secs_f64(),
            ))
        })
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("worker: {e}")))??;
    let (w, h, det, geometry) = scan_meta;
    tracing::info!(
        "scan {w}x{h} ({geometry:?}{}) in {elapsed:.2}s",
        det.as_ref()
            .map(|d| format!(", score {:.2}", d.score))
            .unwrap_or_default()
    );

    let mut r = bytes.into_response();
    let hm = r.headers_mut();
    hm.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(format.mime()),
    );
    set_disposition(hm, &format!("{base}_scan.{}", format.ext()));
    let mut put = |k: &'static str, v: String| {
        if let Ok(v) = HeaderValue::from_str(&v) {
            hm.insert(k, v);
        }
    };
    put("x-doc-size", format!("{w}x{h}"));
    if let Some(d) = det {
        put(
            "x-doc-corners",
            serde_json::to_string(&d.corners).unwrap_or_default(),
        );
        put("x-doc-score", format!("{:.3}", d.score));
        if !d.warnings.is_empty() {
            put("x-doc-warning", d.warnings.join("; "));
        }
    }
    Ok(r)
}

async fn detect(
    State(st): State<Shared>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    let req = parse_multipart(&headers, body).await?;
    let guard = st.scanner.clone().lock_owned().await;
    let (d, w, h) = tokio::task::spawn_blocking(move || -> Result<_, ApiError> {
        let photo = decode(&req.file)?;
        Ok((guard.detect(&photo)?, photo.width, photo.height))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("worker: {e}")))??;
    let mut v = serde_json::to_value(&d)
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    v["width"] = w.into();
    v["height"] = h.into();
    Ok(Json(v))
}

fn set_disposition(hm: &mut HeaderMap, filename: &str) {
    // ASCII only: other characters become '_'
    let safe: String = filename
        .chars()
        .map(|c| {
            if c.is_ascii_graphic() && c != '"' && c != '\\' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{safe}\"")) {
        hm.insert(header::CONTENT_DISPOSITION, v);
    }
}
