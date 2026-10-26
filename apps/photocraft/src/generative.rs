//! Edit › Generative Fill through ComfyUI on this computer, with FLUX.1 Fill.
//!
//! The engine asks for a fill ([`photocraft_engine::generative_cmds`]); this answers it through a
//! ComfyUI server's HTTP API: upload the window and the mask to ComfyUI's scratch area, queue one
//! FLUX.1 Fill graph per variation with its own seed, wait for each, and read the image back.
//!
//! - **Where:** `http://127.0.0.1:8188` unless `PHOTOCRAFT_COMFYUI_URL` says otherwise. Nothing
//!   else is contacted; redirects are refused and every reply is size-bounded.
//! - **Models:** found from what the server lists: a diffusion model whose name contains `fill`,
//!   CLIP-L, a T5-XXL and the FLUX VAE (`ae`). `PHOTOCRAFT_GF_UNET`, `PHOTOCRAFT_GF_CLIP_L`,
//!   `PHOTOCRAFT_GF_T5` and `PHOTOCRAFT_GF_VAE` pick others by file name.
//! - **Size:** the graph scales the window to about a megapixel for the model (FLUX's native
//!   size, never more than 4× up) and back to the window's size, so the engine gets exactly the
//!   pixels it sent.
//! - **No traces:** uploads go to ComfyUI's temp area and results come from a preview node, so
//!   nothing is added to ComfyUI's `input/` or `output/` folders.
//! - **Never blocks the UI:** whether ComfyUI is reachable is checked on a background thread at
//!   most every few seconds; [`GenerativeBackend::unavailable`] only reads the last answer.

use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use photocraft_engine::generative_cmds::{FillImage, FillRequest, GenerativeBackend};
use photocraft_engine::jobs::JobCtx;
use serde_json::{Value, json};

const DEFAULT_URL: &str = "http://127.0.0.1:8188";
/// Pixels the model works at (FLUX.1 is trained near one megapixel).
const MODEL_PIXELS: f64 = 1024.0 * 1024.0;
/// Most a small window is scaled up for the model.
const MAX_UPSCALE: f64 = 4.0;
/// How long one variation may take before giving up (a cold start loads ~20 GB of weights).
const VARIATION_TIMEOUT: Duration = Duration::from_secs(900);
const POLL: Duration = Duration::from_millis(400);
/// How old the reachability answer may get before it is checked again.
const RECHECK: Duration = Duration::from_secs(5);
const MAX_JSON: u64 = 8 * 1024 * 1024;
const MAX_IMAGE: u64 = 512 * 1024 * 1024;
/// FLUX.1 Fill's recommended guidance and steps.
const GUIDANCE: f64 = 30.0;
const STEPS: u32 = 28;
/// How far the model may repaint past the selection's edge (at model size), so the seam falls
/// where the layer mask hides it.
const GROW_MASK: u32 = 8;

/// The model files the graph loads.
#[derive(Clone, Debug, PartialEq)]
struct Models {
    unet: String,
    clip_l: String,
    t5: String,
    vae: String,
}

struct Probe {
    checked: Option<Instant>,
    result: Result<Models, String>,
}

pub struct ComfyFill {
    base: String,
    agent: ureq::Agent,
    probe: Arc<Mutex<Probe>>,
    probing: Arc<AtomicBool>,
}

/// Generative Fill through ComfyUI, for [`photocraft_ui_egui::PlatformServices`].
pub fn service() -> Option<Arc<dyn GenerativeBackend>> {
    let base = std::env::var("PHOTOCRAFT_COMFYUI_URL").ok().filter(|u| !u.trim().is_empty()).unwrap_or_else(|| DEFAULT_URL.to_string());
    let base = base.trim().trim_end_matches('/').to_string();
    if !base.starts_with("http://") {
        log::warn!("Generative Fill: PHOTOCRAFT_COMFYUI_URL must be an http:// address; got {base}");
        return None;
    }
    let config = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .max_redirects(0)
        .max_redirects_will_error(false)
        .timeout_connect(Some(Duration::from_secs(2)))
        .timeout_global(Some(Duration::from_secs(120)))
        .build();
    let fill = ComfyFill {
        base,
        agent: ureq::Agent::new_with_config(config),
        probe: Arc::new(Mutex::new(Probe { checked: None, result: Err("Looking for ComfyUI…".into()) })),
        probing: Arc::new(AtomicBool::new(false)),
    };
    fill.recheck();
    Some(Arc::new(fill))
}

impl GenerativeBackend for ComfyFill {
    fn name(&self) -> String {
        "ComfyUI (FLUX.1 Fill)".into()
    }

    fn unavailable(&self) -> Option<String> {
        let probe = self.probe.lock().unwrap_or_else(PoisonError::into_inner);
        let stale = probe.checked.is_none_or(|t| t.elapsed() > RECHECK);
        let why = probe.result.as_ref().err().cloned();
        drop(probe);
        if stale {
            self.recheck();
        }
        why
    }

    fn fill(&self, req: &FillRequest, job: &JobCtx) -> Result<Vec<FillImage>, String> {
        let models = find_models(self)?;
        job.check_cancel()?;
        job.progress(0.06, "Sending the image to ComfyUI");
        let image = self.upload("source", &encode_png(req.width, req.height, png::ColorType::Rgba, &req.image)?)?;
        let mask = self.upload("mask", &encode_png(req.width, req.height, png::ColorType::Grayscale, &req.mask)?)?;
        let (mw, mh) = model_size(req.width, req.height);
        let base_seed = if req.seed == 0 { nonce() } else { req.seed };
        let mut ids = Vec::new();
        for i in 0..req.variations {
            job.check_cancel()?;
            let graph = flux_fill_graph(&models, &image, &mask, &req.prompt, base_seed.wrapping_add(u64::from(i)), (mw, mh), (req.width, req.height));
            let reply = self.post("/prompt", &json!({"prompt": graph, "client_id": "photocraft"}))?;
            let id = reply.get("prompt_id").and_then(Value::as_str).ok_or_else(|| queue_error(&reply))?;
            ids.push(id.to_string());
        }
        let mut out = Vec::new();
        let n = ids.len().max(1) as f32;
        for (i, id) in ids.iter().enumerate() {
            let at = |f: f32| 0.1 + 0.9 * (i as f32 + f) / n;
            job.progress(at(0.0), &format!("Generating variation {} of {}", i + 1, ids.len()));
            let result = self.wait(id, job, |f| job.progress(at(f), ""));
            let image = match result {
                Ok(v) => v,
                Err(e) => {
                    self.forget(&ids);
                    return Err(e);
                }
            };
            out.push(self.fetch(&image, req.width, req.height)?);
        }
        job.progress(1.0, "Done");
        Ok(out)
    }
}

/// [`JobCtx::check`] as this module's error type.
trait CancelCheck {
    fn check_cancel(&self) -> Result<(), String>;
}
impl CancelCheck for JobCtx {
    fn check_cancel(&self) -> Result<(), String> {
        if self.cancelled() { Err("cancelled".into()) } else { Ok(()) }
    }
}

impl ComfyFill {
    /// Refresh the reachability answer on a background thread (at most one at a time).
    fn recheck(&self) {
        if self.probing.swap(true, Ordering::AcqRel) {
            return;
        }
        let me = ComfyFill { base: self.base.clone(), agent: self.agent.clone(), probe: self.probe.clone(), probing: self.probing.clone() };
        let spawned = std::thread::Builder::new().name("comfyui probe".into()).spawn(move || {
            let result = me.probe_now();
            let mut p = me.probe.lock().unwrap_or_else(PoisonError::into_inner);
            *p = Probe { checked: Some(Instant::now()), result };
            drop(p);
            me.probing.store(false, Ordering::Release);
        });
        if spawned.is_err() {
            self.probing.store(false, Ordering::Release);
        }
    }

    /// Is ComfyUI up, and does it have the models? Blocking; for the probe thread and the worker.
    fn probe_now(&self) -> Result<Models, String> {
        self.get_json("/system_stats").map_err(|_| format!("ComfyUI is not running at {}; start it to use Generative Fill", self.base))?;
        let options = |class: &str, input: &str| -> Result<Vec<String>, String> {
            let info = self.get_json(&format!("/object_info/{class}"))?;
            Ok(info
                .pointer(&format!("/{class}/input/required/{input}/0"))
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
                .unwrap_or_default())
        };
        let unets = options("UNETLoader", "unet_name")?;
        let clips = options("DualCLIPLoader", "clip_name1")?;
        let vaes = options("VAELoader", "vae_name")?;
        pick_models(&unets, &clips, &vaes, |k| std::env::var(k).ok())
    }

    fn check(&self, r: Result<ureq::http::Response<ureq::Body>, ureq::Error>) -> Result<ureq::http::Response<ureq::Body>, String> {
        match r {
            Ok(resp) if resp.status().is_success() => Ok(resp),
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body = read_limited(resp, 64 * 1024).unwrap_or_default();
                let detail = serde_json::from_slice::<Value>(&body).ok().map(|v| queue_error(&v)).unwrap_or_default();
                Err(if detail.is_empty() { format!("ComfyUI answered {status}") } else { format!("ComfyUI answered {status}: {detail}") })
            }
            Err(_) => Err(format!("could not reach ComfyUI at {}", self.base)),
        }
    }

    fn get_json(&self, path: &str) -> Result<Value, String> {
        let resp = self.check(self.agent.get(format!("{}{path}", self.base)).call())?;
        serde_json::from_slice(&read_limited(resp, MAX_JSON)?).map_err(|_| "ComfyUI returned invalid JSON".to_string())
    }

    fn post(&self, path: &str, body: &Value) -> Result<Value, String> {
        let bytes = serde_json::to_vec(body).map_err(|e| e.to_string())?;
        let resp = self.check(self.agent.post(format!("{}{path}", self.base)).content_type("application/json").send(&bytes))?;
        let reply = read_limited(resp, MAX_JSON)?;
        if reply.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&reply).map_err(|_| "ComfyUI returned invalid JSON".to_string())
    }

    /// Upload a PNG to ComfyUI's temp area; the name a LoadImage node reads (`… [temp]`).
    fn upload(&self, role: &str, png: &[u8]) -> Result<String, String> {
        let n = nonce();
        let filename = format!("photocraft-gf-{role}-{}-{n}.png", std::process::id());
        let boundary = format!("photocraft-{n}");
        let mut body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ntemp\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n"
        )
        .into_bytes();
        body.extend_from_slice(png);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let resp = self.check(
            self.agent.post(format!("{}/upload/image", self.base)).header("Content-Type", format!("multipart/form-data; boundary={boundary}")).send(&body),
        )?;
        let reply: Value = serde_json::from_slice(&read_limited(resp, 4096)?).map_err(|_| "ComfyUI returned an invalid upload reply".to_string())?;
        let name = reply.get("name").and_then(Value::as_str).ok_or("ComfyUI did not name the uploaded image")?;
        if reply.get("type").and_then(Value::as_str) != Some("temp") || !safe_name(name) {
            return Err("ComfyUI stored the upload somewhere unexpected".into());
        }
        let sub = reply.get("subfolder").and_then(Value::as_str).unwrap_or("");
        Ok(if sub.is_empty() { format!("{name} [temp]") } else { format!("{sub}/{name} [temp]") })
    }

    /// Wait for queued prompt `id` to finish; the output image's `filename`/`subfolder`/`type`.
    fn wait(&self, id: &str, job: &JobCtx, progress: impl Fn(f32)) -> Result<Value, String> {
        let start = Instant::now();
        loop {
            if job.cancelled() {
                return Err("cancelled".into());
            }
            if start.elapsed() > VARIATION_TIMEOUT {
                return Err("ComfyUI took too long; check its window for errors".into());
            }
            let history = self.get_json(&format!("/history/{id}"))?;
            if let Some(entry) = history.get(id) {
                let status = entry.pointer("/status/status_str").and_then(Value::as_str).unwrap_or("");
                if status == "error" {
                    return Err(execution_error(entry));
                }
                if let Some(image) = first_image(entry) {
                    return Ok(image);
                }
                if entry.pointer("/status/completed").and_then(Value::as_bool) == Some(true) {
                    return Err("ComfyUI finished without an image".into());
                }
            }
            // Still queued or running: an eased guess, so the bar moves without a websocket.
            let t = start.elapsed().as_secs_f32();
            progress((t / (t + 20.0)).min(0.95));
            std::thread::sleep(POLL);
        }
    }

    fn fetch(&self, image: &Value, w: u32, h: u32) -> Result<FillImage, String> {
        let s = |k: &str| image.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let (name, sub, kind) = (s("filename"), s("subfolder"), s("type"));
        if !safe_name(&name) || sub.contains("..") || !matches!(kind.as_str(), "temp" | "output") {
            return Err("ComfyUI returned an unexpected image location".into());
        }
        let url = format!("{}/view?filename={}&subfolder={}&type={}", self.base, encode(&name), encode(&sub), encode(&kind));
        let resp = self.check(self.agent.get(url).call())?;
        let (iw, ih, rgba) = decode_png(&read_limited(resp, MAX_IMAGE)?)?;
        if (iw, ih) != (w, h) {
            return Err(format!("ComfyUI returned {iw}×{ih}; expected {w}×{h}"));
        }
        Ok(FillImage { width: iw, height: ih, rgba })
    }

    /// Drop queued prompts and stop the one running (after a cancel or an error).
    fn forget(&self, ids: &[String]) {
        let _ = self.post("/queue", &json!({"delete": ids}));
        let _ = self.post("/interrupt", &json!({}));
    }
}

fn find_models(c: &ComfyFill) -> Result<Models, String> {
    let cached = c.probe.lock().unwrap_or_else(PoisonError::into_inner).result.clone();
    match cached {
        Ok(m) => Ok(m),
        // Probe again on the worker: ComfyUI may have started since the last check.
        Err(_) => c.probe_now(),
    }
}

/// Choose the model files from what ComfyUI lists; `env` names overrides.
fn pick_models(unets: &[String], clips: &[String], vaes: &[String], env: impl Fn(&str) -> Option<String>) -> Result<Models, String> {
    fn chosen(list: &[String], over: Option<String>, prefer: &[&dyn Fn(&str) -> bool]) -> Option<String> {
        if let Some(o) = over.filter(|o| !o.trim().is_empty()) {
            return list.iter().find(|n| n.eq_ignore_ascii_case(o.trim())).cloned();
        }
        prefer.iter().find_map(|p| list.iter().find(|n| p(&n.to_ascii_lowercase())).cloned())
    }
    let unet = chosen(
        unets,
        env("PHOTOCRAFT_GF_UNET"),
        &[&|n: &str| n.contains("flux") && n.contains("fill") && !n.contains("onereward"), &|n: &str| n.contains("fill")],
    )
    .ok_or("ComfyUI has no FLUX.1 Fill model; put one in models/diffusion_models (or set PHOTOCRAFT_GF_UNET)")?;
    let clip_l = chosen(clips, env("PHOTOCRAFT_GF_CLIP_L"), &[&|n: &str| n.starts_with("clip_l"), &|n: &str| n.contains("clip_l")])
        .ok_or("ComfyUI has no CLIP-L text encoder (clip_l.safetensors) for FLUX")?;
    let t5 = chosen(
        clips,
        env("PHOTOCRAFT_GF_T5"),
        &[&|n: &str| n.contains("t5xxl") && n.contains("fp16"), &|n: &str| n.contains("t5xxl"), &|n: &str| n.contains("t5")],
    )
    .ok_or("ComfyUI has no T5-XXL text encoder for FLUX")?;
    let vae = chosen(
        vaes,
        env("PHOTOCRAFT_GF_VAE"),
        &[&|n: &str| n == "ae.safetensors" || n == "ae.sft", &|n: &str| n.starts_with("ae."), &|n: &str| n.contains("flux") && n.contains("vae")],
    )
    .ok_or("ComfyUI has no FLUX VAE (ae.safetensors)")?;
    Ok(Models { unet, clip_l, t5, vae })
}

/// The size the model works at: about [`MODEL_PIXELS`], at most [`MAX_UPSCALE`]× up, multiples
/// of 16, at least 256 a side.
fn model_size(w: u32, h: u32) -> (u32, u32) {
    let area = f64::from(w.max(1)) * f64::from(h.max(1));
    let scale = (MODEL_PIXELS / area).sqrt().min(MAX_UPSCALE);
    let side = |v: u32| -> u32 {
        let s = (f64::from(v.max(1)) * scale / 16.0).round() * 16.0;
        (s as u32).clamp(256, 4096)
    };
    (side(w), side(h))
}

/// FLUX.1 Fill as a ComfyUI API graph, core nodes only.
fn flux_fill_graph(m: &Models, image: &str, mask: &str, prompt: &str, seed: u64, (mw, mh): (u32, u32), (w, h): (u32, u32)) -> Value {
    json!({
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m.unet, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": m.clip_l, "clip_name2": m.t5, "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m.vae}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image}},
        "5": {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}},
        "6": {"class_type": "ImageScale", "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "width": mw, "height": mh, "crop": "disabled"}},
        "7": {"class_type": "MaskToImage", "inputs": {"mask": ["5", 0]}},
        "8": {"class_type": "ImageScale", "inputs": {"image": ["7", 0], "upscale_method": "bilinear", "width": mw, "height": mh, "crop": "disabled"}},
        "9": {"class_type": "ImageToMask", "inputs": {"image": ["8", 0], "channel": "red"}},
        "10": {"class_type": "GrowMask", "inputs": {"mask": ["9", 0], "expand": GROW_MASK, "tapered_corners": true}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "12": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["11", 0], "guidance": GUIDANCE}},
        "13": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["11", 0]}},
        "14": {"class_type": "InpaintModelConditioning", "inputs": {"positive": ["12", 0], "negative": ["13", 0], "vae": ["3", 0], "pixels": ["6", 0], "mask": ["10", 0], "noise_mask": true}},
        "15": {"class_type": "DifferentialDiffusion", "inputs": {"model": ["1", 0]}},
        "16": {"class_type": "KSampler", "inputs": {"model": ["15", 0], "seed": seed % (1u64 << 53), "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal", "positive": ["14", 0], "negative": ["14", 1], "latent_image": ["14", 2], "denoise": 1.0}},
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["3", 0]}},
        "18": {"class_type": "ImageScale", "inputs": {"image": ["17", 0], "upscale_method": "lanczos", "width": w, "height": h, "crop": "disabled"}},
        "19": {"class_type": "PreviewImage", "inputs": {"images": ["18", 0]}}
    })
}

fn first_image(entry: &Value) -> Option<Value> {
    entry.get("outputs")?.as_object()?.values().find_map(|o| o.get("images")?.as_array()?.first().cloned())
}

fn execution_error(entry: &Value) -> String {
    let msg = entry.pointer("/status/messages").and_then(Value::as_array).and_then(|m| {
        m.iter().find_map(|e| {
            let pair = e.as_array()?;
            (pair.first()?.as_str()? == "execution_error").then(|| {
                let d = pair.get(1)?;
                let node = d.get("node_type").and_then(Value::as_str).unwrap_or("a node");
                let text = d.get("exception_message").and_then(Value::as_str).unwrap_or("failed");
                Some(format!("{node}: {}", text.trim()))
            })?
        })
    });
    msg.map_or_else(|| "ComfyUI reported an error".into(), |m| format!("ComfyUI reported an error in {m}"))
}

/// The useful part of a refused `/prompt`: ComfyUI's `error.message` and the first node error.
fn queue_error(reply: &Value) -> String {
    let main = reply.pointer("/error/message").and_then(Value::as_str).unwrap_or("ComfyUI refused the graph");
    let node = reply.get("node_errors").and_then(Value::as_object).and_then(|n| {
        n.values().find_map(|v| {
            let e = v.get("errors")?.as_array()?.first()?;
            Some(format!("{} ({})", e.get("message")?.as_str()?, e.get("details").and_then(Value::as_str).unwrap_or("")))
        })
    });
    match node {
        Some(n) => format!("{main}: {n}"),
        None => main.to_string(),
    }
}

fn read_limited(resp: ureq::http::Response<ureq::Body>, max: u64) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    resp.into_body().into_reader().take(max + 1).read_to_end(&mut bytes).map_err(|_| "the ComfyUI reply broke off".to_string())?;
    if bytes.len() as u64 > max {
        return Err("the ComfyUI reply is too large".into());
    }
    Ok(bytes)
}

fn encode_png(w: u32, h: u32, color: png::ColorType, data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(color);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_compression(png::Compression::Fast);
        let mut writer = enc.write_header().map_err(|e| format!("could not encode PNG: {e}"))?;
        writer.write_image_data(data).map_err(|e| format!("could not encode PNG: {e}"))?;
    }
    Ok(out)
}

/// Any 8-bit (or normalised) PNG to straight RGBA bytes.
fn decode_png(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = dec.read_info().map_err(|e| format!("ComfyUI returned an unreadable image: {e}"))?;
    let size = reader.output_buffer_size().ok_or("ComfyUI returned an image too large to read")?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("ComfyUI returned an unreadable image: {e}"))?;
    buf.truncate(info.buffer_size());
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => buf.as_chunks::<3>().0.iter().flat_map(|[r, g, b]| [*r, *g, *b, 255]).collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|g| [*g, *g, *g, 255]).collect(),
        png::ColorType::GrayscaleAlpha => buf.as_chunks::<2>().0.iter().flat_map(|[g, a]| [*g, *g, *g, *a]).collect(),
        png::ColorType::Indexed => return Err("ComfyUI returned a palette image".into()),
    };
    Ok((info.width, info.height, rgba))
}

/// ComfyUI file names we accept back: no path parts.
fn safe_name(name: &str) -> bool {
    !name.is_empty() && name.len() < 256 && !name.contains(['/', '\\', '\0']) && !name.contains("..")
}

fn encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn nonce() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or_default();
    t ^ NEXT.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn picks_flux_fill_and_its_encoders() {
        let unets = s(&[
            "flux1-dev.safetensors",
            "flux.1-fill-dev-OneReward-transformer_fp8.safetensors",
            "black-forest-labs_FLUX.1-Fill-dev_flux1-fill-dev_fp8.safetensors",
        ]);
        let clips = s(&["clip_g.safetensors", "clip_l.safetensors", "t5xxl_fp8_e4m3fn.safetensors", "t5xxl_fp16.safetensors"]);
        let vaes = s(&["wan_2.1_vae.safetensors", "ae.safetensors", "ae.sft"]);
        let m = pick_models(&unets, &clips, &vaes, |_| None).unwrap();
        assert_eq!(m.unet, "black-forest-labs_FLUX.1-Fill-dev_flux1-fill-dev_fp8.safetensors");
        assert_eq!(m.clip_l, "clip_l.safetensors");
        assert_eq!(m.t5, "t5xxl_fp16.safetensors");
        assert_eq!(m.vae, "ae.safetensors");
        // An override picks another listed file, case-insensitively.
        let m = pick_models(&unets, &clips, &vaes, |k| (k == "PHOTOCRAFT_GF_UNET").then(|| "FLUX.1-FILL-DEV-ONEREWARD-TRANSFORMER_FP8.SAFETENSORS".into()))
            .unwrap();
        assert!(m.unet.contains("OneReward"));
        // Missing pieces say which.
        assert!(pick_models(&s(&["flux1-dev.safetensors"]), &clips, &vaes, |_| None).unwrap_err().contains("FLUX.1 Fill"));
        assert!(pick_models(&unets, &s(&["clip_l.safetensors"]), &vaes, |_| None).unwrap_err().contains("T5"));
    }

    #[test]
    fn model_size_targets_a_megapixel_in_sixteens() {
        let (w, h) = model_size(4000, 3000);
        assert_eq!((w % 16, h % 16), (0, 0));
        let mp = f64::from(w) * f64::from(h);
        assert!((0.9e6..1.2e6).contains(&mp), "{w}x{h}");
        // Small windows scale up at most 4×.
        assert_eq!(model_size(100, 100), (400, 400));
        // Never below 256 a side.
        assert_eq!(model_size(2000, 20).1, 256);
    }

    #[test]
    fn png_round_trip() {
        let rgba: Vec<u8> = (0..4 * 3 * 2).map(|i| (i * 10) as u8).collect();
        let png = encode_png(3, 2, png::ColorType::Rgba, &rgba).unwrap();
        assert_eq!(decode_png(&png).unwrap(), (3, 2, rgba));
        let gray = vec![0u8, 255, 128, 7];
        let (w, h, out) = decode_png(&encode_png(2, 2, png::ColorType::Grayscale, &gray).unwrap()).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(&out[..8], &[0, 0, 0, 255, 255, 255, 255, 255]);
    }

    #[test]
    fn graph_wires_seed_prompt_and_sizes() {
        let m = Models { unet: "u".into(), clip_l: "c".into(), t5: "t".into(), vae: "v".into() };
        let g = flux_fill_graph(&m, "img.png [temp]", "mask.png [temp]", "a red barn", 42, (1024, 768), (2000, 1500));
        assert_eq!(g["16"]["inputs"]["seed"], json!(42));
        assert_eq!(g["11"]["inputs"]["text"], json!("a red barn"));
        assert_eq!(g["6"]["inputs"]["width"], json!(1024));
        assert_eq!(g["18"]["inputs"]["width"], json!(2000));
        assert_eq!(g["18"]["inputs"]["height"], json!(1500));
        assert_eq!(g["4"]["inputs"]["image"], json!("img.png [temp]"));
        // Every link points at a node in the graph.
        for node in g.as_object().unwrap().values() {
            for v in node["inputs"].as_object().unwrap().values() {
                if let Some([from, _]) = v.as_array().map(Vec::as_slice).and_then(|a| <&[Value; 2]>::try_from(a).ok())
                    && let Some(id) = from.as_str()
                {
                    assert!(g.get(id).is_some(), "dangling link to {id}");
                }
            }
        }
    }

    #[test]
    fn errors_name_the_failing_node() {
        let entry = json!({"status": {"status_str": "error", "messages": [["execution_start", {}], ["execution_error", {"node_type": "UNETLoader", "exception_message": "out of memory "}]]}});
        assert_eq!(execution_error(&entry), "ComfyUI reported an error in UNETLoader: out of memory");
        let refused = json!({"error": {"message": "Prompt outputs failed validation"}, "node_errors": {"1": {"errors": [{"message": "Value not in list", "details": "unet_name: 'x' not in []"}]}}});
        assert_eq!(queue_error(&refused), "Prompt outputs failed validation: Value not in list (unet_name: 'x' not in [])");
    }

    #[test]
    fn names_and_urls_are_safe() {
        assert!(safe_name("ComfyUI_temp_abc_00001_.png"));
        for bad in ["", "../x.png", "a/b.png", "a\\b.png"] {
            assert!(!safe_name(bad), "{bad}");
        }
        assert_eq!(encode("a b&c=d/é"), "a%20b%26c%3Dd%2F%C3%A9");
    }
}
