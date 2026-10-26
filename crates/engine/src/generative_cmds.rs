//! Edit › Generative Fill: replace the selection with new content an image model makes from a
//! prompt, on a new layer masked to the selection (Photoshop's Generative Fill; gap G15).
//!
//! The engine runs no model and opens no connection, as for the release check
//! ([`crate::updates_cmds`]). The desktop app injects a [`GenerativeBackend`] (a local ComfyUI
//! through craft-ai-integrations, for instance). Without one, as in the CLI, tests and the
//! browser, the command is disabled with a reason anyone can act on.
//!
//! The backend runs on a job worker ([`crate::jobs`]): the document is read from a snapshot and
//! never touched off the UI thread, and the result lands as one undo step. Photoshop keeps every
//! variation on its Generative layer; here the variations stay in [`Generative`] and
//! `edit.generativeFill.variation` shows another one, also as one undo step.
//!
//! What is sent to the backend: the visible image under a window around the selection (the
//! selection plus context on every side, clipped to the canvas) as 8-bit straight-alpha RGBA,
//! the selection as an 8-bit mask, the prompt, how many variations and a seed. Nothing else
//! about the document leaves the engine.

use std::sync::Arc;

use photocraft_color::PixelFormat;
use photocraft_doc::{DocId, Document, Layer, LayerId, LayerMask};
use photocraft_geom::Rect;
use photocraft_raster::from_rgba_into;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always};
use crate::jobs::JobCtx;
use crate::vector_cmds::bad;
use crate::{EngineError, Result, Session};

const CMD: &str = "edit.generativeFill";
const VARIATION_CMD: &str = "edit.generativeFill.variation";

/// Most variations one Generative Fill asks for (Photoshop makes three).
pub const MAX_VARIATIONS: u32 = 4;
/// Longest prompt, in characters.
pub const MAX_PROMPT: usize = 2000;
/// Most pixels in the window sent to a backend. Models work near a megapixel and the backend
/// scales the window down for them, but the window is read and held at full size here.
pub const MAX_WINDOW_PIXELS: u64 = 64 * 1024 * 1024;
/// Generative Fill sets kept for `edit.generativeFill.variation`, oldest dropped first.
const MAX_SETS: usize = 16;

/// What a backend is asked to fill.
#[derive(Clone, Debug, PartialEq)]
pub struct FillRequest {
    pub width: u32,
    pub height: u32,
    /// The visible image under the window: straight-alpha RGBA, 8 bits a channel, row-major.
    pub image: Vec<u8>,
    /// What to replace: 255 where the selection reaches, 0 elsewhere; row-major, one byte a pixel.
    pub mask: Vec<u8>,
    /// What should be there. Empty: fill to match the surroundings.
    pub prompt: String,
    /// How many different results to make (`1..=MAX_VARIATIONS`).
    pub variations: u32,
    pub seed: u64,
}

/// One result: the whole window, the same size as the request's, straight-alpha RGBA 8-bit.
#[derive(Clone, Debug, PartialEq)]
pub struct FillImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// An image model that can fill a masked area. Injected by the desktop app.
pub trait GenerativeBackend: Send + Sync {
    /// A short name for messages, e.g. "ComfyUI on this computer".
    fn name(&self) -> String;
    /// Why the backend cannot run right now (not installed, not running, no model), if it can't.
    /// Checked before the command is offered; keep it cheap.
    fn unavailable(&self) -> Option<String> {
        None
    }
    /// Make up to `req.variations` results. Blocking: it is called on a job worker. Report
    /// progress and stop early through `job` ([`JobCtx::progress`], [`JobCtx::cancelled`]).
    fn fill(&self, req: &FillRequest, job: &JobCtx) -> std::result::Result<Vec<FillImage>, String>;
}

/// The variations one Generative Fill made, and which one its layer shows.
#[derive(Clone, Debug)]
struct VariationSet {
    doc: DocId,
    layer: LayerId,
    window: Rect,
    prompt: String,
    images: Vec<FillImage>,
    shown: usize,
}

/// Generative Fill state: the injected backend and the variations made so far.
#[derive(Default)]
pub struct Generative {
    pub backend: Option<Arc<dyn GenerativeBackend>>,
    sets: Vec<VariationSet>,
}

/// What the Properties panel shows for a layer Generative Fill made.
#[derive(Clone, Debug, PartialEq)]
pub struct VariationInfo {
    pub count: usize,
    /// Which variation the layer shows (`0..count`).
    pub shown: usize,
    pub prompt: String,
}

impl Generative {
    /// The variations of `layer` in document `doc`, if Generative Fill made it this session.
    pub fn variations(&self, doc: DocId, layer: LayerId) -> Option<VariationInfo> {
        self.set_for(doc, layer).map(|v| VariationInfo { count: v.images.len(), shown: v.shown, prompt: v.prompt.clone() })
    }

    fn set_for(&self, doc: DocId, layer: LayerId) -> Option<&VariationSet> {
        self.sets.iter().rev().find(|v| v.doc == doc && v.layer == layer)
    }

    fn set_for_mut(&mut self, doc: DocId, layer: LayerId) -> Option<&mut VariationSet> {
        self.sets.iter_mut().rev().find(|v| v.doc == doc && v.layer == layer)
    }

    fn remember(&mut self, set: VariationSet) {
        self.sets.retain(|v| !(v.doc == set.doc && v.layer == set.layer));
        self.sets.push(set);
        let extra = self.sets.len().saturating_sub(MAX_SETS);
        if extra > 0 {
            self.sets.drain(..extra);
        }
    }
}

impl std::fmt::Debug for Generative {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Generative").field("backend", &self.backend.as_ref().map(|b| b.name())).field("sets", &self.sets.len()).finish()
    }
}

/// The request's parameters, checked before anything is read.
struct Params {
    prompt: String,
    variations: u32,
    seed: u64,
}

fn params(p: &Value) -> Result<Params> {
    if !p.is_object() && !p.is_null() {
        return Err(bad(CMD, "params must be an object"));
    }
    let prompt = match p.get("prompt") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.trim().to_string(),
        Some(_) => return Err(bad(CMD, "`prompt` must be a string")),
    };
    if prompt.chars().count() > MAX_PROMPT {
        return Err(bad(CMD, format!("`prompt` is longer than {MAX_PROMPT} characters")));
    }
    let variations = match p.get("variations") {
        None | Some(Value::Null) => 3,
        Some(v) => v
            .as_u64()
            .filter(|n| (1..=u64::from(MAX_VARIATIONS)).contains(n))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| bad(CMD, format!("`variations` must be 1..{MAX_VARIATIONS}")))?,
    };
    let seed = match p.get("seed") {
        None | Some(Value::Null) => 0,
        Some(v) => v.as_u64().ok_or_else(|| bad(CMD, "`seed` must be a non-negative integer"))?,
    };
    Ok(Params { prompt, variations, seed })
}

/// Why Generative Fill can't run, if it can't: no backend, the backend's own reason, no
/// selection.
fn ready(s: &Session) -> std::result::Result<Arc<dyn GenerativeBackend>, String> {
    let backend = s
        .generative
        .backend
        .clone()
        .ok_or_else(|| "Generative Fill needs an image model; set one up in the desktop app (local ComfyUI or a cloud service)".to_string())?;
    if let Some(why) = backend.unavailable() {
        return Err(why);
    }
    let d = s.active().ok_or("no document open")?;
    if d.doc.selection.as_ref().is_none_or(|m| m.content_bounds().intersect(&d.doc.bounds()).is_empty()) {
        return Err("select the area to fill first".into());
    }
    Ok(backend)
}

fn can_fill(s: &Session) -> std::result::Result<(), String> {
    ready(s).map(|_| ())
}

/// The window sent to the backend: the selection's bounds with context on every side (half the
/// selection's longer side, at least 64 pixels), clipped to the canvas.
pub fn context_window(selection: Rect, canvas: Rect) -> Rect {
    let sel = selection.intersect(&canvas);
    let side = sel.width().max(sel.height());
    let margin = i32::try_from((side / 2).max(64)).unwrap_or(i32::MAX / 4);
    sel.inflate(margin).intersect(&canvas)
}

/// Everything a fill's worker needs, read from the snapshot before the job starts.
struct Plan {
    doc: Arc<Document>,
    window: Rect,
    /// The selection's coverage over the window, `0..=1`, row-major.
    cover: Vec<f32>,
    request: FillRequest,
}

fn plan(s: &Session, p: &Params) -> Result<Plan> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = st.doc.clone();
    let sel = doc.selection.as_ref().ok_or_else(|| bad(CMD, "select the area to fill first"))?;
    let canvas = doc.bounds();
    let sb = sel.content_bounds().intersect(&canvas);
    if sb.is_empty() {
        return Err(bad(CMD, "the selection is outside the canvas"));
    }
    let window = context_window(sb, canvas);
    let (w, h) = (window.width(), window.height());
    if u64::from(w) * u64::from(h) > MAX_WINDOW_PIXELS {
        return Err(bad(CMD, "the selection is too large for Generative Fill; select a smaller area"));
    }
    let cover = crate::fill_cmds::read_window_mask(sel, window, &|| false).ok_or(EngineError::Cancelled)?;
    let mask: Vec<u8> = cover.iter().map(|c| if *c > 0.0 { 255 } else { 0 }).collect();
    if !mask.iter().any(|m| *m > 0) {
        return Err(bad(CMD, "the selection is empty"));
    }
    let request = FillRequest { width: w, height: h, image: Vec::new(), mask, prompt: p.prompt.clone(), variations: p.variations, seed: p.seed };
    Ok(Plan { doc, window, cover, request })
}

/// `v` in `0..=1` to a byte, rounding; NaN reads as 0.
fn to_byte(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The visible image under `window` as straight-alpha RGBA bytes.
fn read_image(doc: &Document, window: Rect) -> Vec<u8> {
    let buf = photocraft_compose::render(doc, window);
    let mut out = Vec::with_capacity(buf.px.len().saturating_mul(4));
    for px in &buf.px {
        out.extend(px.iter().map(|v| to_byte(*v)));
    }
    out
}

/// Check what a backend returned against what was asked: a count in `1..=variations`, each the
/// window's size, each buffer exactly `width × height × 4` bytes.
fn check_results(req: &FillRequest, images: &[FillImage], backend: &str) -> Result<()> {
    if images.is_empty() {
        return Err(bad(CMD, format!("{backend} returned no image")));
    }
    if images.len() > req.variations as usize {
        return Err(bad(CMD, format!("{backend} returned {} images; {} were asked for", images.len(), req.variations)));
    }
    let want = (req.width as usize).checked_mul(req.height as usize).and_then(|n| n.checked_mul(4)).ok_or_else(|| bad(CMD, "window too large"))?;
    for (i, im) in images.iter().enumerate() {
        if im.width != req.width || im.height != req.height {
            return Err(bad(CMD, format!("{backend} returned image {} at {}×{}; the window is {}×{}", i + 1, im.width, im.height, req.width, req.height)));
        }
        if im.rgba.len() != want {
            return Err(bad(CMD, format!("{backend} returned image {} with {} bytes; expected {want}", i + 1, im.rgba.len())));
        }
    }
    Ok(())
}

/// `img`'s pixels in `fmt` (opaque wherever the model drew; the layer mask does the cutting).
fn layer_pixels(img: &FillImage, fmt: &PixelFormat) -> Vec<f32> {
    let n = fmt.channels().max(1);
    let mut px = vec![0.0f32; (img.width as usize).saturating_mul(img.height as usize).saturating_mul(n)];
    let mut enc = [0.0f32; 8];
    for (out, [r, g, b, a]) in px.chunks_exact_mut(n).zip(img.rgba.as_chunks::<4>().0) {
        let rgba = [f32::from(*r) / 255.0, f32::from(*g) / 255.0, f32::from(*b) / 255.0, f32::from(*a) / 255.0];
        let k = from_rgba_into(fmt, rgba, &mut enc);
        if let (Some(dst), Some(src)) = (out.get_mut(..k), enc.get(..k)) {
            dst.copy_from_slice(src);
        }
    }
    px
}

/// The layer mask: the selection's coverage over the window, hidden everywhere else.
fn selection_mask(window: Rect, cover: &[f32]) -> LayerMask {
    let mut mask = LayerMask::hide_all();
    mask.surface.write_region(window, cover);
    mask.surface.prune();
    mask
}

/// Write `img` into `layer`'s pixels over `window` (replacing what was there).
fn show(doc: &mut Document, layer: LayerId, window: Rect, img: &FillImage) -> Result<()> {
    let l = doc.layer_mut(layer).ok_or(EngineError::NoLayer(layer))?;
    let surf = l.surface_mut().ok_or_else(|| bad(VARIATION_CMD, "the layer is not a pixel layer"))?;
    let fmt = surf.format();
    surf.write_region(window, &layer_pixels(img, &fmt));
    surf.prune();
    Ok(())
}

/// A layer name from the prompt, like Photoshop's (the prompt, or "Generative Fill").
fn layer_name(doc: &Document, prompt: &str) -> String {
    let short: String = prompt.chars().take(48).collect();
    if short.is_empty() { doc.next_layer_name("Generative Fill") } else { short }
}

fn generative_fill(s: &mut Session, p: &Value) -> Result<Value> {
    let backend = ready(s).map_err(|why| bad(CMD, why))?;
    let p = params(p)?;
    let Plan { doc, window, cover, mut request } = plan(s, &p)?;
    let doc_id = s.active().map(|d| d.doc.id).ok_or(EngineError::NoDocument)?;
    let name = backend.name();
    crate::jobs::run(
        s,
        "Generative Fill",
        true,
        move |ctx| {
            ctx.progress(0.0, "Reading the image");
            request.image = read_image(&doc, window);
            ctx.check()?;
            ctx.progress(0.05, &format!("Generating with {name}"));
            let images = backend.fill(&request, ctx).map_err(|e| bad(CMD, format!("{name}: {e}")))?;
            ctx.check()?;
            check_results(&request, &images, &name)?;
            Ok((images, request.prompt))
        },
        move |s, (images, prompt): (Vec<FillImage>, String)| {
            let first = images.first().cloned().ok_or_else(|| bad(CMD, "no image to apply"))?;
            let count = images.len();
            let layer = s.edit("Generative Fill", |doc, active| {
                let fmt = doc.pixel_format();
                let mut l = Layer::raster(layer_name(doc, &prompt), fmt);
                if let Some(surf) = l.surface_mut() {
                    surf.write_region(window, &layer_pixels(&first, &fmt));
                    surf.prune();
                }
                l.mask = Some(selection_mask(window, &cover));
                let nid = doc.insert_above(*active, l);
                *active = Some(nid);
                // A new pixel layer with a layer mask: the selection is used up.
                doc.selection = None;
                Ok(nid)
            })?;
            s.generative.remember(VariationSet { doc: doc_id, layer, window, prompt: prompt.clone(), images, shown: 0 });
            Ok(json!({"layer": layer.0, "variations": count, "shown": 0, "window": [window.x0, window.y0, window.width(), window.height()], "prompt": prompt}))
        },
    )
}

fn can_vary(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let layer = d.active_layer.ok_or("no active layer")?;
    s.generative.set_for(d.doc.id, layer).map(|_| ()).ok_or_else(|| "the active layer has no Generative Fill variations".into())
}

fn variation(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc_id = d.doc.id;
    let layer = match p.get("layer") {
        None | Some(Value::Null) => d.active_layer.ok_or_else(|| bad(VARIATION_CMD, "no active layer"))?,
        Some(v) => LayerId(v.as_u64().ok_or_else(|| bad(VARIATION_CMD, "`layer` must be a layer id"))?),
    };
    let index = p
        .get("index")
        .and_then(Value::as_u64)
        .and_then(|i| usize::try_from(i).ok())
        .ok_or_else(|| bad(VARIATION_CMD, "`index` must be a variation number from 0"))?;
    let set = s.generative.set_for(doc_id, layer).ok_or_else(|| bad(VARIATION_CMD, "that layer has no Generative Fill variations"))?;
    let img = set
        .images
        .get(index)
        .cloned()
        .ok_or_else(|| bad(VARIATION_CMD, format!("there are {} variations (0..{})", set.images.len(), set.images.len().saturating_sub(1))))?;
    let window = set.window;
    let count = set.images.len();
    s.edit("Generative Fill Variation", |doc, _| show(doc, layer, window, &img))?;
    if let Some(set) = s.generative.set_for_mut(doc_id, layer) {
        set.shown = index;
    }
    Ok(json!({"layer": layer.0, "shown": index, "variations": count}))
}

fn status(s: &mut Session, _p: &Value) -> Result<Value> {
    let backend = s.generative.backend.as_ref();
    let set = s.active().and_then(|d| d.active_layer.and_then(|l| s.generative.set_for(d.doc.id, l)));
    Ok(json!({
        "supported": backend.is_some(),
        "backend": backend.map(|b| b.name()),
        "unavailable": backend.and_then(|b| b.unavailable()),
        "variations": set.map(|v| json!({"layer": v.layer.0, "count": v.images.len(), "shown": v.shown, "prompt": v.prompt})),
    }))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: CMD,
            label: "Generative Fill…",
            menu: &["Edit"],
            shortcut: None,
            params: r#"{"prompt":string="" (what should be there; empty fills to match the surroundings),"variations":1..4=3,"seed":u64=0}"#,
            enabled: can_fill,
            run: generative_fill,
            journal: true,
        },
        CommandSpec {
            id: VARIATION_CMD,
            label: "Generative Fill Variation",
            menu: &[],
            shortcut: None,
            params: r#"{"index":0..,"layer":id?=active}"#,
            enabled: can_vary,
            run: variation,
            journal: true,
        },
        CommandSpec {
            id: "edit.generativeFill.status",
            label: "Generative Fill Status",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: status,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests;
