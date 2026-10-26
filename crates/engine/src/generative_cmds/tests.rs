use std::sync::{Arc, Mutex};

use serde_json::json;

use super::*;
use crate::EngineError;

/// A stand-in image model: fills every pixel with one colour per variation and records what it
/// was asked.
#[derive(Default)]
struct Fake {
    colors: Vec<[u8; 4]>,
    seen: Mutex<Vec<FillRequest>>,
    /// Return this instead of images.
    fail: Option<String>,
    /// Return images this much wider than the window.
    grow: u32,
    unavailable: Option<String>,
}

impl Fake {
    fn with(colors: &[[u8; 4]]) -> Arc<Self> {
        Arc::new(Self { colors: colors.to_vec(), ..Self::default() })
    }
    fn last(&self) -> FillRequest {
        self.seen.lock().unwrap().last().cloned().unwrap()
    }
}

impl GenerativeBackend for Fake {
    fn name(&self) -> String {
        "Fake model".into()
    }
    fn unavailable(&self) -> Option<String> {
        self.unavailable.clone()
    }
    fn fill(&self, req: &FillRequest, job: &JobCtx) -> std::result::Result<Vec<FillImage>, String> {
        self.seen.lock().unwrap().push(req.clone());
        job.progress(0.5, "faking");
        if let Some(e) = &self.fail {
            return Err(e.clone());
        }
        let (w, h) = (req.width + self.grow, req.height);
        Ok(self.colors.iter().map(|c| FillImage { width: w, height: h, rgba: c.iter().copied().cycle().take(w as usize * h as usize * 4).collect() }).collect())
    }
}

const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];

/// A 64×48 white document with a 12×14 selection at (26, 16).
fn session(mode: &str) -> Session {
    let mut s = Session::default();
    s.execute("file.new", json!({"width": 64, "height": 48, "mode": mode, "background": "white"})).unwrap();
    s.execute("select.rect", json!({"x": 26, "y": 16, "width": 12, "height": 14})).unwrap();
    s
}

fn pixel(s: &Session, x: i32, y: i32) -> [f32; 4] {
    let doc = &s.active().unwrap().doc;
    photocraft_compose::render(doc, Rect::new(x, y, x + 1, y + 1)).px[0]
}

fn layers(s: &Session) -> usize {
    s.active().unwrap().doc.layer_count()
}

fn near(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 0.02)
}

#[test]
fn disabled_without_a_backend_with_a_reason() {
    let mut s = session("rgb");
    let why = s.disabled_reason(CMD).unwrap();
    assert!(why.contains("image model"), "{why}");
    assert!(matches!(s.execute(CMD, json!({"prompt": "a boat"})), Err(EngineError::Disabled(..))));
    let st = s.execute("edit.generativeFill.status", json!({})).unwrap();
    assert_eq!(st["supported"], json!(false));
    assert_eq!(layers(&s), 1);
}

#[test]
fn disabled_without_a_selection_or_when_the_backend_says_so() {
    let mut s = session("rgb");
    s.generative.backend = Some(Fake::with(&[RED]));
    s.execute("select.deselect", json!({})).unwrap();
    assert!(s.disabled_reason(CMD).unwrap().contains("select"));

    let mut s = session("rgb");
    s.generative.backend = Some(Arc::new(Fake { unavailable: Some("ComfyUI is not running".into()), ..Fake::default() }));
    assert_eq!(s.disabled_reason(CMD).unwrap(), "ComfyUI is not running");
}

#[test]
fn fills_the_selection_on_a_new_masked_layer() {
    let mut s = session("rgb");
    let fake = Fake::with(&[RED, GREEN, BLUE]);
    s.generative.backend = Some(fake.clone());
    let out = s.execute(CMD, json!({"prompt": "  a red barn  ", "variations": 3, "seed": 7})).unwrap();
    assert_eq!(out["variations"], json!(3));
    assert_eq!(out["shown"], json!(0));
    assert_eq!(out["prompt"], json!("a red barn"));

    // A new layer above, named after the prompt, active, with a layer mask; the selection is gone.
    assert_eq!(layers(&s), 2);
    let d = s.active().unwrap();
    let layer = d.doc.layer(LayerId(out["layer"].as_u64().unwrap())).unwrap();
    assert_eq!(layer.name, "a red barn");
    assert!(layer.mask.is_some());
    assert_eq!(d.active_layer, Some(layer.id));
    assert!(d.doc.selection.is_none());

    // The first variation shows inside the selection; outside, the mask keeps the white.
    assert!(near(pixel(&s, 30, 20), [1.0, 0.0, 0.0, 1.0]), "{:?}", pixel(&s, 30, 20));
    assert!(near(pixel(&s, 26, 16), [1.0, 0.0, 0.0, 1.0]));
    assert!(near(pixel(&s, 37, 29), [1.0, 0.0, 0.0, 1.0]));
    assert!(near(pixel(&s, 25, 20), [1.0, 1.0, 1.0, 1.0]));
    assert!(near(pixel(&s, 38, 20), [1.0, 1.0, 1.0, 1.0]));
    assert!(near(pixel(&s, 2, 2), [1.0, 1.0, 1.0, 1.0]));

    // One undo step takes it all back.
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(layers(&s), 1);
    assert!(near(pixel(&s, 30, 20), [1.0, 1.0, 1.0, 1.0]));
}

#[test]
fn sends_the_window_the_mask_and_the_prompt() {
    let mut s = session("rgb");
    let fake = Fake::with(&[RED]);
    s.generative.backend = Some(fake.clone());
    s.execute(CMD, json!({"prompt": "a cat", "variations": 1, "seed": 42})).unwrap();
    let r = fake.last();
    // 12×14 selection, at least 64 px of context, clipped to the 64×48 canvas: the whole canvas.
    assert_eq!((r.width, r.height), (64, 48));
    assert_eq!(r.prompt, "a cat");
    assert_eq!((r.variations, r.seed), (1, 42));
    assert_eq!(r.image.len(), 64 * 48 * 4);
    assert!(r.image.chunks(4).all(|p| p == [255, 255, 255, 255]), "the visible image is white");
    assert_eq!(r.mask.len(), 64 * 48);
    let at = |x: usize, y: usize| r.mask[y * 64 + x];
    assert_eq!((at(26, 16), at(37, 29), at(25, 16), at(38, 16), at(26, 30)), (255, 255, 0, 0, 0));
    assert_eq!(r.mask.iter().filter(|m| **m == 255).count(), 12 * 14);
}

#[test]
fn context_window_adds_margin_and_clips() {
    let canvas = Rect::new(0, 0, 2000, 1000);
    // Half the longer side as margin.
    assert_eq!(context_window(Rect::new(800, 400, 1000, 500), canvas), Rect::new(700, 300, 1100, 600));
    // At least 64 px.
    assert_eq!(context_window(Rect::new(500, 500, 510, 510), canvas), Rect::new(436, 436, 574, 574));
    // Clipped to the canvas, and a selection past the edge clipped first.
    assert_eq!(context_window(Rect::new(-50, -50, 40, 40), canvas), Rect::new(0, 0, 104, 104));
}

#[test]
fn variations_switch_as_undoable_steps() {
    let mut s = session("rgb");
    s.generative.backend = Some(Fake::with(&[RED, GREEN, BLUE]));
    s.execute(CMD, json!({"prompt": "x"})).unwrap();
    assert!(s.disabled_reason(VARIATION_CMD).is_none());

    s.execute(VARIATION_CMD, json!({"index": 2})).unwrap();
    assert!(near(pixel(&s, 30, 20), [0.0, 0.0, 1.0, 1.0]));
    assert!(near(pixel(&s, 2, 2), [1.0, 1.0, 1.0, 1.0]), "the mask still cuts");
    let st = s.execute("edit.generativeFill.status", json!({})).unwrap();
    assert_eq!(st["variations"]["count"], json!(3));
    assert_eq!(st["variations"]["shown"], json!(2));

    s.execute(VARIATION_CMD, json!({"index": 1})).unwrap();
    assert!(near(pixel(&s, 30, 20), [0.0, 1.0, 0.0, 1.0]));
    s.execute("edit.undo", json!({})).unwrap();
    assert!(near(pixel(&s, 30, 20), [0.0, 0.0, 1.0, 1.0]));

    assert!(s.execute(VARIATION_CMD, json!({"index": 3})).is_err());
    assert!(s.execute(VARIATION_CMD, json!({})).is_err());
    assert!(s.execute(VARIATION_CMD, json!({"index": -1})).is_err());
}

#[test]
fn variations_need_a_generative_layer() {
    let s = session("rgb");
    assert!(s.disabled_reason(VARIATION_CMD).unwrap().contains("no Generative Fill"));
}

#[test]
fn a_failing_or_misbehaving_backend_leaves_the_document_alone() {
    for fake in [
        Arc::new(Fake { fail: Some("out of memory".into()), ..Fake::default() }),
        Arc::new(Fake { colors: vec![RED], grow: 3, ..Fake::default() }),
        Arc::new(Fake { colors: vec![], ..Fake::default() }),
        Arc::new(Fake { colors: vec![RED, GREEN, BLUE], ..Fake::default() }),
    ] {
        let mut s = session("rgb");
        s.generative.backend = Some(fake);
        let err = s.execute(CMD, json!({"prompt": "x", "variations": 2})).unwrap_err().to_string();
        assert!(err.contains("Fake model"), "{err}");
        assert_eq!(layers(&s), 1, "{err}");
        assert!(s.active().unwrap().doc.selection.is_some(), "{err}");
    }
}

#[test]
fn rejects_bad_params_before_calling_the_backend() {
    let mut s = session("rgb");
    let fake = Fake::with(&[RED]);
    s.generative.backend = Some(fake.clone());
    let long = "x".repeat(MAX_PROMPT + 1);
    for p in [
        json!({"variations": 0}),
        json!({"variations": 5}),
        json!({"variations": "3"}),
        json!({"variations": 1.5}),
        json!({"seed": -1}),
        json!({"seed": "7"}),
        json!({"prompt": 3}),
        json!({"prompt": long}),
        json!([1, 2]),
    ] {
        assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
    }
    assert!(fake.seen.lock().unwrap().is_empty());
    assert_eq!(layers(&s), 1);
}

#[test]
fn works_in_cmyk_and_grayscale() {
    for mode in ["cmyk", "grayscale"] {
        let mut s = session(mode);
        s.generative.backend = Some(Fake::with(&[[0, 0, 0, 255]]));
        s.execute(CMD, json!({"prompt": "shadow", "variations": 1})).unwrap();
        assert_eq!(layers(&s), 2, "{mode}");
        let inside = pixel(&s, 30, 20);
        let outside = pixel(&s, 2, 2);
        assert!(inside[0] < 0.1 && inside[1] < 0.1 && inside[2] < 0.1, "{mode}: {inside:?}");
        assert!(outside[0] > 0.9, "{mode}: {outside:?}");
    }
}

#[test]
fn an_empty_prompt_is_allowed_and_names_the_layer_generically() {
    let mut s = session("rgb");
    s.generative.backend = Some(Fake::with(&[RED]));
    let out = s.execute(CMD, json!({})).unwrap();
    let d = s.active().unwrap();
    let name = &d.doc.layer(LayerId(out["layer"].as_u64().unwrap())).unwrap().name;
    assert!(name.starts_with("Generative Fill"), "{name}");
    assert_eq!(out["prompt"], json!(""));
}
