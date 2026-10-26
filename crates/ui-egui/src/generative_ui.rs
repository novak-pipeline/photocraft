//! Edit › Generative Fill…: the prompt dialog for `edit.generativeFill`, and the Properties
//! panel's Generative Fill section, which shows another variation of a layer it made.
//!
//! The engine stays directly callable with parameters by CLI, MCP and Actions; only a call with
//! no parameters (the menu, a context menu) opens the dialog.

use serde_json::{Map, Value, json};

use crate::PhotocraftApp;

pub const COMMAND: &str = "edit.generativeFill";
const VARIATION: &str = "edit.generativeFill.variation";
const MARK: &str = "__generativeFill";
const VARIATIONS: [&str; 4] = ["1", "2", "3", "4"];

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

/// The dialog's fields: an empty prompt and the variation count used last time (3 at first).
pub fn fields(app: &PhotocraftApp) -> Map<String, Value> {
    let mut f = Map::new();
    f.insert(MARK.into(), json!(true));
    f.insert("__command".into(), json!(COMMAND));
    f.insert("__label".into(), json!("Generative Fill…"));
    f.insert("prompt".into(), json!(""));
    let saved = app.session.prefs().dialogs.get(COMMAND).and_then(|v| v.get("variations")).and_then(Value::as_u64).filter(|n| (1..=4).contains(n));
    f.insert("variations".into(), json!(saved.unwrap_or(3)));
    f
}

pub fn open(app: &mut PhotocraftApp) -> u64 {
    let f = fields(app);
    app.ui.open_dialog(crate::state::DialogKind::Command, f)
}

/// Only the command's params go to the engine; the dialog's markers stay here.
pub fn params(f: &Map<String, Value>) -> Value {
    json!({
        "prompt": f.get("prompt").and_then(Value::as_str).unwrap_or("").trim(),
        "variations": f.get("variations").and_then(Value::as_u64).filter(|n| (1..=4).contains(n)).unwrap_or(3),
    })
}

/// Generate: run the fill (a background job in the app) and remember the variation count. The
/// prompt is not remembered: each fill asks for something new.
pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    let p = params(f);
    let result = app.run(COMMAND, p.clone());
    if result.is_ok() {
        app.session.prefs.edit(|prefs| prefs.dialogs.insert(COMMAND.into(), json!({"variations": p["variations"]})));
    }
    result
}

pub fn body(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    ui.spacing_mut().item_spacing.y = 6.0;
    ui.set_min_width(360.0);
    let mut prompt = f.get("prompt").and_then(Value::as_str).unwrap_or("").to_string();
    ui.label(tl!("Prompt:"));
    let edit = egui::TextEdit::multiline(&mut prompt)
        .hint_text(tl!("Describe what to add, or leave empty to fill from the surroundings"))
        .desired_rows(3)
        .desired_width(f32::INFINITY);
    let response = ui.add(edit).accessible_name(tl!("Prompt:"));
    // The prompt is what the dialog is for: focus it when the dialog opens.
    if !f.contains_key("__focused") {
        response.request_focus();
        f.insert("__focused".into(), json!(true));
    }
    if response.changed() {
        f.insert("prompt".into(), json!(prompt));
    }
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(tl!("Variations:"));
        let mut n = f.get("variations").and_then(Value::as_u64).unwrap_or(3).to_string();
        let opts: Vec<(String, &str)> = VARIATIONS.iter().map(|v| (v.to_string(), *v)).collect();
        if crate::widgets::dropdown(ui, "generative-fill-variations", tl!("Variations:"), &mut n, &opts, 60.0)
            && let Ok(v) = n.parse::<u64>()
        {
            f.insert("variations".into(), json!(v));
        }
    });
}

/// The Properties panel's Generative Fill section for a layer the fill made: its prompt and one
/// button per variation (the shown one selected). Nothing for other layers.
pub fn properties(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: &photocraft_doc::Layer) {
    let Some(doc) = app.session.active().map(|d| d.doc.id) else { return };
    let Some(info) = app.session.generative.variations(doc, layer.id) else { return };
    if !crate::props_layout::section(ui, "generative-fill", tl!("Generative Fill")) {
        return;
    }
    if !info.prompt.is_empty() {
        ui.label(egui::RichText::new(format!("“{}”", info.prompt)).italics());
    }
    let mut pick = None;
    ui.horizontal_wrapped(|ui| {
        ui.label(tl!("Variations:"));
        for i in 0..info.count {
            if ui.selectable_label(i == info.shown, (i + 1).to_string()).clicked() && i != info.shown {
                pick = Some(i);
            }
        }
    });
    if let Some(i) = pick
        && let Err(e) = app.run(VARIATION, json!({"layer": layer.id.0, "index": i}))
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_trim_the_prompt_and_bound_the_count() {
        let mut f = Map::new();
        f.insert("prompt".into(), json!("  a red barn \n"));
        f.insert("variations".into(), json!(4));
        assert_eq!(params(&f), json!({"prompt": "a red barn", "variations": 4}));
        f.insert("variations".into(), json!(9));
        assert_eq!(params(&f)["variations"], json!(3));
        assert_eq!(params(&Map::new()), json!({"prompt": "", "variations": 3}));
    }

    #[test]
    fn the_menu_opens_the_dialog_and_explicit_params_run_directly() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.run("select.rect", json!({"x": 4, "y": 4, "width": 8, "height": 8})).unwrap();
        // Without a backend the command is disabled, and says why instead of opening the dialog.
        let ctx = egui::Context::default();
        let err = crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).unwrap_err();
        assert!(err.contains("image model"), "{err}");
        let id = open(&mut app);
        let d = app.ui.dialogs.iter().find(|d| d.id == id).unwrap();
        assert!(owns(&d.fields));
        assert_eq!(params(&d.fields), json!({"prompt": "", "variations": 3}));
    }
}
