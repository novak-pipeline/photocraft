# Generative Fill

> **Status:** working on the `generative-fill` branch of `novak-pipeline/photocraft`. Desktop only.
> Runs locally through ComfyUI with FLUX.1 Fill; nothing leaves the computer.

Select an area, describe what should be there (or leave it empty), and get new content that blends
in, on a new layer masked to the selection. It mirrors Photoshop's **Edit › Generative Fill**, and the
`generative-fill` tool the Craft AI catalogue already declares for PhotoCraft.

## Using it

1. Make a selection (any selection tool; feathered edges are kept).
2. **Edit › Generative Fill…**
3. Type a prompt — *"a wooden rowing boat"*, *"snow on the roof"* — or leave it empty to fill from
   the surroundings (removing an object). Pick how many variations (1–4, default 3).
4. **Generate.** It runs as a background job with progress and **Cancel**; the app stays usable.
5. The result lands on a **new layer named after the prompt**, with a **layer mask** made from the
   selection. Paint the mask to reveal more or less of it; it is one undo step.
6. Flip between variations in **Properties › Generative Fill** (one button per variation; each switch
   is an undo step too).

Agents, Actions, the CLI and MCP call the command directly:

```json
{"command": "edit.generativeFill", "params": {"prompt": "a red barn", "variations": 3, "seed": 0}}
{"command": "edit.generativeFill.variation", "params": {"index": 2}}
{"command": "edit.generativeFill.status"}
```

## What you need

- **ComfyUI running** on this computer (`http://127.0.0.1:8188` by default). When it isn't, the menu
  item is greyed out and says so.
- These model files, which ComfyUI lists automatically:

  | Folder | File | Notes |
  |---|---|---|
  | `models/diffusion_models` | a FLUX.1 **Fill** model, e.g. `flux1-fill-dev_fp8.safetensors` | any name containing `fill` |
  | `models/clip` or `text_encoders` | `clip_l.safetensors` | |
  | `models/clip` or `text_encoders` | `t5xxl_fp16.safetensors` (or `t5xxl_fp8_e4m3fn`) | fp16 preferred |
  | `models/vae` | `ae.safetensors` | the FLUX VAE |

  Only core ComfyUI nodes are used, so no custom node packs are needed.

### Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `PHOTOCRAFT_COMFYUI_URL` | `http://127.0.0.1:8188` | another ComfyUI (plain `http://` only) |
| `PHOTOCRAFT_GF_UNET` | first FLUX Fill model found | e.g. the OneReward fill variant |
| `PHOTOCRAFT_GF_CLIP_L` / `PHOTOCRAFT_GF_T5` / `PHOTOCRAFT_GF_VAE` | found by name | |

## How it works

```
engine (crates/engine/src/generative_cmds.rs)      desktop (apps/photocraft/src/generative.rs)
  edit.generativeFill                                ComfyFill: GenerativeBackend
   ├─ window = selection + context, clipped          ├─ upload window + mask (ComfyUI temp area)
   ├─ read visible pixels + selection coverage       ├─ queue one FLUX.1 Fill graph per variation
   ├─ background job ──── FillRequest ─────────────▶ │   (own seed each; scaled to ~1 MP and back)
   │                  ◀── Vec<FillImage> ──────────  ├─ poll /history, read /view, decode PNG
   └─ new layer + layer mask, one undo step          └─ cancel: drop queued prompts, interrupt
```

- **The engine never touches the network** (the same rule as the release check). It defines the
  `GenerativeBackend` trait; the desktop app injects ComfyUI through `PlatformServices::generative`.
  The CLI, tests and web build have no backend, so the command is disabled with a reason.
- **What is sent:** only the visible pixels inside the window (the selection plus context on each
  side), the selection as a mask, the prompt, the count and a seed.
- **The window** is the selection's bounds plus half its longer side (at least 64 px) of context,
  clipped to the canvas, up to 64 megapixels. ComfyUI scales it to about a megapixel for the model
  (at most 4× up, in multiples of 16) and back to the window's exact size.
- **Seams:** the model may repaint 8 px past the selection; the layer mask (the selection's own,
  feathered coverage) hides that, so edges blend.
- **Colour modes:** RGB, Grayscale and CMYK documents work; the model sees 8-bit RGB and the result
  is converted back to the document's mode on the new layer.

## Tests

- `cargo test -p photocraft-engine --lib generative_cmds` — the command with a stand-in model:
  disabled states, the request sent, the masked layer, undo, variations, misbehaving or failing
  backends leaving the document untouched, bad params, CMYK and Grayscale. The engine's
  `panic_hunt` fuzzes all three commands with hostile params.
- `cargo test -p photocraft --bin photocraft generative` — model picking, sizing, PNG round trip,
  the graph's wiring, ComfyUI error messages, safe names.

## Not done yet

- **Thumbnails** of the variations in Properties (it shows numbered buttons).
- **A contextual task bar** under the selection, as in Photoshop; today it is the Edit menu.
- **Cloud models.** The Craft AI layer (`craft-ai-integrations`) already has OpenAI and BFL
  inpainting adapters. Its ComfyUI adapter can't vary the seed per request yet, which is why this
  branch talks to ComfyUI directly. A backend built on `craft-ai-client` slots in through the same
  trait.
- **Colour management at the model boundary:** the model is sent the document's RGB values as they
  are, which is exact for sRGB documents and slightly off for wide-gamut ones.
- **Generative layers survive saving** as ordinary masked pixel layers; the variations live only
  for the session.
