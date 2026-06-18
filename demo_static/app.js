/* FloodLens — in-browser inference.
 *
 * Mirrors the Python pipeline (space_mapper.inference.predict_with_checkpoint):
 *   read GeoTIFF -> bands / 32767 -> resize to 128 (area-average) -> ONNX U-Net
 *   -> sigmoid -> bilinear upsample to native size -> threshold 0.5 -> overlay.
 * The NDWI baseline matches heuristic_predict: (G - NIR) / (G + NIR) > 0.
 */
"use strict";

const MODEL_URL = "model/tiny_unet_diverse.onnx";
const IN_CH = 13;
const NET_SIZE = 128;
const WATER = [14, 165, 233]; // sky-500

const $ = (id) => document.getElementById(id);
const statusEl = () => $("status-value");

let session = null;
let current = null; // { bands, width, height, name, truthPct?, scene? }
let model = "unet";

function setStatus(html) {
  const el = statusEl();
  if (el) el.innerHTML = html;
}

function setStats(rows) {
  $("stats").innerHTML = rows
    .map(([label, value, green]) =>
      `<div class="stat"><span class="stat-label">${label}</span>` +
      `<span class="stat-value${green ? " g" : ""}">${value}</span></div>`)
    .join("");
}

/* ---------- image helpers ---------- */

function percentileStretch(band, n) {
  // 2-98 percentile stretch to 0..255, like the training-side previews.
  const sample = new Float32Array(band.length);
  sample.set(band);
  sample.sort();
  const lo = sample[Math.floor(0.02 * (n - 1))];
  const hi = sample[Math.floor(0.98 * (n - 1))];
  const scale = 255 / Math.max(hi - lo, 1e-6);
  const out = new Uint8ClampedArray(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.min(255, Math.max(0, (band[i] - lo) * scale));
  }
  return out;
}

function toRGB(bands, w, h) {
  // Sentinel-2 true color: B4 (idx 3) red, B3 (idx 2) green, B2 (idx 1) blue.
  const n = w * h;
  const r = percentileStretch(bands[3], n);
  const g = percentileStretch(bands[2], n);
  const b = percentileStretch(bands[1], n);
  const rgba = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    rgba[i * 4] = r[i]; rgba[i * 4 + 1] = g[i]; rgba[i * 4 + 2] = b[i]; rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

function resizeChannel(src, sw, sh, dw, dh) {
  // Downscale: area average (close to PIL's antialiased bilinear).
  // Upscale: standard bilinear.
  const out = new Float32Array(dw * dh);
  if (dw < sw) {
    const fx = sw / dw, fy = sh / dh;
    for (let y = 0; y < dh; y++) {
      const y0 = Math.floor(y * fy), y1 = Math.min(sh, Math.ceil((y + 1) * fy));
      for (let x = 0; x < dw; x++) {
        const x0 = Math.floor(x * fx), x1 = Math.min(sw, Math.ceil((x + 1) * fx));
        let sum = 0, cnt = 0;
        for (let yy = y0; yy < y1; yy++)
          for (let xx = x0; xx < x1; xx++) { sum += src[yy * sw + xx]; cnt++; }
        out[y * dw + x] = sum / cnt;
      }
    }
  } else {
    const fx = (sw - 1) / Math.max(dw - 1, 1), fy = (sh - 1) / Math.max(dh - 1, 1);
    for (let y = 0; y < dh; y++) {
      const sy = y * fy, y0 = Math.floor(sy), y1 = Math.min(sh - 1, y0 + 1), wy = sy - y0;
      for (let x = 0; x < dw; x++) {
        const sx = x * fx, x0 = Math.floor(sx), x1 = Math.min(sw - 1, x0 + 1), wx = sx - x0;
        out[y * dw + x] =
          src[y0 * sw + x0] * (1 - wx) * (1 - wy) + src[y0 * sw + x1] * wx * (1 - wy) +
          src[y1 * sw + x0] * (1 - wx) * wy + src[y1 * sw + x1] * wx * wy;
      }
    }
  }
  return out;
}

/* ---------- prediction paths ---------- */

async function runUNet(bands, w, h) {
  const t0 = performance.now();
  // Build 1 x 13 x 128 x 128 tensor (pad/truncate channels to 13).
  const input = new Float32Array(IN_CH * NET_SIZE * NET_SIZE);
  for (let c = 0; c < Math.min(IN_CH, bands.length); c++) {
    input.set(resizeChannel(bands[c], w, h, NET_SIZE, NET_SIZE), c * NET_SIZE * NET_SIZE);
  }
  const tensor = new ort.Tensor("float32", input, [1, IN_CH, NET_SIZE, NET_SIZE]);
  const { logits } = await session.run({ input: tensor });
  // sigmoid -> upsample probs to native -> threshold (same order as Python).
  const probsSmall = Float32Array.from(logits.data, (z) => 1 / (1 + Math.exp(-z)));
  const probs = resizeChannel(probsSmall, NET_SIZE, NET_SIZE, w, h);
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < mask.length; i++) mask[i] = probs[i] >= 0.5 ? 1 : 0;
  return { mask, ms: performance.now() - t0, label: "Flood U-Net · edge ONNX" };
}

function runNDWI(bands, w, h) {
  const t0 = performance.now();
  const g = bands[1], nir = bands[3]; // matches heuristic_predict channel picks
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < mask.length; i++) {
    mask[i] = (g[i] - nir[i]) / (g[i] + nir[i] + 1e-6) > 0 ? 1 : 0;
  }
  return { mask, ms: performance.now() - t0, label: "NDWI baseline · classical index" };
}

/* ---------- canvas ---------- */

function draw(canvasId, rgba, w, h, mask) {
  const canvas = $(canvasId);
  canvas.width = w; canvas.height = h;
  const data = new Uint8ClampedArray(rgba); // copy
  if (mask) {
    for (let i = 0; i < w * h; i++) {
      if (mask[i]) {
        data[i * 4]     = 0.3 * data[i * 4]     + 0.7 * WATER[0];
        data[i * 4 + 1] = 0.3 * data[i * 4 + 1] + 0.7 * WATER[1];
        data[i * 4 + 2] = 0.3 * data[i * 4 + 2] + 0.7 * WATER[2];
      }
    }
  }
  canvas.getContext("2d").putImageData(new ImageData(data, w, h), 0, 0);
}

/* ---------- I/O ---------- */

async function loadTiff(buffer, name, meta) {
  setStatus("Reading GeoTIFF…");
  const tiff = await GeoTIFF.fromArrayBuffer(buffer);
  const image = await tiff.getImage();
  const w = image.getWidth(), h = image.getHeight();
  const rasters = await image.readRasters();
  // Normalize exactly like the Python loader: int16 -> / 32767.
  const bands = [];
  for (let c = 0; c < rasters.length; c++) {
    const raw = rasters[c];
    const f = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) f[i] = raw[i] / 32767;
    bands.push(f);
  }
  if (bands.length < 4) {
    setStats([
      ["STATUS", "This file isn't a 13-band Sentinel-2 chip"],
      ["BANDS", String(bands.length) + " (need ≥4, ideally 13)"],
      ["HINT", "Download a sample above and re-upload it"],
    ]);
    current = null;
    return;
  }
  current = {
    bands, width: w, height: h, name,
    truthPct: meta && meta.truthPct != null ? meta.truthPct : null,
    scene: meta && meta.scene ? meta.scene : null,
  };
  draw("canvas-rgb", toRGB(bands, w, h), w, h, null);
  $("canvas-overlay").getContext("2d").clearRect(0, 0, w, h);
  const rows = [["FILE", name], ["BANDS", String(bands.length)], ["SIZE", `${w}×${h} px`],
                ["STATUS", "Ready — hit predict"]];
  if (current.scene) rows.splice(1, 0, ["SCENE", current.scene]);
  setStats(rows);
}

async function predict() {
  if (!current) { setStatus("Load a chip first."); return; }
  if (model === "unet" && !session) { setStatus("Model still loading…"); return; }
  const btn = $("predict-btn");
  btn.disabled = true;
  setStats([["STATUS", "Running " + (model === "unet" ? "flood detection…" : "NDWI baseline…")]]);
  await new Promise((r) => setTimeout(r, 30)); // let the UI paint
  try {
    const { bands, width, height, name } = current;
    const res = model === "unet" ? await runUNet(bands, width, height) : runNDWI(bands, width, height);
    let water = 0;
    for (let i = 0; i < res.mask.length; i++) water += res.mask[i];
    const pct = (100 * water / res.mask.length).toFixed(1);
    draw("canvas-overlay", toRGB(bands, width, height), width, height, res.mask);
    const rows = [
      ["METHOD", res.label],
      ["FLOOD AREA", pct + "%", true],
    ];
    if (current.truthPct != null) {
      const delta = (parseFloat(pct) - current.truthPct).toFixed(1);
      const sign = delta > 0 ? "+" : "";
      rows.push(["LABELED FLOOD", current.truthPct.toFixed(1) + "%"]);
      rows.push(["Δ VS LABEL", sign + delta + " pp"]);
    }
    if (current.scene) rows.push(["SCENE", current.scene]);
    rows.push(["RESOLUTION", `${width}×${height} px`]);
    rows.push(["INFERENCE", res.ms.toFixed(0) + " ms · client-side"]);
    setStats(rows);
  } catch (e) {
    console.error(e);
    setStats([["STATUS", "Prediction failed — see console"]]);
  } finally {
    btn.disabled = false;
  }
}

/* ---------- wiring ---------- */

async function init() {
  // model segment buttons
  $("model-seg").addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    model = b.dataset.model;
    for (const x of $("model-seg").querySelectorAll("button")) x.classList.toggle("on", x === b);
  });

  // upload (recruiter path: download a sample, then re-upload here)
  const dz = $("dropzone"), fi = $("file-input");
  fi.addEventListener("change", async () => {
    if (fi.files[0]) await loadTiff(await fi.files[0].arrayBuffer(), fi.files[0].name, null);
  });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", async (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    const f = e.dataTransfer.files[0];
    if (f) await loadTiff(await f.arrayBuffer(), f.name, null);
  });

  // samples — one-click run + Download buttons for the upload-flow demo
  for (const b of document.querySelectorAll(".examples .ex")) {
    b.addEventListener("click", async () => {
      setStats([["STATUS", "Fetching sample…"]]);
      const meta = {
        truthPct: b.dataset.truth != null ? parseFloat(b.dataset.truth) : null,
        scene: b.dataset.scene || null,
      };
      const resp = await fetch(b.dataset.file);
      await loadTiff(await resp.arrayBuffer(), b.dataset.file.split("/").pop(), meta);
      await predict();
    });
  }

  $("predict-btn").addEventListener("click", predict);

  // ONNX session
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  } catch (e) {
    console.error(e);
    setStats([["STATUS", "Model failed to load — NDWI baseline still works"],
              ["ERROR", String(e && e.message ? e.message : e).slice(0, 300)]]);
    return initAutorun();
  }
  setStats([["STATUS", "Model loaded — click a sample or upload a chip"]]);
  initAutorun();
}

async function initAutorun() {
  // Auto-run the first sample so visitors land on a live result.
  try {
    const first = document.querySelector(".examples .ex");
    if (!first) return;
    const meta = {
      truthPct: first.dataset.truth != null ? parseFloat(first.dataset.truth) : null,
      scene: first.dataset.scene || null,
    };
    const resp = await fetch(first.dataset.file);
    await loadTiff(await resp.arrayBuffer(), first.dataset.file.split("/").pop(), meta);
    await predict();
  } catch (e) { console.error("autorun:", e); }
}

init();
