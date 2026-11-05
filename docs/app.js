/*
 * Battery plate defect checker — runs the quantized ResNet18 (docs/model.onnx)
 * entirely in the browser via ONNX Runtime Web. No server, no API key.
 *
 * Preprocessing matches the training notebook exactly:
 *   stretch to 224x224 -> [0,1] -> normalize(mean, std)
 */

const CLASSES = [
  { id: "good",          label: "Good",           color: "#22c55e", desc: "No visible void, crack or bend" },
  { id: "surface_error", label: "Surface error",  color: "#f59e0b", desc: "Missing paste / void exposing the grid" },
  { id: "crack",         label: "Crack",          color: "#ef4444", desc: "Visible fracture line in the paste" },
  { id: "bend",          label: "Bend",           color: "#8b5cf6", desc: "Lug or plate physically deformed" },
];

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const IMG_SIZE = 224;
const CONF_THRESHOLD = 0.60; // below this the result is flagged "low confidence"

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const preCanvas = $("pre-canvas");
const preCtx = preCanvas.getContext("2d", { willReadFrequently: true });
preCtx.imageSmoothingEnabled = true;
preCtx.imageSmoothingQuality = "high";

let session = null;        // ONNX Runtime InferenceSession
let modelReady = false;
let stream = null;         // camera MediaStream
let facingMode = "environment";
let liveRunning = false;
let inferBusy = false;
let rafId = null;

// ---------------------------------------------------------------------------
// Status + model loading
// ---------------------------------------------------------------------------
function setStatus(msg, kind) {
  statusEl.hidden = false;
  statusEl.className = "status" + (kind ? " " + kind : "");
  statusEl.textContent = msg;
}

async function loadModel() {
  setStatus("Downloading model (\u2248 11 MB)…");
  try {
    const resp = await fetch("model.onnx");
    if (!resp.ok) throw new Error("model.onnx not found (HTTP " + resp.status + ")");
    const total = Number(resp.headers.get("Content-Length")) || 0;
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total) setStatus(`Downloading model… ${Math.round((received / total) * 100)}%`);
    }
    const buf = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }

    setStatus("Compiling model…");
    session = await ort.InferenceSession.create(buf.buffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    modelReady = true;
    setStatus("Model ready — choose a photo or start the camera.", "ok");
  } catch (err) {
    console.error(err);
    setStatus("Failed to load model: " + err.message, "err");
  }
}

// ---------------------------------------------------------------------------
// Tensor + inference
// ---------------------------------------------------------------------------
function imageDataToTensor(imageData) {
  const data = imageData.data;
  const n = IMG_SIZE * IMG_SIZE;
  const input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    input[i]       = (data[i * 4]     / 255 - MEAN[0]) / STD[0]; // R
    input[n + i]   = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1]; // G
    input[2 * n + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2]; // B
  }
  return new ort.Tensor("float32", input, [1, 3, IMG_SIZE, IMG_SIZE]);
}

function drawToCanvas(source) {
  // Stretch (not crop) to 224x224, matching torchvision Resize((224,224)).
  preCtx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
  preCtx.drawImage(source, 0, 0, IMG_SIZE, IMG_SIZE);
  return preCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
}

function softmax(logits) {
  const max = Math.max.apply(null, logits);
  const exps = Array.from(logits, (x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

async function infer(source) {
  if (!modelReady) throw new Error("Model not ready");
  const imageData = drawToCanvas(source);
  const feeds = { input: imageDataToTensor(imageData) };
  const results = await session.run(feeds);
  return softmax(Array.from(results.logits.data)); // [good, surface_error, crack, bend]
}

// ---------------------------------------------------------------------------
// Result rendering
// ---------------------------------------------------------------------------
function renderBars(container, probs) {
  const topIdx = probs.indexOf(Math.max.apply(null, probs));
  container.innerHTML = "";
  CLASSES.forEach((cls, i) => {
    const pct = Math.round(probs[i] * 100);
    const row = document.createElement("div");
    row.className = "bar-row" + (i === topIdx ? " top" : "");
    row.innerHTML = `
      <span class="bar-label">${cls.label}</span>
      <span class="bar-track"><span class="bar-fill" style="background:${cls.color}"></span></span>
      <span class="bar-pct">${pct}%</span>`;
    container.appendChild(row);
  });
  requestAnimationFrame(() => {
    container.querySelectorAll(".bar-row").forEach((rowEl, i) => {
      rowEl.querySelector(".bar-fill").style.width = Math.round(probs[i] * 100) + "%";
    });
  });
}

function renderVerdict(el, probs) {
  const topIdx = probs.indexOf(Math.max.apply(null, probs));
  const cls = CLASSES[topIdx];
  const conf = probs[topIdx];
  if (conf < CONF_THRESHOLD) {
    el.innerHTML = `
      <div class="v-class" style="color:#f59e0b">Uncertain</div>
      <div class="v-conf">${Math.round(conf * 100)}% confidence — best guess: ${cls.label}</div>
      <div class="v-note uncertain">Low confidence — treat with care. ${cls.desc}</div>`;
  } else {
    el.innerHTML = `
      <div class="v-class" style="color:${cls.color}">${cls.label}</div>
      <div class="v-conf">${Math.round(conf * 100)}% confidence</div>
      <div class="v-note">${cls.desc}</div>`;
  }
}

// ---------------------------------------------------------------------------
// Photo / upload mode
// ---------------------------------------------------------------------------
const dropzone = $("dropzone");
const fileInput = $("file-input");
const uploadResult = $("upload-result");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) handleFile(fileInput.files[0]);
});
$("btn-clear").addEventListener("click", () => {
  uploadResult.hidden = true;
  dropzone.hidden = false;
  fileInput.value = "";
});

async function loadBitmap(file) {
  if ("createImageBitmap" in window) {
    try {
      return await createImageBitmap(file, { imageOrientation: "from-image" });
    } catch (_) { /* fall through to <img> path */ }
  }
  return await new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

async function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    setStatus("Please choose an image file.", "err");
    return;
  }
  // Show the preview immediately (browser applies EXIF orientation for display).
  const preview = $("preview");
  preview.src = URL.createObjectURL(file);
  dropzone.hidden = true;
  uploadResult.hidden = false;
  $("upload-verdict").innerHTML = '<div class="v-class">Classifying…</div>';
  $("upload-bars").innerHTML = "";

  try {
    const bmp = await loadBitmap(file);
    const probs = await infer(bmp);
    renderVerdict($("upload-verdict"), probs);
    renderBars($("upload-bars"), probs);
  } catch (err) {
    console.error(err);
    $("upload-verdict").innerHTML =
      '<div class="v-class">Error</div><div class="v-conf">' + err.message + "</div>";
    setStatus("Inference failed: " + err.message, "err");
  }
}

// ---------------------------------------------------------------------------
// Live camera mode
// ---------------------------------------------------------------------------
const video = $("video");
const btnStart = $("btn-start");
const btnSwitch = $("btn-switch");
const liveVerdict = $("live-verdict");
const liveBars = $("live-bars");
const cameraNote = $("camera-note");

btnStart.addEventListener("click", async () => {
  if (stream) { stopCamera(); return; }
  await startCamera();
});

btnSwitch.addEventListener("click", () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  stopCamera(true);
  startCamera();
});

async function startCamera() {
  if (!modelReady) { setStatus("Wait for the model to finish loading.", "err"); return; }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("Camera needs a secure context (https:// or localhost) and a modern browser.", "err");
    cameraNote.textContent = "Open this page over https:// (GitHub Pages is https).";
    return;
  }
  setStatus("Starting camera…");
  cameraNote.textContent = "";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    video.classList.toggle("mirror", facingMode === "user");
    await video.play();
    btnStart.textContent = "Stop camera";
    btnSwitch.hidden = false;
    liveVerdict.innerHTML = '<div class="v-class">Scanning…</div>';
    liveRunning = true;
    setStatus("Live mode active.", "ok");
    tick();
  } catch (err) {
    console.error(err);
    liveRunning = false;
    setStatus("Camera error: " + err.message, "err");
    cameraNote.textContent = "Check camera permission, and that no other app is using the camera.";
  }
}

function stopCamera(keepForSwitch) {
  liveRunning = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  btnStart.textContent = "Start camera";
  btnSwitch.hidden = !keepForSwitch;
  if (!keepForSwitch) {
    liveVerdict.innerHTML = '<div class="v-class">Camera not started</div>';
    liveBars.innerHTML = "";
    setStatus("Model ready — choose a photo or start the camera.", "ok");
  }
}

let lastFrameTime = 0;
async function tick() {
  rafId = requestAnimationFrame(tick);
  if (!liveRunning || !modelReady || inferBusy) return;

  // Throttle: run at most every ~150ms (about 6-7 fps) to keep phones responsive.
  const now = performance.now();
  if (now - lastFrameTime < 150) return;
  if (video.readyState < 2) return;

  inferBusy = true;
  lastFrameTime = now;
  try {
    const probs = await infer(video);
    renderVerdict(liveVerdict, probs);
    renderBars(liveBars, probs);
  } catch (err) {
    console.error(err);
  } finally {
    inferBusy = false;
  }
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const which = tab.dataset.tab;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $("panel-upload").hidden = which !== "upload";
    $("panel-live").hidden = which !== "live";
    if (which !== "live") stopCamera();
  });
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
loadModel();
