// End-to-end smoke test: serves docs/ over HTTP, loads the real site in
// headless Chrome, waits for the ONNX model to compile via WASM, and runs
// inference on a few sample images. Confirms the browser pipeline works
// (model download + WASM + preprocessing + softmax) without a server/API key.
//
// Run:  node model_export/test_browser.mjs
import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const ROOT = resolve(fileURLToPath(new URL(".", import.meta.url)), "..");
const DOCS = join(ROOT, "docs");
const DATA = join(ROOT, "data");
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";

const MIME = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "text/javascript",
  ".onnx": "application/octet-stream",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".wasm": "application/wasm",
};

function serve(dir, reqPath) {
  const rel = decodeURIComponent(reqPath.split("?")[0]);
  const file = normalize(join(dir, rel));
  if (!file.startsWith(dir)) return null;
  return file;
}

const server = http.createServer(async (req, res) => {
  try {
    let file = null;
    if (req.url.startsWith("/data/")) file = serve(DATA, req.url.replace("/data", ""));
    else file = serve(DOCS, req.url);
    if (!file) { res.writeHead(404); res.end("not found"); return; }
    const body = await readFile(file);
    res.writeHead(200, {
      "Content-Type": MIME[extname(file).toLowerCase()] || "application/octet-stream",
      "Cross-Origin-Opener-Policy": "same-origin",
    });
    res.end(body);
  } catch (e) {
    res.writeHead(404);
    res.end(String(e));
  }
});

const CASES = [
  ["/data/good/1.png", "good"],
  ["/data/surface_error/1.jpg", "surface_error"],
  ["/data/bend/01_tab_tilt_30deg_right.jpg", "bend"],
  // 12.jpg is a crack photo the model classifies as "crack" with ~100% confidence.
  ["/data/crack/12.jpg", "crack"],
];

function fmt(probs) {
  const names = ["good", "surface_error", "crack", "bend"];
  return names.map((n, i) => `${n}=${(probs[i] * 100).toFixed(1)}%`).join("  ");
}

const main = async () => {
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const port = server.address().port;
  const url = `http://127.0.0.1:${port}/index.html`;
  console.log("Serving at", url);

  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: "new",
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage();
  page.on("console", (m) => {
    if (m.type() === "error") console.log("[console.error]", m.text());
  });
  page.on("pageerror", (e) => console.log("[pageerror]", e.message));

  await page.goto(url, { waitUntil: "domcontentloaded" });

  // Wait for the model to compile (or fail).
  let ready = false;
  let status = "";
  for (let i = 0; i < 120; i++) {
    status = await page.evaluate(() => document.getElementById("status").textContent);
    ready = status.includes("Model ready") && await page.evaluate(() => typeof infer === "function");
    if (ready) break;
    if (i % 10 === 0) console.log("  waiting…", status);
    await new Promise((r) => setTimeout(r, 500));
  }
  if (!ready) {
    throw new Error("Model did not become ready. Status: " + status);
  }
  console.log("Model ready ✔");

  let pass = 0;
  for (const [path, expected] of CASES) {
    const probs = await page.evaluate(async (p) => {
      const img = new Image();
      img.src = p;
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
      return await infer(img); // global from app.js
    }, path);
    const names = ["good", "surface_error", "crack", "bend"];
    const top = names[probs.indexOf(Math.max(...probs))];
    const ok = top === expected;
    if (ok) pass++;
    console.log(`${ok ? "✔" : "✘"} ${path.split("/").pop().padEnd(30)} -> ${top.padEnd(14)} [${fmt(probs)}]`);
  }
  console.log(`\n${pass}/${CASES.length} correct.`);

  await browser.close();
  server.close();
  process.exit(pass === CASES.length ? 0 : 1);
};

main().catch((e) => { console.error(e); process.exit(2); });
