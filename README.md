# Lead-Acid Battery Plate Defect Checker

Image classification of lead-acid battery plate photographs into four classes, delivered as a **mobile-friendly web app** that runs entirely in the browser (no server, no API key, no data leaves the device).

**🔗 Live site:** https://alialasif.github.io/Lead-Acid_Battery_Plate_Defect_Checker/

| Class | Meaning |
|---|---|
| `good` | No visible void, crack or bend |
| `surface_error` | Missing paste, or a void exposing the grid beneath |
| `crack` | Visible fracture line in the paste surface |
| `bend` | Lug or plate physically deformed |

## The web app

Open the [live site](https://alialasif.github.io/Lead-Acid_Battery_Plate_Defect_Checker/) (or `docs/index.html` locally) to use it. Two modes:

- **Photo** — drop or tap to select an image. It shows the predicted class, confidence, and the probability for **all four** classes as bars.
- **Live camera** — streams the device camera (back camera by default, switchable to front) and overlays the predicted class plus **all four percentages** updating in real time.

The model runs in your browser using [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) over WebAssembly. The quantized model (`docs/model.onnx`, ~11 MB) is loaded once and then cached; inference happens locally.

> Live camera requires a **secure context** — `https://` (GitHub Pages is https) or `localhost`.

### Try it locally

Serve the `docs/` folder with any static server, then open it:

```
cd "Battery defect ML"
python -m http.server 8000 --directory docs
```

Open http://localhost:8000 in your browser. (For a one-line camera + model smoke test that runs headless Chrome, see `model_export/test_browser.mjs`.)

### Publish it as a website (GitHub Pages)

You don't need `git` installed — you can do everything in the GitHub web interface:

1. Create a new repository on github.com (e.g. `battery-plate-defect`). Make it **Public**.
2. Upload the **contents of the `docs/` folder** (`index.html`, `style.css`, `app.js`, `model.onnx`) into a `docs/` folder in the repository.
   - *Web UI path:* repo page → **Add file → Upload files**, then drag the four files and type `docs/` in the "commit message" — or create the folder by adding a `docs/.gitkeep` first, then upload into it.
3. Go to **Settings → Pages**.
4. Under **Build and deployment → Source**, choose **Deploy from a branch**, then select branch `main` and folder **`/docs`**, and click **Save**.
5. After a minute, your site is live at `https://<your-username>.github.io/<repo-name>/`.

> The `model.onnx` file is ~11 MB, well under GitHub's 100 MB per-file limit.

## Project structure

```
Battery defect ML/
├── docs/                       ← the website (GitHub Pages serves this folder)
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── model.onnx              ← int8-quantized model, ~11 MB
├── model_export/
│   ├── export_onnx.py          ← exports + quantizes the model (regenerable)
│   ├── test_browser.mjs        ← headless-Chrome end-to-end smoke test
│   └── package.json            ← dev dependency for the smoke test
├── notebooks/
│   ├── battery_defect_full.ipynb   ← training, evaluation, saves the model
│   ├── plate_classifier.pt         ← trained PyTorch weights (the source of the web model)
│   └── app.py                      ← desktop (OpenCV) version of the app
├── DATA_AUDIT.md               ← data-quality notes for the thesis
├── README.md, LICENSE, .gitignore
└── requirements.txt, environment.yml
```

`data/` (the training images) is kept locally and **not** committed — images are not part of this repository.

## The model

| Item | Setting |
|---|---|
| Architecture | ResNet18, ImageNet weights; only `layer4` and the 4-way output layer are trained |
| Input | 224 × 224 |
| Augmentation (training only) | rotation ±20°, horizontal flip, brightness/contrast jitter, random resized crop |
| Loss | cross-entropy with class weights |
| Optimiser | Adam, learning rate 1e-4, batch size 8, 15 epochs |
| Evaluation | leave-one-plate-out cross-validation |
| Web model | same weights, **statically quantized to int8 (QDQ)** (~4× smaller, ~11 MB) |

**About the quantized web model.** The website uses a statically int8-quantized ONNX model (QDQ format) for fast loading on phones. On the test set it matches the float32 model except on borderline `crack` vs `bend` cases (maximum per-image probability shift ≈ 5%), where the float32 model is already uncertain. The **accuracy figures you report in the thesis must come from the full-precision PyTorch model's cross-validation** (run `notebooks/battery_defect_full.ipynb`), not from the quantized web model.

To regenerate the ONNX files after retraining:

```
.venv\Scripts\python model_export\export_onnx.py
```

## Training and evaluation

```
cd "Battery defect ML\notebooks"
jupyter lab
```

Open `battery_defect_full.ipynb` and run the cells from top to bottom. Sections 5–6 run the leave-one-plate-out cross-validation (the numbers to report); section 8 trains the final model on all images and saves `plate_classifier.pt`.

### ⚠️ Before you trust the cross-validation numbers

The notebook reads the **plate ID from the filename prefix** (everything before the first `_`). Files with no `_` — e.g. `1.png`, `10.jpg` (most of `data/good/` and `data/surface_error/`, and `12.jpg`…`28.jpg` in `data/crack/`) — are each treated as their **own plate**. That silently defeats the plate-level split: those photos can't be properly held out, and real plates get split into many singletons, which **inflates the cross-validation score**.

Fix this by renaming every file to `plateID_something.jpg` (same physical plate → same prefix), then re-run. This is the single most important correction before reporting results.

## Results

Leave-one-plate-out cross-validation on the current dataset (80 images).

> ⚠️ **Preliminary.** These numbers come from the current filenames, where the
> plate split and the `synthetic` flag are **not yet correct** (see
> `DATA_AUDIT.md`). Treat them as a baseline — re-run after fixing the data
> before reporting them in the thesis.

### All images ("with generated images")

Per-plate accuracy: **0.79 ± 0.31** over 45 plates

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| good | 0.77 | 0.71 | 0.74 | 14 |
| surface_error | 1.00 | 1.00 | 1.00 | 24 |
| crack | 0.75 | 0.67 | 0.71 | 27 |
| bend | 0.63 | 0.80 | 0.71 | 15 |
| **accuracy** | | | 0.80 | 80 |
| **macro avg** | 0.79 | 0.80 | 0.79 | 80 |
| **weighted avg** | 0.81 | 0.80 | 0.80 | 80 |

Confusion matrix (rows = actual, columns = predicted; order: good, surface_error, crack, bend):

| actual ↓ / predicted → | good | surface_error | crack | bend |
|---|---|---|---|---|
| good | 10 | 0 | 3 | 1 |
| surface_error | 0 | 24 | 0 | 0 |
| crack | 3 | 0 | 18 | 6 |
| bend | 0 | 0 | 3 | 12 |

### Ablation ("real images only")

Per-plate accuracy: **0.77 ± 0.32** over 45 plates

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| good | 0.77 | 0.71 | 0.74 | 14 |
| surface_error | 1.00 | 1.00 | 1.00 | 24 |
| crack | 0.75 | 0.67 | 0.71 | 27 |
| bend | 0.58 | 0.73 | 0.65 | 15 |
| **accuracy** | | | 0.79 | 80 |
| **macro avg** | 0.77 | 0.78 | 0.77 | 80 |
| **weighted avg** | 0.80 | 0.79 | 0.79 | 80 |

Confusion matrix:

| actual ↓ / predicted → | good | surface_error | crack | bend |
|---|---|---|---|---|
| good | 10 | 0 | 2 | 2 |
| surface_error | 0 | 24 | 0 | 0 |
| crack | 3 | 0 | 18 | 6 |
| bend | 0 | 0 | 4 | 11 |

Package versions for reproducibility: `pip list | findstr /i "torch numpy pillow matplotlib scikit-learn opencv"`

## Desktop app

A desktop (OpenCV) version with webcam / image upload is available at `notebooks/app.py` (`python app.py live`, `image`, or `pick`). The web app in `docs/` is the primary interface.

## Limitations

- Small dataset from few plates; the effective sample size per class is the number of distinct defects, not photos.
- One label per image — the model does not localise the defect.
- Plates came from assembled batteries; a production camera sees plates earlier (post-pasting/curing), where appearance differs. Performance on a real line is untested.
- The `bend` class may rest on few real examples.

## Safety and data

Plates contain lead and may carry acid residue. Handle them with gloves and eye protection, do not create dust, and wash hands afterwards. Do not crack or flex plates on purpose. Get plant permission before publishing images or results.

## Author and license

[Author], [Department], [University]

License: [choose one, e.g. MIT for the code]. Images under `data/` are not covered by the code license.
