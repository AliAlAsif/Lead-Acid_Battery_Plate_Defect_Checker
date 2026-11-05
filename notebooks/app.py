# Plate defect inspection app: live camera or manual image upload.
# Usage:
#   python app.py live                      # webcam, q = quit, s = save frame
#   python app.py pick                      # opens a file dialog, repeat until you cancel
#   python app.py image --path photo.jpg    # one image
# Options: --camera 0 (camera index), --model plate_classifier.pt

import argparse
import collections

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CONF_THRESHOLD = 0.60   # below this the app says "uncertain" instead of guessing

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    classes = ckpt["classes"]
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


def predict(model, classes, pil_img):
    x = preprocess(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    i = int(probs.argmax())
    return classes[i], float(probs[i]), probs


def draw_label(frame, text, is_good):
    color = (0, 200, 0) if is_good else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def run_live(model, classes, cam_index):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Could not open camera", cam_index)
        return
    recent = collections.deque(maxlen=5)   # majority vote over 5 frames, stops flicker
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label, conf, _ = predict(model, classes, Image.fromarray(rgb))
        recent.append(label if conf >= CONF_THRESHOLD else "uncertain")
        shown = collections.Counter(recent).most_common(1)[0][0]
        draw_label(frame, "%s  (%.0f%%)" % (shown, conf * 100), shown == "good")
        cv2.imshow("Plate inspection - q quit, s save", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = "capture_%03d.jpg" % saved
            cv2.imwrite(name, raw)
            saved += 1
            print("saved", name)
    cap.release()
    cv2.destroyAllWindows()


def run_image(model, classes, path):
    img = cv2.imread(path)
    if img is None:
        print("Could not read", path)
        return
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label, conf, probs = predict(model, classes, Image.fromarray(rgb))
    print(path)
    for c, p in zip(classes, probs):
        print("  %-14s %.1f%%" % (c, p * 100))
    shown = label if conf >= CONF_THRESHOLD else "uncertain"
    h, w = img.shape[:2]
    scale = min(1.0, 900.0 / max(h, w))
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    draw_label(img, "%s  (%.0f%%)" % (shown, conf * 100), shown == "good")
    cv2.imshow("Plate inspection - press any key", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pick_file():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose a plate image",
        filetypes=[("Images", "*.jpg *.jpeg *.png")])
    root.destroy()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["live", "image", "pick"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="plate_classifier.pt")
    args = ap.parse_args()

    model, classes = load_model(args.model)
    if args.mode == "live":
        run_live(model, classes, args.camera)
    elif args.mode == "image":
        if not args.path:
            print("Give an image with --path")
            return
        run_image(model, classes, args.path)
    else:
        while True:
            path = pick_file()
            if not path:
                break
            run_image(model, classes, path)


if __name__ == "__main__":
    main()
