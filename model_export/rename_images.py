"""Rename image files under data/ into a clean, ordered, professional scheme.

What it does (ONLY renames — never deletes, moves, or edits pixels):
- Zero-pads the plain-numbered files in good/ and surface_error/.
- Makes crack/ contiguous (the descriptive 01-10 stay, the plain 12-28 become
  11-27 so there is no gap).
- Leaves the descriptive synthetic names (crack 01-10, bend 01-15) as they are,
  because they describe the simulated defect.

Every old->new mapping is written to RENAME_LOG.csv at the project root, so the
change is fully traceable and reversible.
"""
import csv
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOG = os.path.join(ROOT, "RENAME_LOG.csv")


def nat_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def build_mapping():
    mapping = []  # list of (class, old, new)

    # --- good: zero-pad 1..14 (preserve extension) ---
    good = sorted(os.listdir(os.path.join(DATA, "good")), key=nat_key)
    for i, f in enumerate(good, start=1):
        ext = os.path.splitext(f)[1]
        mapping.append(("good", f, f"{i:02d}{ext}"))

    # --- surface_error: zero-pad 1..24 ---
    se = sorted(os.listdir(os.path.join(DATA, "surface_error")), key=nat_key)
    for i, f in enumerate(se, start=1):
        ext = os.path.splitext(f)[1]
        mapping.append(("surface_error", f, f"{i:02d}{ext}"))

    # --- crack: keep descriptive 01-10; renumber plain 12-28 -> 11-27 ---
    crack = sorted(os.listdir(os.path.join(DATA, "crack")), key=nat_key)
    plain = [f for f in crack if "_" not in f]  # e.g. 12.jpg ... 28.jpg
    for f in crack:
        if "_" in f:
            mapping.append(("crack", f, f))  # descriptive: unchanged
    for i, f in enumerate(plain, start=11):
        ext = os.path.splitext(f)[1]
        mapping.append(("crack", f, f"{i:02d}{ext}"))

    # --- bend: already ordered + descriptive; unchanged ---
    for f in sorted(os.listdir(os.path.join(DATA, "bend")), key=nat_key):
        mapping.append(("bend", f, f))

    return mapping


def apply(mapping):
    # Two-phase rename (old -> temp -> new) so no target name is ever occupied
    # by a file that has not yet been renamed.
    changes = [(c, o, n) for c, o, n in mapping if o != n]
    phase1 = []
    for idx, (c, o, n) in enumerate(changes):
        src = os.path.join(DATA, c, o)
        tmp = os.path.join(DATA, c, f"._rename_tmp_{idx:03d}_{o}")
        os.rename(src, tmp)
        phase1.append((c, o, n, tmp))
    for c, o, n, tmp in phase1:
        dst = os.path.join(DATA, c, n)
        os.rename(tmp, dst)
    return len(changes)


def main():
    mapping = build_mapping()
    with open(LOG, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "old_name", "new_name"])
        for c, o, n in mapping:
            w.writerow([c, o, n])
    changed = apply(mapping)
    print(f"Renamed {changed} files. Log written to {LOG}")
    for c, o, n in mapping:
        if o != n:
            print(f"  {c}/{o}  ->  {n}")


if __name__ == "__main__":
    main()
