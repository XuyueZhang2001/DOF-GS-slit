"""
One-time migration after the removal of --focus_mode: every simulated dataset now
has per-pose focus and no _pf suffix.

  1. archive  the earlier GLOBAL-focus datasets (one focus for all poses), their
              trained models and configs: they are moved to --archive (outside
              the repository by default), so that their names are free.
              The reference datasets slitsim_<scene> (area / reference focus of
              all designs) are kept.
  2. rename   the per-pose-focus datasets, models and configs from *_pf* to the
              name without _pf (e.g. slitsim_real_cake_n4_a0.40_cyc_pf ->
              slitsim_real_cake_n4_a0.40_cyc, disksim_real_cake_pf_glayer ->
              disksim_real_cake_glayer); the name is also replaced inside their
              small text files (configs, sim_gt.json, summaries).
  3. update   config_files/scene_info.yaml (backup in the archive folder).

Logs and earlier summary reports are left as they are.

Dry run by default (prints the plan); --apply executes it.

Usage (from the repository root):
    python tools/migrate_focus_names.py                 # show the plan
    python tools/migrate_focus_names.py --apply         # archive + rename
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

PF = re.compile(r"_pf(?=_|$)")                  # the _pf token of a dataset name
TEXT_EXT = {".yaml", ".yml", ".json", ".txt", ".csv", ".md"}
TEXT_NAMES = {"cfg_args"}
SKIP_DIRS = {"point_cloud", "strips", "images", "aif", "defocus", "sparse", "depth"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="./data/slit_sim")
    p.add_argument("--model_root", default="./output/model_dofgs")
    p.add_argument("--config_dir", default="./config_files")
    p.add_argument("--archive", default=os.path.join(os.path.dirname(REPO_ROOT),
                                                     os.path.basename(REPO_ROOT) + "_global_focus_archive"),
                   help="folder for the global-focus data (default: next to the repository)")
    p.add_argument("--apply", action="store_true", help="execute the plan (default: dry run)")
    return p.parse_args()


def load_scene_info(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def is_reference(name, scene_info):
    """slitsim_<source scene>: reference dataset of a source DOF-GS scene."""
    if not name.startswith("slitsim_"):
        return False
    src = name[len("slitsim_"):]
    entry = scene_info.get(src)
    return isinstance(entry, dict) and "slit_sim" not in str(entry.get("source_path", ""))


def focus_kind(ds_dir):
    try:
        with open(os.path.join(ds_dir, "sim_gt.json")) as f:
            gt = json.load(f)
    except (OSError, ValueError):
        return None
    if gt.get("focus_mode") == "per_pose" or gt.get("focus_per_group"):
        return "per_pose"
    return "global"


def replace_name(text, old, new):
    """Replace the dataset name old by new (not inside a longer name; config_<old>_<variant> too)."""
    text = text.replace(f"config_{old}_", f"config_{new}_")      # variant configs config_<name>_<parts>.yaml
    return re.sub(re.escape(old) + r"(?![A-Za-z0-9_])", new, text)


def text_files(root, recursive):
    if os.path.isfile(root):
        return [root]
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS] if recursive else []
        for fn in filenames:
            if os.path.splitext(fn)[1] in TEXT_EXT or fn in TEXT_NAMES:
                fp = os.path.join(dirpath, fn)
                if os.path.getsize(fp) <= 5 * 2 ** 20:
                    out.append(fp)
    return out


def rewrite(files, old, new, apply):
    n = 0
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                s = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        t = replace_name(s, old, new)
        if t != s:
            n += 1
            if apply:
                with open(fp, "w", encoding="utf-8") as f:
                    f.write(t)
    return n


def config_scene(path):
    try:
        with open(path) as f:
            c = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    sn = c.get("scene_name")
    return sn[0] if isinstance(sn, list) and sn else sn


def main():
    a = parse_args()
    info_path = os.path.join(a.config_dir, "scene_info.yaml")
    scene_info = load_scene_info(info_path)

    archive, rename, keep_ref = [], {}, []
    for d in sorted(glob.glob(os.path.join(a.data_root, "*"))):
        name = os.path.basename(d)
        if not os.path.isfile(os.path.join(d, "sim_gt.json")):
            continue
        if is_reference(name, scene_info):
            keep_ref.append(name)
            continue
        kind = focus_kind(d)
        if kind == "global":
            archive.append(name)
        elif kind == "per_pose" and PF.search(name):
            rename[name] = PF.sub("", name)

    # name clashes that the archive does not resolve
    existing = {os.path.basename(d) for d in glob.glob(os.path.join(a.data_root, "*"))}
    clashes = [(o, n) for o, n in rename.items() if n in existing and n not in archive]
    new_names = list(rename.values())
    dup = {n for n in new_names if new_names.count(n) > 1}
    if clashes or dup:
        for o, n in clashes:
            print(f"[ERROR] {o} -> {n}: {n} exists and is not a global-focus dataset")
        for n in dup:
            print(f"[ERROR] several datasets would be renamed to {n}")
        sys.exit("Nothing changed. Resolve the clashes first.")

    model_dirs = lambda n: sorted(glob.glob(os.path.join(a.model_root, "*", n)))
    configs = {}
    for c in sorted(glob.glob(os.path.join(a.config_dir, "config_*.yaml"))):
        sc = config_scene(c)
        if sc is not None:
            configs.setdefault(sc, []).append(c)

    print(f"Reference datasets kept: {len(keep_ref)}" + (f" ({', '.join(keep_ref)})" if keep_ref else ""))
    print(f"\n== archive {len(archive)} global-focus dataset(s) -> {a.archive}")
    for n in archive:
        print(f"  {n}: dataset, {len(model_dirs(n))} model(s), {len(configs.get(n, []))} config(s)")
    print(f"\n== rename {len(rename)} per-pose dataset(s)")
    for o, n in rename.items():
        print(f"  {o} -> {n}: dataset, {len(model_dirs(o))} model(s), {len(configs.get(o, []))} config(s)")
    if not a.apply:
        print("\nDry run: nothing changed. Run again with --apply to execute.")
        return

    # ---- 1. archive ----
    def move(src):
        dst = os.path.join(a.archive, os.path.relpath(src, REPO_ROOT))
        if os.path.exists(dst):
            sys.exit(f"[ERROR] {dst} exists in the archive; stopped (partially applied, rerun after checking).")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)

    os.makedirs(os.path.join(a.archive, "config_files"), exist_ok=True)
    shutil.copy2(info_path, os.path.join(a.archive, "config_files", "scene_info.yaml.bak"))
    for n in archive:
        move(os.path.join(a.data_root, n))
        for m in model_dirs(n):
            move(m)
        for c in configs.get(n, []):
            move(c)

    # ---- 2. rename ----
    edited = 0
    for o, n in rename.items():
        src = os.path.join(a.data_root, o)
        dst = os.path.join(a.data_root, n)
        os.rename(src, dst)
        edited += rewrite(text_files(dst, recursive=False), o, n, True)
        for m in model_dirs(o):
            m_new = os.path.join(os.path.dirname(m), n)
            os.rename(m, m_new)
            edited += rewrite(text_files(m_new, recursive=True), o, n, True)
        for c in configs.get(o, []):
            base = os.path.basename(c)                  # config_<old>.yaml or config_<old>_<parts>.yaml
            c_new = os.path.join(os.path.dirname(c), f"config_{n}" + base[len(f"config_{o}"):]
                                 if base.startswith(f"config_{o}") else base)
            os.rename(c, c_new)
            edited += rewrite([c_new], o, n, True)

    # ---- 3. scene_info.yaml (line based, keeps the layout) ----
    with open(info_path) as f:
        lines = f.read().split("\n")
    out, drop = [], False
    for ln in lines:
        m = re.match(r"^([A-Za-z0-9_.=\-]+)\s*:", ln)
        if m:
            key = m.group(1)
            drop = key in archive
            if key in rename:
                ln = replace_name(ln, key, rename[key])
        elif drop:
            continue
        else:
            for o, n in rename.items():
                if o in ln:
                    ln = replace_name(ln, o, n)
        if not drop:
            out.append(ln)
    with open(info_path, "w") as f:
        f.write("\n".join(out))

    print(f"\nDone: {len(archive)} dataset(s) archived, {len(rename)} renamed, {edited} text file(s) updated.")
    print(f"scene_info.yaml backup: {os.path.join(a.archive, 'config_files', 'scene_info.yaml.bak')}")


if __name__ == "__main__":
    main()
