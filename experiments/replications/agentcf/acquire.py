"""Acquire a small, hash-verified upstream source/data snapshot; never execute it.

Git blob identity is checked before saving. No models, secrets, or pickle files.
"""
import argparse
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

COMMIT = "71ff3b5bce69432556884c49fe7f19528b6756ec"
REPO = "leoleojie/AgentCF-WWW"


def selected(path):
    if path in {"README.md", "LICENSE"}:
        return True
    if path.startswith("agentcf/props/") and path.endswith(".yaml"):
        return True
    if path.startswith("agentcf/") and path.endswith(".py"):
        return True
    if path.startswith("agentcf/dataset/CDs-100-user-"):
        return path.endswith((".inter", ".random", "CDs.item")) and "/saved/" not in path
    return False


def acquire(tree_path, destination):
    tree = json.loads(Path(tree_path).read_text())
    if tree["sha"] != COMMIT or tree.get("truncated"):
        raise ValueError("Tree identity mismatch or truncated listing")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    (root / "upstream_tree.json").write_text(json.dumps(tree, indent=2) + "\n")

    def fetch(entry):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe upstream path")
        target = root / "upstream" / relative
        if target.exists():
            data = target.read_bytes()
        else:
            req = Request(entry["url"], headers={"User-Agent": "MAS-replication-audit"})
            with urlopen(req, timeout=45) as response:
                blob = json.load(response)
            data = base64.b64decode(blob["content"])
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != entry["sha"] or len(data) != entry["size"]:
            raise ValueError("Blob verification failed: " + str(relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        return {"path": str(relative), "git_blob": digest, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}

    entries = [e for e in tree["tree"] if e["type"] == "blob" and selected(e["path"])]
    # A failed acquisition leaves verified files for resume but no success manifest.
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, entries))
    manifest = {"repository": "https://github.com/" + REPO, "commit": COMMIT,
                "scope": "source and plain-text CDs splits only; not a complete repository",
                "files": sorted(files, key=lambda x: x["path"]), "model_calls": 0}
    (root / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"verified_files": len(files), "bytes": sum(f["bytes"] for f in files)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    acquire(args.tree, args.destination)
