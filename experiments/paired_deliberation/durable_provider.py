"""V5 raw-first checkpoints and deterministic fresh/resumed return values."""
import json
import os
from pathlib import Path

from experiments.consensus_recommendation.common import create, fingerprint
from experiments.paired_deliberation.verified_state import canonical


class DurableProvider:
    def __init__(self, root, manifest, transport, parser, authorized=False, reuse_initial=None, reuse_all=False):
        self.root = Path(root)
        self.manifest = json.loads(canonical(manifest))
        self.transport, self.parser, self.authorized = transport, parser, authorized
        self.reuse_initial = Path(reuse_initial) if reuse_initial else None
        self.reuse_all = reuse_all

    def complete(self, slot, request):
        if not isinstance(slot, str) or not slot or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in slot):
            raise ValueError("unsafe slot")
        request = json.loads(canonical(request))
        dependency = fingerprint({"manifest": self.manifest, "request": request, "slot": slot})
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (slot + ".json")
        raw_path = self.root / (slot + ".raw.json")
        intent = self.root / (slot + ".intent.json")
        if (not path.exists() and not raw_path.exists() and self.reuse_initial
                and (self.reuse_all or request.get("kind") == "initial")):
            old_path = self.reuse_initial / (slot + ".raw.json")
            if old_path.exists():
                old = json.loads(old_path.read_text())
                if (old["request"] == request and old["integrity"] == fingerprint(old["raw"])
                        and old["raw"]["status_code"] == 200):
                    create(raw_path, {"dependency_hash": dependency, "request": request,
                                      "raw": old["raw"], "integrity": old["integrity"],
                                      "reused_from": str(old_path)})
        if path.exists():
            saved = json.loads(path.read_text())
            self._check(saved, dependency)
            if saved["integrity"] != fingerprint(saved["record"]):
                raise ValueError("tampered parsed checkpoint")
            return saved["record"]
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
            self._check(raw, dependency)
            if raw["integrity"] != fingerprint(raw["raw"]):
                raise ValueError("tampered raw checkpoint")
        else:
            if intent.exists():
                raise RuntimeError("pending request; billing unknown; automatic repeat forbidden")
            if not self.authorized:
                raise RuntimeError("network transport not authorized")
            if hasattr(self.transport, "preflight"):
                self.transport.preflight(request)
            try:
                fd = os.open(intent, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                raise RuntimeError("another collector owns this slot") from None
            with os.fdopen(fd, "w") as stream:
                stream.write(canonical({"dependency_hash": dependency, "request": request}))
                stream.flush()
                os.fsync(stream.fileno())
            response = self.transport(request)
            raw = {"dependency_hash": dependency, "request": request,
                   "raw": json.loads(canonical(response)), "integrity": fingerprint(response)}
            create(raw_path, raw)
            intent.unlink()
        # The paid raw response is durable before any schema/semantic parsing.
        record = json.loads(canonical(self.parser(raw["raw"], request)))
        create(path, {"dependency_hash": dependency, "record": record,
                      "integrity": fingerprint(record)})
        return record

    @staticmethod
    def _check(saved, dependency):
        if saved["dependency_hash"] != dependency:
            raise ValueError("request or manifest changed; use a new experiment version")
