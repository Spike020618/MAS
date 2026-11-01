"""Symmetric exact-metadata echo projection; raw responses remain untouched."""
import copy
import json

from experiments.consensus_recommendation.common import fingerprint


class SchemaEchoClient:
    def __init__(self, client):
        self.client = client
        self.ignored_echoes = []

    @property
    def limits(self): return self.client.limits

    def action_fits(self, *args, **kwargs): return self.client.action_fits(*args, **kwargs)

    def summary(self): return self.client.summary()

    def account_shared(self, **kwargs): return self.client.account_shared(**kwargs)

    def complete(self, slot, request):
        record = self.client.complete(slot, request)
        payload = json.loads(request["messages"][1]["content"])
        template, report = payload.get("schema"), record["report"]
        if (isinstance(template, dict) and isinstance(report, dict)
                and set(report) == set(template) | {"schema"} and report.get("schema") == template):
            projected = copy.deepcopy(record)
            projected["report"].pop("schema")
            self.ignored_echoes.append({"slot": slot, "request_hash": fingerprint(request),
                "raw_report_hash": fingerprint(report), "projected_report_hash": fingerprint(projected["report"]),
                "operation": "remove exact request-template metadata echo only"})
            return projected
        return record
