import copy
import unittest

from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.paired_deliberation.schema_echo_client import SchemaEchoClient


class EchoTests(unittest.TestCase):
    def run_record(self, value):
        class Client:
            def complete(self, slot, request):
                return {"report": copy.deepcopy(value), "usage": {"prompt_tokens": 10}}
        client = SchemaEchoClient(Client())
        request = RequestsV5.envelope("anything", {"schema": {"ranking": ["five IDs"]}}, "Return JSON.")
        return client.complete("slot", request), client

    def test_only_exact_additional_template_is_ignored(self):
        raw = {"ranking": ["1"], "schema": {"ranking": ["five IDs"]}}
        got, client = self.run_record(raw)
        self.assertEqual(got["report"], {"ranking": ["1"]})
        self.assertIn("schema", raw)
        self.assertEqual(len(client.ignored_echoes), 1)

    def test_missing_fields_wrappers_and_different_templates_stay_invalid(self):
        for value in [{"schema": {"ranking": ["1"]}}, {"ranking": ["1"], "schema": {}},
                      {"ranking": ["1"], "schema": {"ranking": ["five IDs"]}, "extra": 1}]:
            got, client = self.run_record(value)
            self.assertEqual(got["report"], value)
            self.assertEqual(client.ignored_echoes, [])


if __name__ == "__main__":
    unittest.main()
