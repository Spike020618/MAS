import http.client
import json
import threading
import unittest

from game_consensus_recsys.http_server import ConsensusHTTPServer
from game_consensus_recsys.workflow import ConsensusWorkflow


class HTTPServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Unit tests must never discover a developer .env or spend API quota.
        cls.server = ConsensusHTTPServer(("127.0.0.1", 0), workflow=ConsensusWorkflow())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, payload=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if body else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = json.loads(response.read().decode())
        connection.close()
        return response.status, data

    def test_health(self):
        status, data = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")

    def test_search_without_key(self):
        status, data = self.request("POST", "/v1/search", {"query": "科幻电影", "top_k": 2})
        self.assertEqual(status, 200)
        self.assertEqual(len(data["decisions"]), 2)

    def test_unknown_field_is_rejected(self):
        status, data = self.request("POST", "/v1/search", {
            "query": "科幻电影", "secret": "must-not-pass"
        })
        self.assertEqual(status, 422)
        self.assertEqual(data["error"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
