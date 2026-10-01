
import base64, json, threading, time, unittest
from http.client import HTTPConnection
from pathlib import Path

from app.shelf_api import create_server, preprocess_image, InventoryAdapter

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "pruned_model" / "optimized_pruned_shelf_detector.pt"
SAMPLE = ROOT / "sample_shelf.png"


class Week4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(str(MODEL), port=0)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.thread.start()
        time.sleep(0.10)
        cls.image_b64 = base64.b64encode(
            SAMPLE.read_bytes()
        ).decode("ascii")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)
        cls.server.server_close()

    def request(self, method, path, payload=None):
        conn = HTTPConnection(self.host, self.port, timeout=10)
        if payload is None:
            conn.request(method, path)
        else:
            body = json.dumps(payload).encode("utf-8")
            conn.request(method, path, body=body,
                         headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, data

    def test_preprocess(self):
        tensor, original = preprocess_image(SAMPLE.read_bytes())
        self.assertEqual(tuple(tensor.shape), (1, 3, 128, 128))
        self.assertEqual(original, [128, 128])

    def test_health(self):
        status, data = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")

    def test_missing_body(self):
        status, data = self.request("POST", "/analyze", {})
        self.assertEqual(status, 400)
        self.assertEqual(data["status"], "error")

    def test_invalid_base64(self):
        status, data = self.request(
            "POST", "/analyze",
            {"image_base64": "not-valid-base64", "shelf_id": "A-01"}
        )
        self.assertEqual(status, 400)

    def test_end_to_end_analyze(self):
        status, data = self.request(
            "POST", "/analyze",
            {"image_base64": self.image_b64, "shelf_id": "A-01"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["input"]["processed_size"], [128, 128])
        self.assertIsInstance(data["detection_count"], int)
        self.assertIn("downstream_event", data)
        self.assertEqual(
            data["downstream_event"]["event_type"],
            "SHELF_COUNT_UPDATE"
        )
        self.assertEqual(data["downstream_event"]["shelf_id"], "A-01")


    def test_inventory_adapter_low_count(self):
        adapter = InventoryAdapter(minimum_expected_count=5)
        event = adapter.build_event("A-02", 3, 2.5)
        self.assertTrue(event["review_required"])
        self.assertEqual(event["event_type"], "SHELF_COUNT_UPDATE")
        self.assertEqual(event["detected_count"], 3)

    def test_unknown_route(self):
        status, data = self.request("GET", "/unknown")
        self.assertEqual(status, 404)
        self.assertEqual(data["status"], "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
