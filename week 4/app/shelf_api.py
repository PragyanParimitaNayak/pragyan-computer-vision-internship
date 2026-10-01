
"""
YuvaIntern Week 4
Pragyan Parimita Nayak
Real-Time Retail Shelf Object Detection and Counting
"""
import base64, binascii, io, json, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image
from torch import nn

IMG, GRID = 128, 8
CELL = IMG / GRID


class TinyShelfDetector(nn.Module):
    def __init__(self, channels=(12, 24, 48, 72)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Conv2d(c4, 5, 1)

    def forward(self, x):
        return self.head(self.features(x))


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = list(image.size)
    resized = image.resize((IMG, IMG))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor, original_size


def decode_detections(prediction, confidence_threshold=0.50):
    objectness = torch.sigmoid(prediction[0])
    detections = []

    for gy in range(GRID):
        for gx in range(GRID):
            confidence = float(objectness[gy, gx])
            if confidence < confidence_threshold:
                continue

            tx, ty, tw, th = torch.sigmoid(
                prediction[1:, gy, gx]
            ).tolist()

            cx = ((gx + tx) * CELL) / IMG
            cy = ((gy + ty) * CELL) / IMG
            width = max(0.01, min(0.50, tw))
            height = max(0.01, min(0.50, th))

            x1 = max(0.0, min(1.0, cx - width / 2))
            y1 = max(0.0, min(1.0, cy - height / 2))
            x2 = max(0.0, min(1.0, cx + width / 2))
            y2 = max(0.0, min(1.0, cy + height / 2))

            detections.append({
                "class": "object",
                "confidence": round(confidence, 4),
                "bbox_normalized": [
                    round(x1, 4), round(y1, 4),
                    round(x2, 4), round(y2, 4)
                ],
            })

    return detections


class InventoryAdapter:
    """Simulated downstream inventory-monitoring component."""
    def __init__(self, minimum_expected_count=5):
        self.minimum_expected_count = minimum_expected_count

    def build_event(self, shelf_id, detection_count, inference_ms):
        return {
            "event_type": "SHELF_COUNT_UPDATE",
            "shelf_id": shelf_id,
            "detected_count": detection_count,
            "minimum_expected_count": self.minimum_expected_count,
            "review_required": detection_count < self.minimum_expected_count,
            "inference_ms": round(inference_ms, 3),
        }


class ShelfMonitoringService:
    def __init__(self, model_path, minimum_expected_count=5):
        self.model = TinyShelfDetector()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        torch.set_num_threads(4)
        self.inventory = InventoryAdapter(minimum_expected_count)

    def analyze(self, image_bytes, shelf_id="demo-shelf"):
        tensor, original_size = preprocess_image(image_bytes)

        start = time.perf_counter()
        with torch.inference_mode():
            prediction = self.model(tensor)[0]
        inference_ms = (time.perf_counter() - start) * 1000.0

        detections = decode_detections(prediction)
        event = self.inventory.build_event(
            shelf_id, len(detections), inference_ms
        )

        return {
            "request_id": str(uuid.uuid4()),
            "status": "ok",
            "model": "week3_pruned_tinyshelf_detector",
            "input": {
                "original_size": original_size,
                "processed_size": [IMG, IMG],
            },
            "detection_count": len(detections),
            "detections": detections,
            "inference_ms": round(inference_ms, 3),
            "downstream_event": event,
        }


class RequestHandler(BaseHTTPRequestHandler):
    service = None

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "service": "retail-shelf-monitor"
            })
            return
        self._send_json(404, {
            "status": "error",
            "message": "Endpoint not found"
        })

    def do_POST(self):
        if self.path != "/analyze":
            self._send_json(404, {
                "status": "error",
                "message": "Endpoint not found"
            })
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                self._send_json(400, {
                    "status": "error",
                    "message": "Request body is required"
                })
                return

            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image_b64 = payload.get("image_base64")
            if not image_b64:
                raise ValueError("image_base64 is required")

            image_bytes = base64.b64decode(image_b64, validate=True)
            shelf_id = payload.get("shelf_id", "demo-shelf")
            result = self.service.analyze(image_bytes, shelf_id)
            self._send_json(200, result)

        except (ValueError, json.JSONDecodeError, binascii.Error) as exc:
            self._send_json(400, {
                "status": "error",
                "message": str(exc)
            })
        except Exception as exc:
            self._send_json(500, {
                "status": "error",
                "message": "Internal processing error",
                "detail": str(exc)
            })

    def log_message(self, format, *args):
        return


def create_server(model_path, host="127.0.0.1", port=0):
    RequestHandler.service = ShelfMonitoringService(model_path)
    return ThreadingHTTPServer((host, port), RequestHandler)


if __name__ == "__main__":
    server = create_server(
        "../pruned_model/optimized_pruned_shelf_detector.pt", port=8000
    )
    print("Retail Shelf Monitoring API: http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
