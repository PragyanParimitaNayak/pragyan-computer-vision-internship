import base64, json
from http.client import HTTPConnection
from pathlib import Path

image_path = Path("sample_shelf.png")
image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

body = json.dumps({
    "image_base64": image_b64,
    "shelf_id": "A-01"
}).encode("utf-8")

conn = HTTPConnection("127.0.0.1", 8000, timeout=10)
conn.request(
    "POST", "/analyze",
    body=body,
    headers={"Content-Type": "application/json"}
)
response = conn.getresponse()
print(response.status)
print(response.read().decode("utf-8"))
conn.close()
