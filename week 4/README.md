# YuvaIntern Week 4 - Pragyan Parimita Nayak

## Project
Real-Time Retail Shelf Object Detection and Counting

## Week 4 focus
Integration of the Week 3 optimized detector into a small cross-functional prototype.

## Architecture
Client -> REST API -> Image Preprocessing -> Optimized CV Model -> Detection Decoder -> Inventory Adapter

## Endpoints
GET /health
POST /analyze

`/analyze` accepts a JSON body with a base64-encoded image and optional `shelf_id`.
It returns detection count, normalized bounding boxes, inference time and a simulated
`SHELF_COUNT_UPDATE` downstream event.

## Verified tests
7 automated tests passed.

20 live `/analyze` requests were also sent through the local HTTP server.
The benchmark record is in `api_benchmark.json`.

## Scope note
The trained model comes from the Week 3 synthetic-data prototype. The integration
demonstrates software architecture and end-to-end communication; it does not claim
production accuracy on real retail shelves or a full SKU-110K benchmark.
