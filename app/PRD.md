# Product Requirements Document (PRD)

## 1. Product Summary
A lightweight web application for real‑time (image + short video) vehicle licence plate detection using the trained YOLOv8 model already produced in this repository.

## 2. Goals
- Allow user to upload an image (JPG/PNG/WebP) and instantly receive annotated output.
- Allow user to upload a short video (<= ~10s or capped at 300 frames) and receive an annotated video.
- Expose a clean REST API (`/infer`) + a single-page HTML UI.
- Reuse existing trained weights at a fixed, transparent path.

## 3. Non‑Goals
- OCR / reading plate text.
- Multi-class detection beyond the single `licence` class.
- Long video streaming / RTSP ingestion.

## 4. Users
- Internal testers / developers validating model performance.
- Demo stakeholders.

## 5. Key Features & Requirements
| Feature | Requirement | Priority |
|---------|-------------|----------|
| Single model load | Cache YOLO model once per process | P0 |
| Image inference | < 1s for typical 640px image on CPU/GPU | P0 |
| Video inference | Frame-by-frame annotate; cap frames at 300 | P0 |
| Confidence control | Adjustable via form input | P1 |
| Return format | JSON with base64 annotated media | P0 |
| Front-end | Responsive dark theme single page | P1 |
| CORS | Allow all origins | P2 |

## 6. Model
Absolute path persisted in `memory.md` as:
`/home/naki/Desktop/Y'all aint seen shit/YOLO Vehicle number plate /runs_yolo/plate_det_v8n/weights/best.pt`
Fallback: `last.pt` (same directory) if needed.

## 7. API
- `GET /` -> HTML SPA
- `POST /infer` -> multipart form (`file`, `conf`) returns JSON:
```
{
  "type": "image" | "video",
  "detections": <int>,
  "annotated": <base64 string>
}
```
Errors -> `{ "error": "message" }` with appropriate HTTP code.

## 8. Performance Constraints
- Memory: < 1GB typical.
- Video: processed sequentially; no async GPU batching.

## 9. Security & Privacy
- No persistent storage of uploads (transient in-memory / temp). Files overwritten.
- No user auth (controlled environment assumption).

## 10. Observability
- Basic console logs only.

## 11. Risks / Mitigations
| Risk | Mitigation |
|------|------------|
| Large video upload stalls | Hard frame cap (300) & rely on client to limit duration |
| Missing weights | Explicit FileNotFoundError on startup |
| Memory leak on repeated loads | LRU cache single load |

## 12. Acceptance Criteria
- Startup shows model path in UI.
- Upload image -> annotated response appears with at least one detection (if present).
- Upload short video -> annotated mp4 playable in browser.
- Adjusting confidence threshold changes results (fewer boxes at higher conf).

## 13. Future Enhancements
- WebSocket streaming for webcam.
- OCR integration.
- Multi-class extension.
- GPU utilization metrics.

## 14. Dev Environment
Install requirements: `pip install -r app/requirements.txt`
Run: `uvicorn app.main:app --reload`

## 15. Ownership
- Model: ML engineer.
- API & UI: Full-stack engineer.

(End)
