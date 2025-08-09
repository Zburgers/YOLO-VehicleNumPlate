import io
import cv2
import base64
import numpy as np
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from .model_loader import get_model, MODEL_WEIGHTS_PATH
from loguru import logger
import time
import tempfile
import uuid
import os
import shutil, subprocess  # added

# Configure loguru (could also be configured via env vars)
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")
logger.add("server.log", rotation="10 MB", retention="10 days", level="DEBUG", enqueue=True)

app = FastAPI(title="Vehicle Number Plate Detection", description="YOLOv8 licence plate detector", version="1.1")

# Request/response timing middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    request_id = f"{int(start*1000)}-{uuid.uuid4().hex[:6]}"
    request.state.request_id = request_id
    logger.bind(request_id=request_id).info(
        f"➡️  {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}"
    )
    try:
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        logger.bind(request_id=request_id).info(
            f"✅ {request.method} {request.url.path} completed in {duration:.1f} ms status={response.status_code}"
        )
        return response
    except Exception as e:
        duration = (time.time() - start) * 1000
        logger.bind(request_id=request_id).exception(
            f"❌ {request.method} {request.url.path} failed after {duration:.1f} ms error={e}"
        )
        raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / 'static'
STATIC_DIR.mkdir(exist_ok=True)
VIDEOS_DIR = STATIC_DIR / 'videos'
VIDEOS_DIR.mkdir(exist_ok=True)
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

# Helper to write video with fallbacks
# (Replaced with ffmpeg-first streaming implementation for better browser compatibility)
FFMPEG_BIN = shutil.which("ffmpeg")

def _ensure_even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    new_h = h - (h % 2)
    new_w = w - (w % 2)
    if new_h != h or new_w != w:
        return frame[0:new_h, 0:new_w]
    return frame

def write_video_opencv(frames, fps: float, stem: str) -> tuple[Path, str]:
    h, w = frames[0].shape[:2]
    tried = []
    
    # Try web-compatible codecs in order of preference - MP4 first, then fallback
    codecs_to_try = [
        ("mp4v", "MPEG-4 Part 2", ".mp4"),
        ("XVID", "Xvid", ".mp4"),
        ("MJPG", "MotionJPEG", ".mp4"),  # Try MJPG in MP4 container first
        ("MJPG", "MotionJPEG-AVI", ".avi")  # Only use AVI as last resort
    ]
    
    for tag, note, ext in codecs_to_try:
        out_path = VIDEOS_DIR / f"{stem}{ext}"
        fourcc = cv2.VideoWriter_fourcc(*tag)
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        
        if not vw.isOpened():
            tried.append(f"{tag}:{note}:open_fail")
            vw.release()
            continue
        
        # Write frames
        logger.debug(f"Writing {len(frames)} frames with codec {tag} to {ext}")
        success = True
        for i, fr in enumerate(frames):
            if not vw.write(fr):
                logger.warning(f"Failed to write frame {i} with codec {tag}")
                success = False
                break
        
        vw.release()
        
        if success and out_path.exists():
            size = out_path.stat().st_size
            if size > 1024:  # At least 1KB
                logger.info(f"Video (OpenCV) written using codec {tag} format={ext} size_bytes={size}")
                return out_path, f"{tag}{ext}"
            else:
                tried.append(f"{tag}:{note}:size={size}")
                try:
                    out_path.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            tried.append(f"{tag}:{note}:write_fail")
    
    logger.error(f"All OpenCV encoders failed tried={tried}")
    raise RuntimeError("Failed to encode video with available OpenCV codecs")

def write_video_stream_ffmpeg(frame_iter, fps: float, stem: str, frame_shape: tuple[int, int]) -> tuple[Path, str]:
    if not FFMPEG_BIN:
        raise RuntimeError("ffmpeg binary not found")
    h, w = frame_shape
    out_path = VIDEOS_DIR / f"{stem}.mp4"
    
    # Build ffmpeg command with web-optimized settings
    cmd = [
        FFMPEG_BIN, '-loglevel', 'warning', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', f'{fps}', '-i', '-',
        '-an',  # no audio
        '-c:v', 'libx264',  # H.264 codec - most compatible
        '-preset', 'medium',  # better quality than fast
        '-pix_fmt', 'yuv420p',  # compatible pixel format
        '-movflags', '+faststart',  # optimize for web streaming
        '-crf', '23',  # good quality/size balance
        '-profile:v', 'baseline',  # most compatible H.264 profile
        '-level', '3.0',  # compatible level
        '-maxrate', '2M',  # limit bitrate
        '-bufsize', '4M',  # buffer size
        str(out_path)
    ]
    logger.debug(f"Running ffmpeg: {' '.join(cmd)}")
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_count = 0
    
    try:
        for fr in frame_iter:
            frame_count += 1
            if proc.stdin:
                proc.stdin.write(fr.tobytes())
            if frame_count % 50 == 0:  # Log progress every 50 frames
                logger.debug(f"Encoded {frame_count} frames with ffmpeg")
    except BrokenPipeError:
        logger.warning("ffmpeg process terminated early")
    finally:
        if proc.stdin:
            proc.stdin.close()
        stderr_output = ""
        if proc.stderr:
            stderr_output = proc.stderr.read().decode('utf-8')
        ret = proc.wait()
    
    if ret != 0:
        logger.error(f"ffmpeg failed with return code {ret}: {stderr_output}")
        raise RuntimeError(f"ffmpeg failed ret={ret}: {stderr_output}")
    
    if not out_path.exists():
        logger.error("ffmpeg did not create output file")
        raise RuntimeError("ffmpeg did not create output file")
    
    file_size = out_path.stat().st_size
    if file_size < 1024:  # Less than 1KB is likely an error
        logger.error(f"ffmpeg output file too small: {file_size} bytes")
        raise RuntimeError(f"ffmpeg output file too small: {file_size} bytes")
    
    logger.info(f"Video (ffmpeg/H.264) written frames={frame_count} size_kb={file_size/1024:.1f}")
    return out_path, 'H.264'

# Public helper used by inference branch

def write_video(frames_or_iter, fps: float, stem: str, frame_shape: tuple[int,int]|None=None, streaming: bool=False) -> tuple[Path,str]:
    """Write video using ffmpeg if available, else OpenCV.
    frames_or_iter: list of frames (BGR) or iterator yielding frames.
    If streaming=True and ffmpeg present, frames_or_iter should be an iterator and frame_shape must be given.
    """
    if streaming and FFMPEG_BIN:
        try:
            return write_video_stream_ffmpeg(frames_or_iter, fps, stem, frame_shape)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"ffmpeg streaming failed: {e}; falling back to OpenCV buffered encoding")
    # If we get here we need a list of frames
    if not isinstance(frames_or_iter, list):
        frames = list(frames_or_iter)
    else:
        frames = frames_or_iter
    if not frames:
        raise RuntimeError("No frames to encode")
    return write_video_opencv(frames, fps, stem)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8' />
<title>Vehicle Number Plate Detection</title>
<meta name='viewport' content='width=device-width, initial-scale=1' />
<style>
body { font-family: system-ui, Arial, sans-serif; margin:0; background:#0f1115; color:#e9eef3; }
header { padding:1.2rem 2rem; background:#161a21; box-shadow:0 2px 4px rgba(0,0,0,.4); }
main { max-width:1100px; margin:1.5rem auto 3rem; padding:0 1rem; }
.card { background:#1e2430; border-radius:14px; padding:1.5rem 1.5rem 2.2rem; box-shadow:0 4px 10px rgba(0,0,0,.35); }
label { display:block; margin-top:1rem; font-weight:600; letter-spacing:.5px; }
input[type=file] { margin-top:.4rem; }
button { margin-top:1.2rem; background:#2563eb; color:#fff; border:none; padding:.8rem 1.4rem; font-size:1rem; border-radius:8px; cursor:pointer; font-weight:600; letter-spacing:.5px; }
button:hover { background:#1d4ed8; }
#result-grid { margin-top:2rem; display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:1.5rem; }
.panel { background:#11151c; padding:1rem; border-radius:10px; position:relative; }
.panel h3 { margin-top:0; font-size:1rem; font-weight:600; text-transform:uppercase; letter-spacing:.75px; color:#7dd3fc; }
img, video { max-width:100%; border-radius:6px; }
.badge { position:absolute; top:10px; right:10px; background:#334155; padding:.35rem .6rem; font-size:.7rem; text-transform:uppercase; border-radius:6px; letter-spacing:.75px; color:#e2e8f0; }
footer { text-align:center; font-size:.75rem; color:#64748b; margin:2rem 0; }
.progress { width:100%; height:6px; background:#334155; border-radius:4px; margin-top:1rem; overflow:hidden; display:none; }
.progress span { display:block; height:100%; width:0%; background:#06b6d4; animation:load 2s linear infinite; }
@keyframes load { 0% { transform:translateX(-100%); } 100% { transform:translateX(100%); } }
.code { font-family:monospace; font-size:.8rem; color:#94a3b8; margin-top:.75rem; word-break:break-all; }
.bad { color:#f87171; }
.stats { font-size:.75rem; margin-top:.5rem; color:#94a3b8; }
video { background:#000; }
.notice { font-size:.7rem; color:#64748b; margin-top:.4rem; }
</style>
</head>
<body>
<header><h1 style='margin:0;font-size:1.35rem;'>Vehicle Number Plate Detection</h1></header>
<main>
  <div class='card'>
    <p>Upload an image or a short video. The model will return an annotated media with detected licence plates.</p>
    <p class='code'>Model weights path: {model_path}</p>
    <form id='upload-form'>
      <label>Media (image/video)</label>
      <input id='file' name='file' type='file' accept='image/*,video/*' required />
      <label>Confidence Threshold</label>
      <input id='conf' name='conf' type='number' step='0.01' value='0.25' min='0' max='1' />
      <button type='submit'>Run Detection</button>
      <div class='progress' id='progress'><span></span></div>
    </form>
    <div class='stats' id='stats'></div>
    <div id='result-grid'></div>
  </div>
  <footer>Powered by FastAPI + YOLOv8</footer>
</main>
<script>
const form = document.getElementById('upload-form');
const resultGrid = document.getElementById('result-grid');
const progress = document.getElementById('progress');
const statsEl = document.getElementById('stats');
form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const fileInput = document.getElementById('file');
  if (!fileInput.files.length) return;
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  fd.append('conf', document.getElementById('conf').value || '0.25');
  progress.style.display = 'block';
  resultGrid.innerHTML = '';
  statsEl.textContent = 'Processing...';
  const t0 = performance.now();
  try {
    const res = await fetch('/infer', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Request failed: '+res.status);
    const data = await res.json();
    console.log('Response data:', data);  // Debug log
    
    const panel = document.createElement('div');
    panel.className = 'panel';
    panel.innerHTML = '<h3>Result</h3>';
    
    if (data.type === 'image') {
      panel.innerHTML += `<img src="data:image/png;base64,${data.annotated}" alt='annotated' />`;
    } else if (data.type === 'video') {
      if (data.video_url) {
        const videoUrl = data.video_url + '?t=' + Date.now();
        console.log('Loading video from:', videoUrl);  // Debug log
        
        const video = document.createElement('video');
        video.id = 'resultVideo';
        video.controls = true;
        video.autoplay = false;  // Don't autoplay to avoid issues
        video.muted = true;
        video.playsInline = true;
        video.preload = 'metadata';
        video.style.width = '100%';
        video.style.borderRadius = '6px';
        video.src = videoUrl;
        
        // Add comprehensive error handling
        video.onerror = function(e) {
          console.error('Video error:', e);
          console.error('Video error details:', e.target.error);
          const errorDiv = document.createElement('div');
          errorDiv.className = 'bad';
          errorDiv.innerHTML = `Video playback error: ${e.target.error?.message || 'Unknown error'}<br>
                               Codec: ${data.codec}<br>
                               Try downloading the video directly: <a href="${videoUrl}" download>Download Video</a>`;
          panel.appendChild(errorDiv);
        };
        
        video.onloadstart = function() {
          console.log('Video load started');
        };
        
        video.onloadedmetadata = function() {
          console.log('Video metadata loaded, duration:', video.duration);
        };
        
        video.oncanplay = function() {
          console.log('Video can play');
        };
        
        video.oncanplaythrough = function() {
          console.log('Video can play through');
        };
        
        panel.appendChild(video);
        panel.innerHTML += `<div class='notice'>Saved server-side. Codec: ${data.codec}. Frames: ${data.frames}</div>`;
        panel.innerHTML += `<div class='notice'><a href="${videoUrl}" download>Download Video</a></div>`;
      } else if (data.annotated) { // fallback legacy base64
        console.log('Using base64 fallback');
        const binary = atob(data.annotated);
        const len = binary.length;
        const bytes = new Uint8Array(len);
        for (let i=0;i<len;i++) bytes[i] = binary.charCodeAt(i);
        const blob = new Blob([bytes], {type: 'video/mp4'});
        const url = URL.createObjectURL(blob);
        panel.innerHTML += `<video controls autoplay muted playsinline src="${url}" style='width:100%;border-radius:6px;'></video>`;
        panel.innerHTML += `<div class='notice'>Base64 inline fallback.</div>`;
      } else {
        console.error('No video data in response');
        panel.innerHTML += `<div class='bad'>No video data returned. Error: ${data.error || 'Unknown'}</div>`;
      }
    }
    panel.innerHTML += `<div class='badge'>Frames: ${data.frames || data.detections}</div>`;
    resultGrid.appendChild(panel);
    const dt = (performance.now()-t0).toFixed(1);
    statsEl.textContent = `Completed in ${dt} ms. Total detections: ${data.total_detections ?? data.detections}`;
  } catch (err) {
    console.error('Fetch error:', err);
    alert('Error: ' + err.message);
  } finally {
    progress.style.display = 'none';
  }
});
</script>
</body>
</html>
"""

@app.get('/', response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE.replace('{model_path}', str(MODEL_WEIGHTS_PATH))

def draw_boxes_pillow(image: Image.Image, results, conf: float):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    det_count = 0
    if hasattr(results, 'boxes'):
        for b in results.boxes:
            c = float(b.conf)
            if c < conf:
                continue
            det_count += 1
            x1,y1,x2,y2 = map(float, b.xyxy[0])
            draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=3)
            label = f"licence {c:.2f}"
            try:
                tw, th = draw.textbbox((0,0), label, font=font)[2:]
            except Exception:
                tw, th = font.getsize(label)
            draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=(0,255,0))
            draw.text((x1+3, max(0, y1 - th - 2)), label, fill=(0,0,0), font=font)
    return image, det_count

@app.post('/infer')
async def infer(request: Request, file: UploadFile = File(...), conf: float = Form(0.25)):
    t_start = time.time()
    contents = await file.read()
    size_kb = len(contents)/1024
    suffix = Path(file.filename).suffix.lower()
    model = get_model()
    logger.info(f"/infer received file={file.filename} type={suffix} size_kb={size_kb:.1f} conf={conf}")

    if suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        logger.debug(f"Image shape={image.size}")
        t0 = time.time()
        results = model.predict(image, conf=conf, verbose=False)[0]
        logger.debug(f"Raw detections={len(results.boxes) if hasattr(results, 'boxes') and results.boxes is not None else 0}")
        t_infer = time.time()-t0
        annotated, det_count = draw_boxes_pillow(image.copy(), results, conf)
        buf = io.BytesIO()
        annotated.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        total_ms = (time.time()-t_start)*1000
        logger.info(f"Image inference done detections={det_count} time_ms={total_ms:.1f}")
        return JSONResponse({ 'type': 'image', 'detections': det_count, 'annotated': b64, 'infer_ms': t_infer*1000, 'total_ms': total_ms })

    if suffix in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
        tmp_dir = Path(tempfile.gettempdir())
        in_path = tmp_dir / f"upload_{uuid.uuid4().hex}{suffix}"
        with open(in_path, 'wb') as f:
            f.write(contents)
        logger.debug(f"Saved upload to {in_path}")
        
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            logger.error(f"Could not open video file: {in_path}")
            return JSONResponse({'type': 'video', 'frames': 0, 'total_detections': 0, 'video_url': None, 'error': 'video_open_failed'})
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug(f"Video properties: fps={fps}, total_frames={total_frames}")
        
        if not fps or fps != fps or fps <= 1:
            fps = 24
            logger.warning(f"Invalid FPS detected, using default: {fps}")
        
        total_detections = 0
        frame_count = 0
        t_infer = 0.0
        stem = f"annotated_{uuid.uuid4().hex}"

        # Decide whether we can stream encode (memory friendly)
        can_stream = FFMPEG_BIN is not None
        logger.debug(f"Video encoding strategy: {'ffmpeg streaming' if can_stream else 'opencv buffered'}")

        frames_buffer = []  # only used if not streaming
        first_shape = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame = _ensure_even(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            t0 = time.time()
            results = model.predict(pil_img, conf=conf, verbose=False)[0]
            t_infer += time.time()-t0
            annotated, det_count = draw_boxes_pillow(pil_img, results, conf)
            total_detections += det_count
            out_frame = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
            if first_shape is None:
                first_shape = out_frame.shape[:2]
                logger.debug(f"Frame shape: {first_shape}, total shape: {out_frame.shape}")
            if can_stream:
                # Lazy start ffmpeg when calling write_video at end with iterator not possible; collect frames in generator pattern
                frames_buffer.append(out_frame)  # temporarily collect until threshold then flush streaming (simpler)
            else:
                frames_buffer.append(out_frame)
            
            # Log progress periodically
            if frame_count % 30 == 0:
                logger.debug(f"Processed frame={frame_count} cumulative_dets={total_detections}")
            
            if frame_count >= 600:
                logger.warning("Frame cap reached (600). Early stop.")
                break
        cap.release()
        
        # Clean up input file
        try:
            in_path.unlink()
        except Exception:
            pass

        if not frames_buffer:
            logger.warning("No frames decoded from video")
            return JSONResponse({'type': 'video', 'frames': 0, 'total_detections': 0, 'video_url': None, 'error': 'no_frames'})

        # Try video encoding with priority to web-compatible formats
        logger.debug(f"Starting video encoding with {len(frames_buffer)} frames")

        # Priority: ffmpeg H.264 > OpenCV MP4 > fallback
        try:
            if FFMPEG_BIN:
                logger.debug("Attempting ffmpeg H.264 encoding")
                out_path, codec_used = write_video_stream_ffmpeg(iter(frames_buffer), fps, stem, first_shape)
            else:
                logger.debug("ffmpeg not available, using OpenCV")
                out_path, codec_used = write_video_opencv(frames_buffer, fps, stem)
        except Exception as e:
            logger.warning(f"Primary video encoding failed: {e}")
            try:
                logger.debug("Attempting OpenCV fallback encoding")
                out_path, codec_used = write_video_opencv(frames_buffer, fps, stem)
            except Exception as e2:
                logger.exception(f"All video encoding methods failed: {e2}")
                return JSONResponse({'type': 'video', 'frames': frame_count, 'total_detections': total_detections, 'video_url': None, 'error': 'encode_failed', 'details': str(e2)})

        total_ms = (time.time()-t_start)*1000
        fps_est = frame_count / (t_infer if t_infer>0 else 1)
        rel_url = f"/static/videos/{out_path.name}"
        logger.info(f"Video inference frames={frame_count} total_detections={total_detections} time_ms={total_ms:.1f} fps_est={fps_est:.2f}")
        return JSONResponse({'type': 'video', 'frames': frame_count, 'total_detections': total_detections, 'video_url': rel_url, 'codec': codec_used, 'fps_est': fps_est, 'total_ms': total_ms})
    return JSONResponse({'error': 'Unsupported file type'}, status_code=400)

# Simple MJPEG stream (best effort, CPU heavy) for webcam demo
@app.get('/stream')
async def stream(conf: float = 0.25):
    model = get_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return JSONResponse({'error': 'Cannot open webcam'}, status_code=500)

    def gen():
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                results = model.predict(pil_img, conf=conf, verbose=False)[0]
                annotated, _ = draw_boxes_pillow(pil_img, results, conf)
                jpg_bytes = cv2.imencode('.jpg', cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR))[1].tobytes();
                yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg_bytes)).encode() + b"\r\n\r\n" + jpg_bytes + b"\r\n");
        finally:
            cap.release()
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        import uvicorn
        uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)
    except ImportError:
        print("uvicorn not available for direct run, use: uvicorn app.main:app --reload")
