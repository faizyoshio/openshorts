# OpenShorts (Fork by faizyoshio)

OpenShorts is a self-hosted app that turns long videos (YouTube URL or local upload) into short clips for TikTok, Instagram Reels, and YouTube Shorts.

This repository is a fork of:
- Upstream: `https://github.com/mutonby/openshorts`
- This fork: `https://github.com/faizyoshio/openshorts`

## What This Project Does
- Download video from YouTube or accept local file upload
- Transcribe with Faster-Whisper
- Detect high-potential clip moments with Gemini
- Export clips with optional:
  - subtitles
  - hook overlays
  - translation/dubbing
  - direct social posting via Upload-Post

## Quick Start (Recommended)

### 1. Clone
```bash
git clone https://github.com/faizyoshio/openshorts.git
cd openshorts
```

### 2. (Optional) Prepare `.env`
Copy `.env.example` to `.env` and fill optional values:
- AWS S3 settings (if you want automatic clip backup)
- `YOUTUBE_COOKIES` (if YouTube blocks download requests)

### 3. Run with Docker
```bash
docker compose up --build
```

### 4. Open dashboard
- Frontend: `http://localhost:5175`
- Backend API: `http://localhost:8000`

### 5. First usage
1. Open `Settings`
2. Enter Gemini API key
3. (Optional) Enter Upload-Post API key
4. Go to Dashboard, paste URL/upload file, click `Generate Clips`

## Key Runtime Requirements
- Docker + Docker Compose
- Gemini API key (required for AI analysis)
- Upload-Post API key (optional, for one-click social posting)
- ElevenLabs API key (optional, for dubbing/translation)

## Main Processing Options
From dashboard input:
- Output orientation:
  - `Vertical` (9:16 reframing with tracking)
  - `Horizontal` (keep original framing)
- Clip duration: `Auto` or `Custom`
- Output count: `Auto` or `Custom`

## Output Format Defaults
This fork normalizes output for compatibility:
- Container: `MP4`
- Video codec: `H.264` (`libx264`, `yuv420p`)
- Audio codec: `AAC`
- `+faststart` enabled for web playback

## Social Posting Setup (Upload-Post)
1. Login/register: `https://app.upload-post.com/login`
2. Create profile: `https://app.upload-post.com/manage-users`
3. Connect social accounts to that profile
4. Generate API key: `https://app.upload-post.com/api-keys`
5. Paste key in app settings, then connect profile

## Troubleshooting

### YouTube download quality looks low
- This fork uses `yt-dlp` format selector `bv*+ba/b` and explicit format sorting.
- If source itself is low quality, output cannot exceed source quality.
- If blocked by YouTube anti-bot, set `YOUTUBE_COOKIES` in `.env`.

### Frontend not reachable
- Confirm `docker compose` is running
- Confirm port `5175` is free
- Frontend URL is `http://localhost:5175` (not 5173 in this fork)

### Auto post result quality changes
- Some platforms may transcode uploaded videos.
- This fork returns diagnostics (including `video_was_transcoded` signal from Upload-Post history/status when available).

### Local verification scripts fail
- Install Python dependencies first (at least `Pillow` for hook checks), or run inside Docker.

## Development (Local, Non-Docker)

### Backend
```bash
python -m pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd dashboard
npm install
npm run dev
```

Useful checks:
```bash
npm --prefix dashboard run build
npm --prefix dashboard run lint
python -m py_compile app.py main.py editor.py hooks.py subtitles.py
```

## What Changed in This Fork (vs upstream/mutonby)

The following are the major fork-specific changes added and maintained here:

1. Output orientation switch (Vertical or Horizontal)
- Added end-to-end option from UI -> API -> processing pipeline
- Horizontal mode keeps source framing; vertical mode keeps AI reframing flow

2. Output normalization to MP4/H.264/AAC
- Main clip pipeline standardized to MP4 + H.264 + AAC
- Added playback-friendly flags (`yuv420p`, `+faststart`)
- Applied to core pipeline and edit/subtitle/hook paths

3. Improved YouTube download quality selection
- Updated yt-dlp format selector and sort strategy for best available source quality
- Added logs for selected/downloaded formats

4. Auto-post diagnostics (Upload-Post)
- Backend now includes post diagnostics payload:
  - local video specs
  - request id
  - status/history fetch
  - `video_was_transcoded` indicator (when available)
- UI post feedback now surfaces this information

5. Frontend/dev config fixes
- Fixed frontend port consistency to `5175` across:
  - `dashboard/vite.config.js`
  - `dashboard/Dockerfile`
  - `docker-compose.yml`

6. Lint/tooling compatibility updates
- Updated dashboard ESLint setup to run cleanly with current flat config tooling
- Fixed lint script invocation for modern ESLint CLI

7. Verification scripts hardened for Windows terminals
- Reworked `verify_*.py` scripts to avoid encoding-related crashes and provide clearer dependency messages

## Security Notes
- API keys are stored client-side in browser local storage by dashboard logic.
- Keys are sent to backend only when needed to process requests.
- Do not expose your self-hosted instance publicly without authentication/proxy hardening.

## License
MIT (same as upstream unless noted otherwise).
