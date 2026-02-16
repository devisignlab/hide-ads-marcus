# Shield AI - SaaS Frontend Redesign

## Overview
Dark tech dashboard (Vercel/Linear style) with auto-magic flow + advanced settings toggle.

## Flow
1. Upload → 2. Auto diagnosis (smart_analyze) → 3. One-click process → 4. Results with download

## Screens
- **Upload**: Drag & drop, clean dark UI
- **Diagnosis**: Risk cards, flags, recommended attacks, time estimate, "Protect Video" CTA
- **Processing**: Step-by-step progress with ETA per step
- **Results**: Side-by-side video players, before/after scores, download button

## Backend Changes
- New `POST /api/analyze` endpoint (runs smart_analyze)
- Update `POST /api/process` to accept `mode: "auto"` (uses UAP + PGD optimized pipeline)
- Pipeline: UAP CLIP (~3-4min fixed) + PGD anti-deepfake (5s/keyframe, interval=30) + remux

## Tech
- Vanilla HTML/CSS/JS, dark glassmorphism
- HTML5 video players for comparison
- WebSocket progress with step-level granularity
