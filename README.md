<p align="center">
  <h1 align="center">Shastra 2026 — Transaction Intelligence</h1>
</p>
<p align="center">
  <em><code>AI-powered EDA, anomaly detection, and insights for bank transactions.</code></em>
</p>
<p align="center">
  <img src="https://img.shields.io/github/license/6Humans/Shaastra2026?style=flat&logo=opensourceinitiative&logoColor=white&color=0ea5e9" alt="license">
  <img src="https://img.shields.io/github/last-commit/6Humans/Shaastra2026?style=flat&logo=git&logoColor=white&color=0ea5e9" alt="last-commit">
  <img src="https://img.shields.io/github/languages/top/6Humans/Shaastra2026?style=flat&color=0ea5e9" alt="top-language">
  <img src="https://img.shields.io/github/languages/count/6Humans/Shaastra2026?style=flat&color=0ea5e9" alt="language-count">
  <img src="https://img.shields.io/github/contributors/6Humans/Shaastra2026?style=flat&logo=github&logoColor=white&color=0ea5e9" alt="contributors">
  <img src="https://img.shields.io/github/stars/6Humans/Shaastra2026?style=flat&logo=github&logoColor=white&color=0ea5e9" alt="stars">
  <img src="https://img.shields.io/github/forks/6Humans/Shaastra2026?style=flat&logo=github&logoColor=white&color=0ea5e9" alt="forks">
  <img src="https://img.shields.io/github/issues/6Humans/Shaastra2026?style=flat&logo=github&logoColor=white&color=0ea5e9" alt="issues">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-0ea5e9?style=flat&logo=react&logoColor=white" alt="React">
  <img src="https://img.shields.io/badge/TypeScript-2d79c7?style=flat&logo=typescript&logoColor=white" alt="TypeScript">
</p>

<br>

## What it does
- Upload a CSV of bank transactions and get instant EDA, anomaly detection, and AI insights.
- Consolidated anomaly report with high-risk columns, type validation errors, and ML outliers.
- Frontend dashboard (Vite + React) consuming the FastAPI backend.

## Quick start
```bash
# Backend (FastAPI)
cd backend
pip install -e .
uvicorn api:app --reload --port 8000

# Frontend (Vite React)
cd ../frontend
npm install
npm run dev
```

### Try the API quickly
```bash
curl -X POST "http://localhost:8000/analyze-transactions?num_samples=5" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/transactions.csv"
```

## Key endpoints
- `GET /health` — service heartbeat
- `POST /analyze-transactions?num_samples=5` — upload CSV and receive full analysis

## Frontend entry points
- `frontend/src/App.tsx` — main dashboard shell
- `frontend/src/components/upload/AnalysisStepper.tsx` — upload + run analysis
- `frontend/src/components/anomaly/AnomalySummary.tsx` — anomaly highlights
- `frontend/src/lib/api.ts` — API client

## Data samples
- Sample CSVs: `data/*.csv` (multi-bank variants included)
