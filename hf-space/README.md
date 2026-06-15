---
title: TruthLens ML Service
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# TruthLens ML Service

Fake News Detection (DeBERTa NLI) + Phishing Website Detection (XGBoost Ensemble).

## Endpoints

- `GET /health` — Health check
- `POST /predict/news` — Fake news detection
- `POST /predict/website` — Phishing website detection
