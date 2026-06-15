# 🛡️ TruthLens — Fake News & Scam Website Detection Platform

TruthLens is a full-stack AI-powered platform that detects fake news and identifies phishing/scam websites. It combines a **DeBERTa NLI deep learning model** for news verification with an **XGBoost ensemble** for website phishing detection, served through a modern web dashboard.

---

## 📁 Project Structure

```
FakingDetection/
├── frontend/          → HTML/CSS/JS web interface
├── backend/           → ASP.NET Core 9 API (C#)
├── ml-service/        → Python FastAPI ML prediction service
├── ml-models/         → Trained model files (required)
├── ml-source/         → Python modules for feature extraction
├── Notebooks/         → Training notebooks (reference only)
├── Data/              → Training datasets (not needed to run)
├── .env               → API keys (Serper)
├── requirements.txt   → Python dependencies
└── README.md          → This file
```

---

## ⚙️ Prerequisites

Make sure you have the following installed:

| Tool              | Version  | Download Link                                      |
|-------------------|----------|----------------------------------------------------|
| **Python**        | 3.10+    | https://www.python.org/downloads/                  |
| **.NET SDK**      | 9.0      | https://dotnet.microsoft.com/download/dotnet/9.0   |
| **Node.js** (optional) | 18+  | https://nodejs.org/ (only if you want a dev server)|

---

## 🚀 How to Run (Step by Step)

### Step 1: Set Up the Python Environment

Open a terminal in the project root folder (`FakingDetection/`):

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

> **Note:** Installing PyTorch may take a while (~2GB). If you have an NVIDIA GPU, install the CUDA version for faster inference:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 2: Configure the API Key

The news model requires a **Serper.dev API key** for Google Search evidence.

1. Get a free API key at: https://serper.dev/
2. Open the `.env` file in the project root
3. Set your key:
   ```
   SERPER_API_KEY=your-api-key-here
   ```

### Step 3: Start the ML Service (Python)

```bash
cd ml-service
python app.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
[INFO] ✅ Website model loaded (42 features)
[INFO] ✅ News model loaded and ready on cpu
[INFO]   TRUTHLENS ML SERVICE — All Models Loaded!
```

> **⏳ First startup takes ~1-2 minutes** as it loads the 870MB DeBERTa model into memory.
> The server will accept requests while models are loading (website model loads in ~10s, news model in ~60s).

**Keep this terminal open.**

### Step 4: Start the Backend (ASP.NET Core)

Open a **new terminal** in the project root:

```bash
cd backend
dotnet build
dotnet run
```

You should see:
```
[TruthLens] Starting server...
```

The backend runs on `http://localhost:5159`.

**Keep this terminal open.**

### Step 5: Open the Frontend

Open a **new terminal** and start a local web server for the frontend:

```bash
cd frontend
python -m http.server 5500
```

Then open your browser and go to:

```
http://localhost:5500/Truthlens.html
```

> **Alternative:** You can simply double-click `frontend/Truthlens.html` to open it directly in your browser. However, some features may not work without a proper HTTP server due to CORS restrictions.

---

## 🔑 Default Test Flow

1. Open `http://localhost:5500/Truthlens.html`
2. Click **Sign Up** to create an account
3. After signing in, use:
   - **Analyze News** → Enter a claim (e.g., "The Eiffel Tower is in Paris") to fact-check it
   - **Check Website** → Enter a URL (e.g., "https://google.com") to scan for phishing
4. View your results in the **Dashboard**

---

## 🧠 Models Overview

### Fake News Detection
- **Base Model:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- **Fine-tuned** with LoRA on the LIAR-Plus dataset (3-class NLI)
- **Pipeline:** User claim → Google Search (Serper API) → NLI comparison → Verdict (Verified / Inconclusive / Fake)

### Website Phishing Detection
- **Model:** XGBoost Soft-Voting Ensemble
- **42 engineered features** extracted from URL structure, domain info, and content analysis
- **Pipeline:** URL → Feature extraction → Model prediction → Risk score (Safe / Suspicious / Phishing)

---

## 📋 Ports Summary

| Service      | Port  | URL                          |
|-------------|-------|------------------------------|
| Frontend    | 5500  | http://localhost:5500        |
| Backend API | 5159  | http://localhost:5159/api    |
| ML Service  | 8000  | http://localhost:8000        |

---

## 🛑 Troubleshooting

| Issue | Solution |
|-------|----------|
| `News model takes long` | Normal on CPU (~15-30s per prediction). Use a GPU machine for faster inference. |
| `503 Service Unavailable` | Models are still loading. Wait for "All Models Loaded!" in the ML service terminal. |
| `Serper API error` | Check your API key in `.env`. Get a free key at https://serper.dev/ |
| `dotnet build` fails | Make sure .NET 9 SDK is installed: `dotnet --version` |
| `CORS errors in browser` | Use `python -m http.server 5500` instead of opening HTML files directly. |

---

## 📄 License

This project was built for academic purposes as a graduation project.
