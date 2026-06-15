# -*- coding: utf-8 -*-
"""
================================================================================
TRUTHLENS — ML PREDICTION SERVICE (FastAPI)
================================================================================
Serves both Fake News Detection (DeBERTa + Serper) and Phishing/Website
Detection (XGBoost Ensemble + optional deep scan) over HTTP.

Endpoints:
  POST /predict/news      — Fake News detection
  POST /predict/website   — Phishing / Fake Website detection
  GET  /health            — Health check

Usage:
  uvicorn app:app --host 0.0.0.0 --port 8000
================================================================================
"""

import os
import sys
import json
import logging
import warnings
import threading
from typing import Optional, List

import torch
import joblib
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Suppress noisy warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Custom Ensemble Class (required for joblib deserialization) ──────────────
import numpy as np

class SoftVotingEnsemble:
    """Custom soft voting ensemble — must match the class used during training."""

    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1] * len(models)

    def predict_proba(self, X):
        probas = [m.predict_proba(X) for m in self.models]
        total = sum(self.weights)
        return sum(w * p for w, p in zip(self.weights, probas)) / total

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

# ─── Resolve Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML_SOURCE_DIR = os.path.join(BASE_DIR, "ml-source")
MODELS_DIR = os.path.join(BASE_DIR, "ml-models")

# Add ml-source to Python path so we can import phishing modules
sys.path.insert(0, ML_SOURCE_DIR)

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="TruthLens ML Service",
    description="Fake News & Phishing Detection API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response Models ────────────────────────────────────────────────

class NewsRequest(BaseModel):
    content: str
    serper_api_key: Optional[str] = None

class NewsResponse(BaseModel):
    verdict: str                 # "Verified", "Inconclusive", "Fake"
    confidence: float            # 0-1, confidence of the verdict
    credibility_score: float     # 0-100, overall credibility
    probabilities: dict          # { verified, neutral, fake }
    evidence: str                # Serper evidence text
    reasons: List[str]           # Human-readable reasons

class WebsiteRequest(BaseModel):
    url: str
    deep_scan: bool = False

class WebsiteResponse(BaseModel):
    verdict: str                 # "Safe", "Suspicious", "Phishing"
    phishing_probability: float  # 0-1
    threat_score: int            # 0-100
    signals: List[str]           # Detection signals
    deep_scan_results: Optional[dict] = None
    reasons: List[str]


# =============================================================================
# NEWS MODEL — DeBERTa + Serper
# =============================================================================

news_model = None
news_tokenizer = None
news_device = None


def load_news_model():
    """Load DeBERTa base model + LoRA adapter for NLI-based fake news detection."""
    global news_model, news_tokenizer, news_device

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel

    base_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    adapter_path = os.path.join(MODELS_DIR, "finetuned_model")
    merged_path = os.path.join(MODELS_DIR, "merged_deberta_model.pt")

    news_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer — prefer local cache in finetuned_model dir for speed
    local_tokenizer_path = adapter_path
    if os.path.exists(os.path.join(local_tokenizer_path, "tokenizer.json")):
        logger.info(f"Loading news tokenizer from local adapter dir (fast)")
        news_tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
    else:
        logger.info(f"Loading news tokenizer from: {base_model_name}")
        news_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if os.path.exists(merged_path):
        logger.info(f"⚡ Found pre-merged model at {merged_path}! Loading instantly...")
        news_model = torch.load(merged_path, map_location=news_device, weights_only=False)
    else:
        logger.info(f"Loading base model into memory (this happens once)...")
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=3
        )
        logger.info(f"Applying LoRA adapter from: {adapter_path}")
        news_model = PeftModel.from_pretrained(base, adapter_path)
        
        # Merge LoRA weights directly into the base weights
        logger.info("Merging LoRA weights into base model...")
        news_model = news_model.merge_and_unload()
        news_model.to(news_device)

        # Save merged model for instant future loading
        logger.info("💾 Saving merged model to disk for instant future loading...")
        torch.save(news_model, merged_path)
        
        import gc
        gc.collect()

    news_model.eval()
    logger.info(f"✅ News model loaded and ready on {news_device}")


# ─── Evidence Quality Scoring ─────────────────────────────────────────────────
# Credibility tiers for known domains.  Unlisted domains default to 0.5.
DOMAIN_CREDIBILITY = {
    # Tier 1 — Gold-standard reference sources
    "wikipedia.org": 1.0, "britannica.com": 1.0,
    # Tier 2 — Major wire services & newspapers of record
    "reuters.com": 0.95, "apnews.com": 0.95, "bbc.com": 0.95,
    "bbc.co.uk": 0.95, "nytimes.com": 0.90, "theguardian.com": 0.90,
    "washingtonpost.com": 0.90, "nature.com": 0.95, "science.org": 0.95,
    "who.int": 0.95, "cdc.gov": 0.95, "nih.gov": 0.95,
    # Tier 3 — Reputable news & fact-checkers
    "snopes.com": 0.90, "factcheck.org": 0.90, "politifact.com": 0.90,
    "cnn.com": 0.80, "nbcnews.com": 0.80, "abcnews.go.com": 0.80,
    "aljazeera.com": 0.80, "france24.com": 0.80, "dw.com": 0.80,
    # Tier 4 — Lower-quality / user-generated
    "medium.com": 0.45, "quora.com": 0.40, "reddit.com": 0.40,
    "blogspot.com": 0.30, "wordpress.com": 0.35, "tumblr.com": 0.30,
    "tiktok.com": 0.25, "facebook.com": 0.30, "twitter.com": 0.35,
    "x.com": 0.35,
}

MIN_EVIDENCE_QUALITY = 0.25          # Drop snippets below this threshold
MAX_QUALITY_SNIPPETS  = 5            # Keep only the best N after scoring


def _extract_domain(url_str: str) -> str:
    """Return the registrable domain from a URL (e.g. 'en.wikipedia.org' → 'wikipedia.org')."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url_str).netloc.lower()
        parts = host.split(".")
        # Handle two-part TLDs like .co.uk
        if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "gov", "ac", "net"):
            return ".".join(parts[-3:])
        return ".".join(parts[-2:]) if len(parts) >= 2 else host
    except Exception:
        return ""


def _score_domain(domain: str) -> float:
    """Return a credibility score for *domain* (0-1). Unknown domains get 0.5."""
    if not domain:
        return 0.5
    # Check exact match first, then try parent domain
    if domain in DOMAIN_CREDIBILITY:
        return DOMAIN_CREDIBILITY[domain]
    # e.g. 'en.wikipedia.org' → 'wikipedia.org'
    parts = domain.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[i:])
        if parent in DOMAIN_CREDIBILITY:
            return DOMAIN_CREDIBILITY[parent]
    # Government / educational TLDs get a small boost
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.80
    return 0.5


def get_serper_evidence(claim: str, api_key: str, max_results: int = 8) -> dict:
    """Search Google via Serper.dev and return quality-scored evidence.

    Returns
    -------
    dict  with keys:
        snippets : list[str]   – quality-filtered evidence texts (top N)
        domains  : list[str]   – corresponding source domains
        quality  : list[float] – credibility score per snippet
        avg_quality : float    – mean credibility of kept snippets
    """
    empty = {"snippets": [], "domains": [], "quality": [], "avg_quality": 0.0}
    if not api_key or not api_key.strip():
        return empty

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": claim, "num": max_results})
    headers = {
        "X-API-KEY": api_key.strip(),
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        if response.status_code == 403:
            logger.warning("Serper API: Invalid API key")
            return empty
        response.raise_for_status()
        results = response.json()

        raw: list[tuple[str, str, float]] = []   # (snippet, domain, score)
        seen = set()

        # High-quality source: Knowledge Graph description (e.g. Wikipedia)
        kg = results.get("knowledgeGraph", {})
        kg_desc = kg.get("description", "").strip()
        if kg_desc and len(kg_desc) > 30:
            seen.add(kg_desc)
            raw.append((kg_desc, "wikipedia.org", 1.0))  # KG is usually Wikipedia

        # Organic search snippets
        for res in results.get("organic", [])[:max_results]:
            snippet = res.get("snippet", "").strip()
            link    = res.get("link", "")
            if snippet and snippet not in seen and len(snippet) > 20:
                seen.add(snippet)
                domain = _extract_domain(link)
                score  = _score_domain(domain)
                raw.append((snippet, domain, score))

        # Filter out low-quality sources and keep top N by score
        raw = [(s, d, q) for s, d, q in raw if q >= MIN_EVIDENCE_QUALITY]
        raw.sort(key=lambda x: x[2], reverse=True)
        raw = raw[:MAX_QUALITY_SNIPPETS]

        if not raw:
            return empty

        snippets = [s for s, _, _ in raw]
        domains  = [d for _, d, _ in raw]
        quality  = [q for _, _, q in raw]
        avg_q    = sum(quality) / len(quality)

        logger.info(f"Evidence quality: {list(zip(domains, [f'{q:.2f}' for q in quality]))}")

        return {
            "snippets": snippets,
            "domains": domains,
            "quality": quality,
            "avg_quality": avg_q,
        }

    except Exception as e:
        logger.error(f"Serper search failed: {e}")
        return empty


def _calibrate_confidence(raw_confidence: float, evidence: dict) -> float:
    """Penalise overconfident verdicts when evidence is weak or thin.

    Calibration factors (multiplicative):
      • < 3 sources          → ×0.85
      • only 1 source        → ×0.70  (stacks with above)
      • < 2 unique domains   → ×0.80
      • avg quality < 0.5    → ×0.75
    Result is capped at 0.99.
    """
    cal = raw_confidence
    n_sources = len(evidence.get("snippets", []))
    n_domains = len(set(evidence.get("domains", [])))
    avg_q     = evidence.get("avg_quality", 0.5)

    if n_sources < 3:
        cal *= 0.85
    if n_sources <= 1:
        cal *= 0.70
    if n_domains < 2:
        cal *= 0.80
    if avg_q < 0.5:
        cal *= 0.75

    return min(cal, 0.99)


def predict_news(claim: str, api_key: str) -> dict:
    """Run the full news prediction pipeline: Serper → DeBERTa NLI.
    
    IMPORTANT: The model was fine-tuned with tokenizer(premise, hypothesis)
    where premise = FULL evidence text and hypothesis = claim.
    We must replicate this exact format for accurate results.

    Improvements over v1:
      • Evidence Quality Scoring — snippets are weighted by source credibility,
        low-quality blogs are filtered out, only the top 5 are kept.
      • Confidence Calibration — raw model confidence is penalised when
        evidence is thin, low-diversity, or low-quality.
    """
    import gc

    # Step 1: Get quality-scored evidence snippets from Google
    evidence = get_serper_evidence(claim, api_key)
    snippets = evidence["snippets"]

    if not snippets:
        return {
            "verdict": "Inconclusive",
            "confidence": 1.0,
            "credibility_score": 50.0,
            "probabilities": {
                "verified": 0.0,
                "neutral": 1.0,
                "fake": 0.0,
            },
            "evidence": "No internet evidence found on Google.",
            "reasons": ["Search Failed: Google search results do not definitively prove or disprove this."],
        }

    # Step 2: Build the full evidence string (matches training format)
    # The model was trained on LIAR-Plus where the premise is one complete
    # justification paragraph.  Concatenating all snippets into a single
    # premise replicates that format and gives the model maximum context.
    evidence_text = " ".join(snippets)

    # Step 3: Single-pass NLI — tokenizer(premise=evidence, hypothesis=claim)
    # This matches the Colab training: tokenizer(batch["premise"], batch["hypothesis"])
    with torch.inference_mode():
        inputs = news_tokenizer(
            evidence_text,     # sequence A = premise (evidence)
            claim,             # sequence B = hypothesis (claim)
            truncation=True,
            max_length=512,    # matches the Gradio app setting
            padding=True,
            return_tensors="pt"
        ).to(news_device)

        output = news_model(**inputs)
        probs = torch.softmax(output.logits.float(), dim=-1).tolist()[0]
        del inputs, output

    prob_verified = probs[0]  # entailment
    prob_neutral = probs[1]   # neutral
    prob_fake = probs[2]      # contradiction

    logger.info(f"DEBUG [News]: Probs: verified={prob_verified:.4f}, neutral={prob_neutral:.4f}, fake={prob_fake:.4f}")

    # Step 4: Verdict — simple argmax (model is well-calibrated from training)
    max_idx = probs.index(max(probs))
    if max_idx == 0:
        verdict = "Verified"
        credibility = prob_verified * 100
    elif max_idx == 2:
        verdict = "Fake"
        credibility = (1 - prob_fake) * 100
    else:
        verdict = "Inconclusive"
        credibility = 50 + (prob_verified - prob_fake) * 50

    # Step 5: Calibrate confidence based on evidence quality & diversity
    raw_conf = max(probs)
    calibrated_conf = _calibrate_confidence(raw_conf, evidence)

    logger.info(
        f"DEBUG [News]: Confidence calibration: raw={raw_conf:.4f} → calibrated={calibrated_conf:.4f} "
        f"(sources={len(snippets)}, domains={len(set(evidence['domains']))}, avg_q={evidence['avg_quality']:.2f})"
    )

    # Clean up GPU/CPU memory to prevent OOM
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build reasons
    reasons = []
    if verdict == "Verified":
        reasons.append("✅ VERIFIED: Google Search evidence supports this claim.")
        if calibrated_conf > 0.8:
            reasons.append("High confidence match with multiple sources")
    elif verdict == "Fake":
        reasons.append("❌ FAKE NEWS: Google Search evidence contradicts this claim.")
        if calibrated_conf > 0.8:
            reasons.append("Strong contradiction with verified sources")
    else:
        reasons.append("⚠️ INCONCLUSIVE: Google search results do not definitively prove or disprove this.")

    # Add evidence-quality insight for transparency
    n_src = len(snippets)
    avg_q = evidence["avg_quality"]
    if avg_q >= 0.8:
        reasons.append(f"Evidence sourced from {n_src} high-credibility source{'s' if n_src != 1 else ''} (avg quality: {avg_q:.0%})")
    elif avg_q >= 0.5:
        reasons.append(f"Evidence sourced from {n_src} moderate-credibility source{'s' if n_src != 1 else ''} (avg quality: {avg_q:.0%})")
    else:
        reasons.append(f"⚠️ Evidence quality is low ({avg_q:.0%}) — results may be less reliable")

    return {
        "verdict": verdict,
        "confidence": round(calibrated_conf, 4),
        "credibility_score": round(max(0, min(100, credibility)), 1),
        "probabilities": {
            "verified": round(prob_verified, 4),
            "neutral": round(prob_neutral, 4),
            "fake": round(prob_fake, 4),
        },
        "evidence": evidence_text,
        "reasons": reasons,
    }


# =============================================================================
# WEBSITE MODEL — XGBoost Ensemble + Deep Scan
# =============================================================================

website_model = None
website_feature_names = None


def load_website_model():
    """Load the trained phishing detection ensemble model."""
    global website_model, website_feature_names

    model_path = os.path.join(MODELS_DIR, "final_ensemble.joblib")
    features_path = os.path.join(MODELS_DIR, "feature_names.joblib")

    logger.info(f"Loading website model from: {model_path}")
    website_model = joblib.load(model_path)
    website_feature_names = joblib.load(features_path)
    logger.info(f"✅ Website model loaded ({len(website_feature_names)} features)")


def predict_website(url: str, deep_scan: bool = False) -> dict:
    """Run the full website prediction pipeline."""
    # Import from ml-source
    from phishing_detection_clean import extract_engineered_features

    # Step 1: Normalize & Unshorten URL
    url_clean = url.strip()
    if not url_clean.startswith(("http://", "https://")):
        url_clean = "https://" + url_clean
        
    from urllib.parse import urlparse
    parsed = urlparse(url_clean)
    hostname = parsed.netloc.split(':')[0].lower()
    
    # Fast-Path: Local development environments
    if hostname in ("localhost", "127.0.0.1") or hostname.startswith("192.168.") or hostname.startswith("10."):
        return {
            "verdict": "Safe",
            "phishing_probability": 0.0,
            "threat_score": 0,
            "signals": ["Local development environment detected"],
            "reasons": ["URL points to a local or private IP address."],
            "deep_scan_results": None
        }

    try:
        # Auto-unshorten only if it is a known shortener to avoid unintentional redirects adding slashes to legitimate roots
        from urllib.parse import urlparse
        domain = urlparse(url_clean).netloc.lower()
        shorteners = {'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'is.gd', 'ow.ly', 'buff.ly', 'adf.ly'}
        
        if domain.replace('www.', '') in shorteners:
            head_resp = requests.head(url_clean, allow_redirects=True, timeout=5)
            if head_resp.url and head_resp.url != url_clean:
                logger.info(f"Unshortened {url_clean} -> {head_resp.url}")
                url_clean = head_resp.url
    except Exception as e:
        logger.warning(f"URL Unshortening failed for {url_clean}: {e}")

    # Step 2: Extract URL-based features
    features = extract_engineered_features(url_clean)
    if features is None:
        raise ValueError(f"Could not extract features from URL: {url}")
        
    logger.info(f"DEBUG [{url_clean}]: Extracted Features: {features}")

    # --- Structural Bias Mitigation (Data Leakage Fix) ---
    # The original dataset was heavily biased regarding URL paths.
    # To prevent legitimate URLs with paths from instantly getting flagged, we neutralize path structural constraints here.
    features['PathLength'] = 0
    features['NoOfSlashInURL'] = 2
    # -----------------------------------------------------

    # Step 3: Align with model features
    feature_vector = pd.DataFrame([features])
    for feat in website_feature_names:
        if feat not in feature_vector.columns:
            feature_vector[feat] = 0
    feature_vector = feature_vector[website_feature_names]
    
    logger.info(f"DEBUG [{url_clean}]: Feature Vector: {feature_vector.to_dict(orient='records')[0]}")

    # Step 4: Predict
    proba = website_model.predict_proba(feature_vector)[0]

    # Model classes: 0=Legitimate, 1=Phishing
    base_phishing_prob = proba[1] if len(proba) > 1 else 1.0 - proba[0]
    
    # Tier 1 Improvement: Tranco Whitelist Fast-Path
    # If domain is highly trusted (Top 10k), not a shortener, and doesn't explicitly impersonate a brand in subdomains/paths
    if features.get("DomainTrancoRank", 0) == 1 and not features.get("IsShortenedURL", 0):
        if not features.get("HasBrandInSubdomain", 0) and not features.get("HasBrandInPath", 0):
            base_phishing_prob = min(base_phishing_prob, 0.05) # Cap at 5%
            
    # Tier 2 Improvement: Strong Phishing Heuristics Override
    # Always penalize shorteners significantly because they hide the true destination
    if features.get("IsShortenedURL", 0):
        base_phishing_prob = max(base_phishing_prob, 0.85)
        
    # For domains not in the top 10k (or overriding low base probabilities)
    if features.get("DomainTrancoRank", 0) == 0:
        if features.get("HasIPInURL", 0):
            base_phishing_prob = max(base_phishing_prob, 0.85)
        if features.get("HasBrandInSubdomain", 0) or features.get("HasBrandInPath", 0):
            base_phishing_prob = max(base_phishing_prob, 0.80)
        if features.get("IsTyposquatting", 0):
            base_phishing_prob = max(base_phishing_prob, 0.85)
        if features.get("HasPhishingKeywords", 0):
            base_phishing_prob = max(base_phishing_prob, 0.75)
            
    final_phishing_prob = base_phishing_prob

    # Step 5: Build URL-based signals
    signals = []
    if features.get("IsTyposquatting", 0):
        signals.append(f"Typosquatting detected (similarity: {features.get('MaxBrandSimilarity', 0):.0%})")
    if features.get("NormalizedMatchesBrand", 0):
        signals.append("Leetspeak brand impersonation detected")
    if features.get("HasBrandInSubdomain", 0):
        signals.append("Brand name found in subdomain")
    if features.get("HasBrandInPath", 0):
        signals.append("Brand name found in URL path")
    if features.get("HasIPInURL", 0):
        signals.append("IP address used instead of domain name")
    if features.get("HasSuspiciousTLD", 0):
        signals.append("Suspicious top-level domain detected")
    if features.get("HasPhishingKeywords", 0):
        signals.append(f"Phishing keywords found ({features['HasPhishingKeywords']})")
    if features.get("DomainTrancoRank", 0):
        signals.append("Domain is in Tranco Top 10K (trusted)")
    if features.get("IsShortenedURL", 0):
        signals.append("URL shortener detected")
    if not features.get("IsHTTPS", 0):
        signals.append("No HTTPS encryption")

    # Step 6: Deep scan (optional)
    deep_scan_results = None
    heur_reasons = []
    content_reasons = []

    if deep_scan:
        try:
            from external_features import extract_external_features
            from predict_url import (
                apply_heuristic_boost,
                apply_content_boost,
                analyze_webpage_content,
            )

            # External features (WHOIS, SSL, IP)
            ext_data = extract_external_features(url_clean, fetch_content=False)

            # Heuristic boost
            final_phishing_prob, heur_boost, heur_reasons = apply_heuristic_boost(
                final_phishing_prob, ext_data, features
            )

            # Content analysis
            content_features = analyze_webpage_content(url_clean)
            final_phishing_prob, cont_boost, content_reasons = apply_content_boost(
                final_phishing_prob, content_features, features
            )

            deep_scan_results = {
                "domain_age_days": ext_data.get("DomainAge", -1),
                "ssl_issuer": ext_data.get("SSLIssuer", "Unknown"),
                "ssl_age_days": ext_data.get("SSLAgeDays", -1),
                "hosting_ip": ext_data.get("HostingIP", "Unknown"),
                "hosting_isp": ext_data.get("HostingISP", "Unknown"),
                "hosting_country": ext_data.get("HostingCountry", "Unknown"),
                "url_is_live": ext_data.get("UrlIsLive", -1),
            }
        except Exception as e:
            logger.error(f"Deep scan failed: {e}")
            deep_scan_results = {"error": str(e)}

    # Step 7: Determine verdict
    logger.info(f"DEBUG [{url_clean}]: Raw Ensemble Proba: {proba}")
    logger.info(f"DEBUG [{url_clean}]: Base Phishing Prob: {base_phishing_prob}")
    logger.info(f"DEBUG [{url_clean}]: Final Phishing Prob (after boosts): {final_phishing_prob}")

    # Combine signals from deep scan
    all_details = []
    
    if deep_scan:
        logger.info(f"DEBUG [{url_clean}]: Heur Boost applied: {heur_boost}")
        logger.info(f"DEBUG [{url_clean}]: Cont Boost applied: {cont_boost}")

    final_phishing_prob = min(max(final_phishing_prob, 0.0), 0.99)
    threat_score = int(final_phishing_prob * 100)
    logger.info(f"DEBUG [{url_clean}]: Final Threat Score: {threat_score}%")

    if final_phishing_prob >= 0.7:
        verdict = "Phishing"
    elif final_phishing_prob >= 0.4:
        verdict = "Suspicious"
    else:
        verdict = "Safe"

    for s in signals:
        all_details.append(s)
    for r in heur_reasons:
        all_details.append(r)
    for r in content_reasons:
        all_details.append(r)

    # Build comprehensive reasons for the frontend
    reasons = []
    
    if verdict == "Safe":
        reasons.append(f"Threat probability is exclusively low ({final_phishing_prob:.1%}).")
        reasons.append("URL structure and domain patterns appear completely legitimate.")
        if features.get("DomainTrancoRank", 0):
            reasons.append("Domain is well-known and verified in the Tranco Top 10K trusted list.")
    elif verdict == "Phishing":
        reasons.append(f"Critically high threat probability detected ({final_phishing_prob:.1%}).")
        reasons.append("Multiple aggressive phishing indicators or deceptive patterns found.")
    else:
        reasons.append(f"Moderate threat probability ({final_phishing_prob:.1%}).")
        reasons.append("Some suspicious indicators found that warrant caution.")

    # Append all specific heuristic, content, and feature signals as detailed reasons
    for detail in all_details:
        reasons.append(detail)

    # Cleanup
    del features
    del feature_vector
    if deep_scan:
        pass # Optional deeper cleanup
    import gc
    gc.collect()

    return {
        "verdict": verdict,
        "phishing_probability": round(final_phishing_prob, 4),
        "threat_score": threat_score,
        "signals": signals,
        "deep_scan_results": deep_scan_results,
        "reasons": reasons,
    }


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Eagerly load all models in a background thread right after the server is up."""
    logger.info("=" * 60)
    logger.info("  TRUTHLENS ML SERVICE — Starting Up")
    logger.info("=" * 60)

    def _load_all_models():
        """Background thread: loads both models so the server can accept /health immediately."""
        try:
            logger.info("[BG] Loading website model...")
            load_website_model()
        except Exception as e:
            logger.error(f"[BG] Website model failed to load: {e}")

        try:
            logger.info("[BG] Loading news model (this may take a minute)...")
            load_news_model()
        except Exception as e:
            logger.error(f"[BG] News model failed to load: {e}")

        logger.info("=" * 60)
        logger.info("  TRUTHLENS ML SERVICE — All Models Loaded!")
        logger.info("=" * 60)

    # Launch in a daemon thread so the HTTP server starts immediately
    threading.Thread(target=_load_all_models, daemon=True).start()

    logger.info("Server is UP — models are loading in the background...")
    logger.info("=" * 60)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "news_model_loaded": news_model is not None,
        "website_model_loaded": website_model is not None,
    }


@app.post("/predict/news", response_model=NewsResponse)
async def predict_news_endpoint(req: NewsRequest):
    if news_model is None:
        raise HTTPException(status_code=503, detail="News model is still loading, please wait...")
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    # Use env variable as fallback for API key
    api_key = req.serper_api_key or os.environ.get("SERPER_API_KEY", "")

    try:
        result = predict_news(req.content, api_key)
        return NewsResponse(**result)
    except Exception as e:
        logger.error(f"News prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/website", response_model=WebsiteResponse)
async def predict_website_endpoint(req: WebsiteRequest):
    if website_model is None:
        raise HTTPException(status_code=503, detail="Website model is still loading, please wait...")
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    try:
        result = predict_website(req.url, deep_scan=req.deep_scan)
        return WebsiteResponse(**result)
    except Exception as e:
        logger.error(f"Website prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
