# -*- coding: utf-8 -*-
"""
================================================================================
PHISHING DETECTION - INTERACTIVE URL TESTER
================================================================================
Simple script to test URLs against the trained model.

Modes:
  - Fast Mode (default): URL-only features, instant results
  - Deep Scan Mode: adds WHOIS, SSL, IP reputation, content analysis

Usage: python src/predict_url.py
================================================================================
"""

import sys
import os
import warnings
import logging

# Suppress noise
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)
logging.getLogger('whois').setLevel(logging.CRITICAL)
logging.getLogger('whois.whois').setLevel(logging.CRITICAL)

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import urllib3
urllib3.disable_warnings()

from phishing_detection_clean import extract_engineered_features, SoftVotingEnsemble
from external_features import extract_external_features

# Paths
MODEL_DIR = r'D:\FakingDetection\Models'

# Load model
print("=" * 60)
print("  PHISHING URL DETECTOR")
print("=" * 60)
print("\nLoading model...", end=" ")
model = joblib.load(os.path.join(MODEL_DIR, 'final_ensemble.joblib'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
print(f"OK ({len(feature_names)} features)\n")


# =============================================================================
# POST-PROCESSING: Content Analysis
# =============================================================================

def count_external_links(soup, domain):
    """Count links pointing to external domains."""
    external = 0
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http') and domain not in href:
            external += 1
    return external


def analyze_webpage_content(url, timeout=5):
    """Extract content-based features from actual webpage."""
    try:
        response = requests.get(url, timeout=timeout, verify=False,
                               headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        domain = urlparse(url).netloc

        content = {
            'line_count': len(response.text.split('\n')),
            'html_length': len(response.text),
            'has_forms': len(soup.find_all('form')) > 0,
            'form_count': len(soup.find_all('form')),
            'has_password_field': bool(soup.find('input', {'type': 'password'})),
            'has_hidden_fields': len(soup.find_all('input', {'type': 'hidden'})) > 0,
            'external_links': count_external_links(soup, domain),
            'image_count': len(soup.find_all('img')),
            'script_count': len(soup.find_all('script')),
            'has_title': bool(soup.find('title')),
            'has_favicon': bool(soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon')),
        }
        return content
    except Exception:
        return None


# =============================================================================
# POST-PROCESSING: Heuristic Boosts
# =============================================================================

def apply_heuristic_boost(base_prob, external_features, url_features):
    """Adjust prediction using WHOIS/Age, SSL, IP, and URL heuristics."""
    boost = 0.0
    reasons = []

    # 1. External Features
    if external_features:
        # A. Domain Age (WHOIS)
        age = external_features.get('DomainAge', 3650)
        if age != -1:
            if age < 30:
                boost += 0.30
                reasons.append(f"🕐 New Domain ({age} days)")
            elif age < 180:
                boost += 0.15
                reasons.append(f"🕐 Young Domain ({age} days)")

        # B. SSL Analysis
        ssl_age = external_features.get('SSLAgeDays', -1)
        ssl_issuer = external_features.get('SSLIssuer', 'Unknown')

        if ssl_age > -1 and ssl_age < 2:
            boost += 0.40
            reasons.append("🔒 Brand New SSL (< 48h)")

        if "Let's Encrypt" in ssl_issuer and age < 30:
            boost += 0.20
            reasons.append("🔒 Free SSL + New Domain")

        # C. IP/ASN Analysis
        isp = external_features.get('HostingISP', 'Unknown')
        if any(cloud in isp for cloud in ['DigitalOcean', 'Choopa', 'Vultr', 'Namecheap']):
            if age < 90:
                boost += 0.15
                reasons.append(f"🌐 Suspicious Hosting ({isp})")

    # 2. URL Heuristics (Brand in Subdomain/Path but NOT Tranco)
    tranco_score = url_features.get('DomainTrancoRank', 0)

    if tranco_score == 0:
        if url_features.get('HasBrandInSubdomain', 0) > 0:
            boost += 0.35
            reasons.append("🏷️ Brand in Subdomain (Unverified)")
        if url_features.get('HasBrandInPath', 0) > 0:
            boost += 0.20
            reasons.append("🏷️ Brand in Path (Unverified)")
        if url_features.get('IsTyposquatting', 0) > 0:
            boost += 1.0
            reasons.append("⚠️ Typosquatting Detected (Critical)")

    adjusted_prob = min(base_prob + boost, 1.0)
    return adjusted_prob, boost, reasons


def apply_content_boost(base_prob, content_features, url_features):
    """Adjust prediction using webpage content analysis."""
    if content_features is None:
        return base_prob, 0.0, []

    boost = 0.0
    reasons = []

    if content_features['line_count'] < 150:
        boost += 0.15
        reasons.append(f"📄 Very few lines ({content_features['line_count']})")

    if not content_features['has_title'] or not content_features['has_favicon']:
        boost += 0.10
        reasons.append("📄 Missing title/favicon")

    if (content_features['has_password_field'] and
        url_features.get('HasSuspiciousTLD', 0) == 1):
        boost += 0.25
        reasons.append("🔑 Password field + suspicious TLD")

    if content_features['has_forms'] and url_features.get('IsHTTPS', 0) == 0:
        boost += 0.20
        reasons.append("📝 Form without HTTPS")

    if (content_features['has_hidden_fields'] and
        url_features.get('HasPhishingKeywords', 0) > 0):
        boost += 0.15
        reasons.append("👁️ Hidden fields + phishing keywords")

    if content_features['external_links'] > 20:
        boost += 0.10
        reasons.append(f"🔗 Many external links ({content_features['external_links']})")

    adjusted_prob = min(base_prob + boost, 1.0)
    return adjusted_prob, boost, reasons


# =============================================================================
# PREDICTION
# =============================================================================

def predict(url: str, deep_scan: bool = False):
    """Predict if a URL is phishing or safe."""

    # Step 1: Extract URL-based features
    features = extract_engineered_features(url)
    if features is None:
        print("  [ERROR] Could not extract features from this URL.\n")
        return

    # Step 2: Align with model features
    feature_vector = pd.DataFrame([features])
    for feat in feature_names:
        if feat not in feature_vector.columns:
            feature_vector[feat] = 0
    feature_vector = feature_vector[feature_names]

    # Step 3: Get base model prediction
    proba = model.predict_proba(feature_vector)[0]
    # Dataset labels: 1=Legitimate, 0=Phishing
    prob_safe = proba[1] if len(proba) > 1 else proba[0]
    base_phishing_prob = 1.0 - prob_safe
    final_phishing_prob = base_phishing_prob

    # Step 4: Deep Scan boosts
    ext_data = None
    heur_reasons = []
    content_reasons = []

    if deep_scan:
        # A. External features (WHOIS, SSL, IP)
        print("  ⏳ Running deep scan...")
        print("     → WHOIS lookup...", end=" ", flush=True)
        ext_data = extract_external_features(url, fetch_content=False)
        print("done")

        # B. Heuristic boost
        print("     → Heuristic analysis...", end=" ", flush=True)
        final_phishing_prob, heur_boost, heur_reasons = apply_heuristic_boost(
            final_phishing_prob, ext_data, features
        )
        print("done")

        # C. Content analysis
        print("     → Content analysis...", end=" ", flush=True)
        content_features = analyze_webpage_content(url)
        final_phishing_prob, cont_boost, content_reasons = apply_content_boost(
            final_phishing_prob, content_features, features
        )
        print("done")

    is_phishing = final_phishing_prob >= 0.5

    # Display result
    print()
    if is_phishing:
        print("  ╔══════════════════════════════════════╗")
        print("  ║  ⚠️  PHISHING DETECTED               ║")
        print("  ╚══════════════════════════════════════╝")
    else:
        print("  ╔══════════════════════════════════════╗")
        print("  ║  ✅  SAFE                             ║")
        print("  ╚══════════════════════════════════════╝")

    print(f"  Phishing Probability: {final_phishing_prob:.1%}")
    if deep_scan and final_phishing_prob != base_phishing_prob:
        print(f"  Base Model Probability: {base_phishing_prob:.1%}")
    print(f"  Confidence: {max(1-final_phishing_prob, final_phishing_prob):.1%}")

    # Show key signals from URL features
    signals = []
    if features.get('IsTyposquatting', 0):
        signals.append(f"🔴 Typosquatting (similarity: {features.get('MaxBrandSimilarity', 0):.0%})")
    if features.get('NormalizedMatchesBrand', 0):
        signals.append("🔴 Leetspeak brand impersonation")
    if features.get('HasBrandInSubdomain', 0):
        signals.append("🔴 Brand name in subdomain")
    if features.get('HasBrandInPath', 0):
        signals.append("🔴 Brand name in path")
    if features.get('HasIPInURL', 0):
        signals.append("🔴 IP address in URL")
    if features.get('HasSuspiciousTLD', 0):
        signals.append("🔴 Suspicious TLD")
    if features.get('HasPhishingKeywords', 0):
        signals.append(f"🔴 Phishing keywords ({features['HasPhishingKeywords']})")
    if features.get('DomainTrancoRank', 0):
        signals.append("🟢 Trusted domain (Tranco Top 10K)")

    if signals:
        print(f"\n  URL Signals:")
        for s in signals:
            print(f"    {s}")

    # Show deep scan results
    if deep_scan:
        if ext_data:
            print(f"\n  Deep Scan Results:")
            age = ext_data.get('DomainAge', -1)
            age_str = f"{age} days" if age != -1 else "Unknown"
            print(f"    🌐 Domain Age: {age_str}")
            print(f"    🔒 SSL Issuer: {ext_data.get('SSLIssuer', 'Unknown')}")
            ssl_age = ext_data.get('SSLAgeDays', -1)
            print(f"    🔒 SSL Age: {ssl_age} days" if ssl_age != -1 else "    � SSL Age: Unknown")
            print(f"    🖥️ Hosting IP: {ext_data.get('HostingIP', 'Unknown')}")
            print(f"    �🟢 ISP: {ext_data.get('HostingISP', 'Unknown')}")
            print(f"    🌍 Country: {ext_data.get('HostingCountry', 'Unknown')}")
            print(f"    📡 URL Live: {'Yes' if ext_data.get('UrlIsLive', 0) == 1 else 'No'}")

        if heur_reasons or content_reasons:
            print(f"\n  Boost Signals:")
            for r in heur_reasons:
                print(f"    {r}")
            for r in content_reasons:
                print(f"    {r}")

    print()


# =============================================================================
# MAIN LOOP
# =============================================================================

# Mode selection
print("Select mode:")
print("  [1] Fast Mode   — URL analysis only (instant)")
print("  [2] Deep Scan   — URL + WHOIS + SSL + IP + Content (slower)")
print()

while True:
    mode_input = input("  Mode [1/2] > ").strip()
    if mode_input in ('1', '2', ''):
        break
    print("  Please enter 1 or 2.")

deep_scan = (mode_input == '2')
mode_name = "Deep Scan 🔍" if deep_scan else "Fast Mode ⚡"
print(f"\n  Using: {mode_name}")
print(f"\nEnter a URL to check (or 'quit' to exit):\n")

while True:
    try:
        url = input("  URL > ").strip()

        if not url:
            continue
        if url.lower() in ('quit', 'exit', 'q'):
            print("\nGoodbye! Stay safe online. 🛡️")
            break

        # Add scheme if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        predict(url, deep_scan=deep_scan)

    except KeyboardInterrupt:
        print("\n\nGoodbye! Stay safe online. 🛡️")
        break
    except Exception as e:
        print(f"  [ERROR] {e}\n")
