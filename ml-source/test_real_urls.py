# -*- coding: utf-8 -*-
"""
================================================================================
PHISHING DETECTION - REAL URL TESTING
================================================================================
Test the trained model with real URLs to verify performance.
================================================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import urllib3
from typing import Dict, Any, List, Optional

# Suppress warnings
warnings.filterwarnings('ignore')
urllib3.disable_warnings()
logging.basicConfig(level=logging.ERROR)

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, r'D:\FakingDetection\src')

import tldextract

# Import feature extraction from the main script
from phishing_detection_clean import (
    brand_manager,
    TrancoBrandManager,
    SoftVotingEnsemble,
    calculate_similarity,
    shannon_entropy,
    normalize_leetspeak,
    extract_engineered_features,
    high_value_brands_set,
    CACHE_DIR
)

# Import external feature extraction
from external_features import extract_external_features

# =============================================================================
# CONTENT ANALYSIS FOR POST-PROCESSING
# =============================================================================

def analyze_webpage_content(url, timeout=5):
    """
    Extract content-based features from actual webpage.
    This is SEPARATE from the model - used only for post-processing.
    """
    try:
        response = requests.get(url, timeout=timeout, verify=False,
                               headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        domain = urlparse(url).netloc

        # Extract content features
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
    except Exception as e:
        logging.debug(f"Content analysis failed for {url}: {e}")
        return None

def count_external_links(soup, domain):
    """Count links pointing to external domains."""
    external = 0
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http') and domain not in href:
            external += 1
    return external

def apply_content_boost(base_prob, content_features, url_features):
    """
    Adjust base model prediction using content analysis.
    This is post-processing logic, not part of the model.
    """
    if content_features is None:
        return base_prob, 0.0, []

    boost = 0.0
    reasons = []

    # Rule 1: Very few lines of code (likely fake landing page)
    if content_features['line_count'] < 150:
        boost += 0.15
        reasons.append(f"Very few lines ({content_features['line_count']})")

    # Rule 2: No title or favicon (unprofessional)
    if not content_features['has_title'] or not content_features['has_favicon']:
        boost += 0.10
        reasons.append("Missing title/favicon")

    # Rule 3: Password field + suspicious TLD
    if (content_features['has_password_field'] and
        url_features.get('HasSuspiciousTLD', 0) == 1):
        boost += 0.25
        reasons.append("Password field + suspicious TLD")

    # Rule 4: Forms but no HTTPS
    if content_features['has_forms'] and url_features.get('IsHTTPS', 0) == 0:
        boost += 0.20
        reasons.append("Form without HTTPS")

    # Rule 5: Hidden fields + phishing keywords
    if (content_features['has_hidden_fields'] and
        url_features.get('HasPhishingKeywords', 0) > 0):
        boost += 0.15
        reasons.append("Hidden fields + phishing keywords")

    # Rule 6: Too many external links (redirect farm)
    if content_features['external_links'] > 20:
        boost += 0.10
        reasons.append(f"Many external links ({content_features['external_links']})")

    adjusted_prob = min(base_prob + boost, 1.0)

    return adjusted_prob, boost, reasons



def apply_heuristic_boost(base_prob, external_features, url_features):
    """
    Adjust prediction using WHOIS/Age AND aggressive URL heuristics.
    """
    boost = 0.0
    reasons = []
    
    # 1. External Features (Apps if present)
    if external_features:
        # A. Domain Age
        age = external_features.get('DomainAge', 3650)
        if age < 30:
            boost += 0.30
            reasons.append(f"New Domain ({age} days)")
        elif age < 180:
            boost += 0.15
            reasons.append(f"Young Domain ({age} days)")
            
        # B. SSL Analysis (New)
        ssl_age = external_features.get('SSLAgeDays', -1)
        ssl_issuer = external_features.get('SSLIssuer', 'Unknown')
        
        if ssl_age > -1 and ssl_age < 2:
            boost += 0.40
            reasons.append("Brand New SSL (< 48h)")
        
        if 'Let\'s Encrypt' in ssl_issuer and age < 30:
            boost += 0.20
            reasons.append("Free SSL + New Domain")
            
        # C. IP/ASN Analysis (New)
        isp = external_features.get('HostingISP', 'Unknown')
        if any(cloud in isp for cloud in ['DigitalOcean', 'Choopa', 'Vultr', 'Namecheap']):
             if age < 90:
                 boost += 0.15
                 reasons.append(f"Suspicious Hosting ({isp})")

    # 2. Strong URL Heuristics (Brand in Subdomain/Path but NOT Tranco)
    tranco_score = url_features.get('DomainTrancoRank', 0)
    brand_sub = url_features.get('HasBrandInSubdomain', 0)
    brand_path = url_features.get('HasBrandInPath', 0)
    
    if tranco_score == 0:
        if brand_sub > 0:
            boost += 0.35
            reasons.append("Brand in Subdomain (Unverified)")
        if brand_path > 0:
            boost += 0.20
            reasons.append("Brand in Path (Unverified)")
        if url_features.get('IsTyposquatting', 0) > 0:
             # AUTO-FLAG: Typosquatting of a top 500 brand on a non-ranked domain is 100% Phishing
             boost += 1.0 
             reasons.append("Typosquatting Detected (Critical)")
        
    adjusted_prob = min(base_prob + boost, 1.0)
    return adjusted_prob, boost, reasons


# Paths
MODEL_DIR = r'D:\FakingDetection\Models'

# Load model and features
print("[LOAD] Loading model and features...")
model = joblib.load(os.path.join(MODEL_DIR, 'final_ensemble.joblib'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
print(f"[OK] Model loaded. Using {len(feature_names)} features.")

# NOTE: extract_engineered_features, normalize_leetspeak, and high_value_brands_set
# are imported from phishing_detection_clean to avoid code duplication


def predict_url(url: str, verbose: bool = True, use_external_features: bool = False, use_content_boost: bool = False) -> Dict[str, Any]:
    """
    Predict if a URL is phishing or legitimate.
    
    Args:
        url: URL to predict
        verbose: Print detailed output
        use_external_features: If True, extract external features (UrlIsLive, DomainAge, etc.)
        use_content_boost: If True, analyze webpage content for prediction boost (slower)
    
    Returns:
        dict with: prediction, probability, is_phishing, content_boost_info
    """
    try:
        # Step 1: Extract URL-based features
        url_features = extract_engineered_features(url)
        
        if url_features is None:
            return {"error": "Feature extraction failed", "url": url}
        
        # Step 2: Extract external features if requested
        if use_external_features:
            external_feats = extract_external_features(url, fetch_content=False)
            if external_feats:
                url_features.update(external_feats)
        
        # Step 3: Align features with model
        feature_vector = pd.DataFrame([url_features])
        
        # Fill missing features with 0
        for feat in feature_names:
            if feat not in feature_vector.columns:
                feature_vector[feat] = 0
        
        # Select only training features
        feature_vector = feature_vector[feature_names]
        
        # Step 4: Get base model prediction
        proba = model.predict_proba(feature_vector)[0]
        # Model classes: 0=Legitimate, 1=Phishing
        base_phishing_prob = proba[1] if len(proba) > 1 else 1.0 - proba[0]
        
        # Step 5: Post-processing boosts (External + Content)
        # We apply boosts to the PHISHING probability
        final_phishing_prob = base_phishing_prob
        
        # A. Heuristic/External Boost
        heuristic_info = None
        
        # Extract external features if needed (already in url_features)
        ext_feats = url_features if use_external_features else None
             
        # Apply combined heuristic boost
        heur_prob, heur_boost, heur_reasons = apply_heuristic_boost(final_phishing_prob, ext_feats, url_features)
        final_phishing_prob = heur_prob
        
        if heur_boost > 0:
            heuristic_info = {
                'boost_applied': heur_boost,
                'boost_reasons': heur_reasons
            }

        # B. Content Boost
        # NOTE: Typosquatting boost is already handled in apply_heuristic_boost()
        # via url_features['IsTyposquatting'] — no need for redundant re-detection here

        content_info = None
        if use_content_boost:
            content_features = analyze_webpage_content(url)
            if content_features:
                cont_prob, cont_boost, cont_reasons = apply_content_boost(
                    final_phishing_prob, content_features, url_features
                )
                final_phishing_prob = cont_prob
                content_info = {
                    'boost_applied': cont_boost,
                    'boost_reasons': cont_reasons,
                    'base_probability': base_phishing_prob
                }
        
        # Step 6: Make prediction
        is_phishing = final_phishing_prob >= 0.5
        prediction = "PHISHING" if is_phishing else "SAFE"
        
        result = {
            "url": url,
            "prediction": prediction,
            "phishing_probability": float(final_phishing_prob),
            "is_phishing": bool(is_phishing),
            "confidence": float(max(prob_safe, 1-prob_safe)),
            "external_features_used": use_external_features,
            "heuristic_boost_info": heuristic_info,
            "content_boost_used": use_content_boost,
            "content_boost_info": content_info
        }
        
        if verbose:
            status = "[DANGER]" if is_phishing else "[SAFE]"
            print(f"{status} {url}")
            print(f"   Probability: {final_phishing_prob:.2%} (Base: {base_phishing_prob:.2%})")
            if heuristic_info:
                 print(f"   Heuristic Boost: +{heuristic_info['boost_applied']:.2f} ({', '.join(heuristic_info['boost_reasons'])})")
            if content_info:
                print(f"   Content Boost: +{content_info['boost_applied']:.2f} ({', '.join(content_info['boost_reasons'])})")
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"Prediction error: {str(e)}"
        if verbose:
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
        return {"error": error_msg, "url": url}


def predict_url_fast(url: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Fast prediction without external feature extraction or content boost.
    Use this for batch processing or when speed is critical.
    
    Returns:
        dict with: prediction, probability, is_phishing
    """
    return predict_url(url, verbose=verbose, use_external_features=False, use_content_boost=False)


def test_real_urls(use_external_features: bool = False):
    """Test the model with various real URLs."""
    
    print("\n" + "="*70)
    print("REAL URL TESTING")
    print("="*70)
    
    # Test URLs - mix of legitimate and potential phishing
    test_cases = [
        # Legitimate sites
        {"url": "https://www.google.com", "expected": "safe", "category": "Legitimate"},
        {"url": "https://www.microsoft.com", "expected": "safe", "category": "Legitimate"},
        {"url": "https://www.amazon.com", "expected": "safe", "category": "Legitimate"},
        {"url": "https://www.paypal.com", "expected": "safe", "category": "Legitimate"},
        {"url": "https://www.github.com", "expected": "safe", "category": "Legitimate"},
        {"url": "https://www.netflix.com", "expected": "safe", "category": "Legitimate"},
        {"url": "https://www.apple.com", "expected": "safe", "category": "Legitimate"},
        {"url": "https://www.facebook.com", "expected": "safe", "category": "Legitimate"},
        
        # Typosquatting (should be phishing)
        {"url": "https://www.paypa1.com", "expected": "phishing", "category": "Typosquatting"},
        {"url": "https://www.g00gle.com", "expected": "phishing", "category": "Typosquatting"},
        {"url": "https://www.amaz0n.com", "expected": "phishing", "category": "Typosquatting"},
        {"url": "https://www.micros0ft.com", "expected": "phishing", "category": "Typosquatting"},
        
        # Subdomain deception (should be phishing)
        {"url": "https://paypal.com.secure-login.xyz", "expected": "phishing", "category": "Subdomain Deception"},
        {"url": "https://google.com.verify-account.net", "expected": "phishing", "category": "Subdomain Deception"},
        {"url": "https://amazon.login-update.com", "expected": "phishing", "category": "Subdomain Deception"},
        {"url": "https://secure-paypal.malicious.com", "expected": "phishing", "category": "Subdomain Deception"},
        
        # Suspicious keywords (likely phishing)
        {"url": "https://login-verify-account.xyz/paypal", "expected": "phishing", "category": "Suspicious Keywords"},
        {"url": "https://update-your-password.com/google", "expected": "phishing", "category": "Suspicious Keywords"},
        {"url": "https://secure-banking-login.com", "expected": "phishing", "category": "Suspicious Keywords"},
        
        # IP-based URLs (suspicious)
        {"url": "http://192.168.1.1/login", "expected": "phishing", "category": "IP-based URL"},
        {"url": "http://45.33.32.156/paypal/login", "expected": "phishing", "category": "IP-based URL"},
        
        # Random/suspicious domains
        {"url": "https://xk3js9df.xyz/verify", "expected": "phishing", "category": "Random Domain"},
        {"url": "https://a1b2c3d4.net/login", "expected": "phishing", "category": "Random Domain"},
    ]
    
    results = []
    correct = 0
    total = 0
    
    print("\n{:<50} {:<12} {:<12} {:<8}".format("URL", "Expected", "Predicted", "Match"))
    print("-" * 90)
    
    for test in test_cases:
        url = test["url"]
        expected = test["expected"]
        
        result = predict_url(url, verbose=False)
        
        if "error" in result:
            print(f"[ERROR] {url}: {result['error']}")
            continue
        
        predicted = "phishing" if result["is_phishing"] else "safe"
        is_correct = (predicted == expected)
        
        if is_correct:
            correct += 1
        total += 1
        
        status = "[OK]" if is_correct else "[X]"
        
        # Truncate URL for display
        url_display = url[:47] + "..." if len(url) > 50 else url
        
        print(f"{url_display:<50} {expected:<12} {predicted:<12} {status}")
        
        results.append({
            **result,
            "expected": expected,
            "correct": is_correct,
            "category": test["category"]
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Accuracy: {correct}/{total} ({correct/total:.1%})")
    
    # By category
    print("\nBy Category:")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1
    
    for cat, stats in categories.items():
        acc = stats["correct"] / stats["total"]
        print(f"   {cat}: {stats['correct']}/{stats['total']} ({acc:.0%})")
    
    # Failures
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n[WARN] Failures ({len(failures)}):")
        for f in failures:
            print(f"   {f['url']}")
            print(f"      Expected: {f['expected']}, Got: {'phishing' if f['is_phishing'] else 'safe'}")
            print(f"      Probability: {f['phishing_probability']:.2%}")
            
            # DEBUG: Print features causing failure
            print("      Features (Debug):")
            url_features = extract_engineered_features(f['url'])
            # Print only non-zero features or critical ones
            critical_debug = ['HasIPInURL', 'IsTyposquatting', 'HasSuspiciousTLD', 'DomainTrancoRank']
            for k, v in url_features.items():
                if k in critical_debug or v != 0:
                     if isinstance(v, float):
                         print(f"         {k}: {v:.4f}")
                     else:
                         print(f"         {k}: {v}")
    
    return results


if __name__ == "__main__":
    results = test_real_urls(use_external_features=True)
