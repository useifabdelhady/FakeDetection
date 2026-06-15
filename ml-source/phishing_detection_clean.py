# -*- coding: utf-8 -*-
"""
================================================================================
PHISHING WEBSITE DETECTION - CLEAN IMPLEMENTATION
================================================================================
Comprehensive phishing/fake website detection using machine learning.

Features:
- 56 original dataset features
- 35+ engineered features (brand detection, entropy, lexical analysis)
- Ensemble model (XGBoost + Random Forest + Logistic Regression)
- Adversarial robustness

Author: Yousef
Date: 2026
================================================================================
"""

# =============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import joblib
import os
import re
import math
import logging
from collections import Counter
from typing import Dict, Tuple, Optional, List, Any
from urllib.parse import urlparse
import tldextract

# ML imports
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, f1_score, average_precision_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

# Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
DATA_PATH = r'D:\FakingDetection\Data\PhiUSIIL_Phishing_URL_Dataset.csv'
OUTPUT_DIR = r'D:\FakingDetection\Models'
CACHE_DIR = r'D:\FakingDetection\Data\cache'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Webpage-dependent features to EXCLUDE (cannot be extracted from URL alone)
WEBPAGE_FEATURES = [
    'URLSimilarityIndex', 'LineOfCode', 'LargestLineLength',
    'NoOfSelfRef', 'NoOfExternalRef', 'NoOfImage', 'NoOfCSS', 'NoOfJS',
    'NoOfSelfRedirect', 'NoOfEmptyRef', 'NoOfImgaeNonOrigional',
    'HasTitle', 'HasDescription', 'HasFavicon', 'HasSocialNet',
    'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField',
    'Bank', 'Pay', 'Crypto', 'HasCopyrightInfo',
    'NoOfPopup', 'NoOfExternalFormSubmit', 'RatioOfIntRedirection',
    'RatioOfExtRedirection', 'RatioOfResources', 'NoIframeRedirection',
    'HasResponseTime', 'WebsiteTraffic', 'PageRank', 'GoogleIndex',
    'LinksPointingToPage', 'IsResponsive'
]

print("[OK] All imports loaded successfully!")
print(f"[INFO] Excluding {len(WEBPAGE_FEATURES)} webpage-dependent features from training")


# =============================================================================
# SECTION 2: BRAND MANAGER (TRANCO OPTIMIZED)
# =============================================================================

class TrancoBrandManager:
    """Manages top domains using Tranco list for fast, accurate brand detection."""
    
    def __init__(self, cache_dir: str = None, top_n: int = 10000):
        self.cache_dir = cache_dir or CACHE_DIR
        self.top_n = top_n
        self.top_domains = set()
        self.brand_domains = []  # List of high-value targets for typosquatting checks
        self.confusables_map = self._load_confusables()
        self._load_tranco_list()
    
    def _load_confusables(self) -> Dict[str, str]:
        return {
            '0': 'o', '1': 'l', '3': 'e', '5': 's', '8': 'b',
            'i': 'l', 'I': 'l', 'O': 'o', '@': 'a'
        }
    
    def _load_tranco_list(self):
        """Download or load Tranco list."""
        list_file = os.path.join(self.cache_dir, 'tranco_top_1m.csv')
        
        if not os.path.exists(list_file):
            print("[INFO] Tranco list not found. Downloading...")
            try:
                import requests
                import zipfile
                import io
                
                url = "https://tranco-list.eu/top-1m.csv.zip"
                r = requests.get(url, stream=True)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(self.cache_dir)
                # Rename extracted file (usually top-1m.csv)
                if os.path.exists(os.path.join(self.cache_dir, 'top-1m.csv')):
                    os.rename(os.path.join(self.cache_dir, 'top-1m.csv'), list_file)
                print("[OK] Tranco list downloaded.")
            except Exception as e:
                print(f"[WARN] Failed to download Tranco list: {e}. Using fallback.")
                self._load_fallback_brands()
                return

        # Load top N domains
        try:
            print(f"[LOAD] Loading top {self.top_n} domains from Tranco...")
            df = pd.read_csv(list_file, header=None, names=['rank', 'domain'], nrows=self.top_n)
            self.top_domains = set(df['domain'].astype(str).str.lower().str.strip())
            
            # Create a smaller list of high-value targets for expensive typosquatting checks
            # We take the top 500 domains that look like brands (exclude simple infra)
            self.brand_domains = list(df['domain'].head(500).astype(str).tolist())
            print(f"[OK] Loaded {len(self.top_domains)} domains. Brand targets: {len(self.brand_domains)}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load Tranco list: {e}")
            self._load_fallback_brands()

    def _load_fallback_brands(self):
        """Fallback list if download fails."""
        brands = [
            'google.com', 'youtube.com', 'facebook.com', 'amazon.com',
            'twitter.com', 'instagram.com', 'linkedin.com', 'microsoft.com',
            'apple.com', 'netflix.com', 'paypal.com', 'github.com'
        ]
        self.top_domains = set(brands)
        self.brand_domains = brands
        print(f"[WARN] Using {len(brands)} fallback brands.")

    def get_rank(self, domain: str) -> int:
        """Return 1 if in top N, else 0."""
        return 1 if domain in self.top_domains else 0

    def is_known_brand(self, domain: str) -> bool:
        return domain in self.top_domains

    def get_high_value_brands(self) -> List[str]:
        return self.brand_domains


# =============================================================================
# SECTION 3: UTILITY FUNCTIONS
# =============================================================================

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((count/length) * math.log2(count/length) for count in counts.values())

def calculate_similarity(s1: str, s2: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, s1, s2).ratio() if s1 and s2 else 0.0

print("[OK] Utility functions loaded!")

# =============================================================================
# SECTION 4: FEATURE EXTRACTION (STRICT & LEAK-FREE)
# =============================================================================

# Optimization: Global extractor
try:
    # Try with default sources first (will update cache if Internet available)
    tld_extractor = tldextract.TLDExtract(include_psl_private_domains=True, cache_dir=CACHE_DIR)
    tld_extractor('google.com') # Trigger download/update check
except Exception as e:
    print(f"[WARN] TLD Update failed ({e}), using cached/internal list only.")
    # Fallback: Do not attempt to download
    tld_extractor = tldextract.TLDExtract(include_psl_private_domains=True, cache_dir=CACHE_DIR, suffix_list_urls=[])

brand_manager = TrancoBrandManager(top_n=10000)

# Optimization: Pre-process high value brands for fast substring checks
# EXCLUDE 'www' to prevent false positives on standard subdomains
high_value_brands_set = set(b.split('.')[0] for b in brand_manager.get_high_value_brands() if b.split('.')[0] != 'www')

def normalize_leetspeak(text: str) -> str:
    """Normalize simple leetspeak (0->o, 1->l, 3->e, @->a)."""
    replacements = {
        '0': 'o', '1': 'l', '3': 'e', '4': 'a', '@': 'a', 
        '5': 's', '7': 't', '$': 's', '!': 'i'
    }
    for char, rep in replacements.items():
        text = text.replace(char, rep)
    return text

def extract_engineered_features(url: str) -> Optional[Dict[str, Any]]:
    """
    Extract strictly URL-based features. 
    NO CSV lookups, NO external calls, NO leakage.
    Matching logic: Fast regex & math only.
    """
    features = {}
    
    try:
        if not url or not isinstance(url, str): return None
        
        # 1. Parsing
        url = url.strip() # Keep case for some features, but mostly lower needed
        url_lower = url.lower()
        if not url_lower.startswith(('http://', 'https://')):
            url_lower = 'http://' + url_lower
            
        parsed = urlparse(url_lower)
        ext = tld_extractor(url_lower) # Use global instance
        
        domain_name = ext.domain
        suffix = ext.suffix
        subdomain = ext.subdomain
        domain_full = f"{domain_name}.{suffix}" if suffix else domain_name
        
        # --- A. STRUCTURAL FEATURES (Fast) ---
        features['URLLength'] = len(url)
        features['DomainLength'] = len(domain_full)
        features['SubdomainLength'] = len(subdomain)
        features['PathLength'] = len(parsed.path)
        features['QueryLength'] = len(parsed.query)
        
        features['NoOfSubDomain'] = subdomain.count('.') + 1 if subdomain else 0
        features['IsHTTPS'] = 1 if parsed.scheme == 'https' else 0
        
        # Counts
        for char in ['@', '-', '_', '/', ':', '~', '.', '=', '?', '&']:
            name = f'NoOf{char}InURL' if char not in ['@', '.', '/', ':'] else f'NoOf{char}Symbol' # Naming fix
            if char == '.': name = 'NoOfDotInURL'
            if char == '/': name = 'NoOfSlashInURL'
            if char == ':': name = 'NoOfColonInURL'
            if char == '@': name = 'NoOfAtSymbol'
            features[name] = url.count(char)
            
        features['NoOfDegitsInURL'] = sum(c.isdigit() for c in url)
        features['NoOfLettersInURL'] = sum(c.isalpha() for c in url)
        
        # Ratios
        total_len = max(len(url), 1)
        features['DegitRatioInURL'] = features['NoOfDegitsInURL'] / total_len
        features['LetterRatioInURL'] = features['NoOfLettersInURL'] / total_len
        features['SpacialCharRatioInURL'] = (total_len - features['NoOfDegitsInURL'] - features['NoOfLettersInURL']) / total_len

        # --- B. ENTROPY & RANDOMNESS (New) ---
        features['DomainEntropy'] = shannon_entropy(domain_full)
        features['PathEntropy'] = shannon_entropy(parsed.path)
        
        # Vowel/Consonant Ratio (New - good for random domains)
        vowels = set('aeiou')
        v_count = sum(1 for c in domain_name if c in vowels)
        c_count = sum(1 for c in domain_name if c.isalpha() and c not in vowels)
        features['VowelRatio'] = v_count / max(len(domain_name), 1)
        features['ConsonantRatio'] = c_count / max(len(domain_name), 1)
        
        # Hex check (New - obfuscation)
        features['HasHexChars'] = 1 if '%' in url or re.search(r'0x[0-9a-fA-F]', url) else 0

        # --- C. BRAND & TRUST (Tranco) ---
        # 1. Tranco Rank (Top 10k = Trust)
        features['DomainTrancoRank'] = brand_manager.get_rank(domain_full)
        
        # 2. Typosquatting & Brand Presence
        is_typo = 0
        max_sim = 0.0
        normalized_matches_brand = 0
        
        # Check if brand name appears in subdomain or path (Strong Phishing Signal)
        features['HasBrandInSubdomain'] = 0
        features['HasBrandInPath'] = 0
        
        # Only check expensive overlaps if we have a subdomain or path
        if subdomain:
            for brand in high_value_brands_set:
                if len(brand) >= 4 and brand in subdomain:
                    features['HasBrandInSubdomain'] = 1
                    break
        
        if parsed.path and len(parsed.path) > 1:
            for brand in high_value_brands_set:
                if len(brand) >= 4 and brand in parsed.path:
                    features['HasBrandInPath'] = 1
                    break

        # --- NEW: Leetspeak detection features ---
        leet_chars = set('0134579$!@')
        leet_count = sum(1 for c in domain_name if c in leet_chars)
        features['HasLeetspeakChars'] = 1 if leet_count > 0 else 0
        features['LeetspeakCharCount'] = leet_count
        
        # Domain mixes digits and letters (not pure alpha or pure numeric)
        has_digits = any(c.isdigit() for c in domain_name)
        has_letters = any(c.isalpha() for c in domain_name)
        features['DomainHasDigitLetterMix'] = 1 if (has_digits and has_letters) else 0

        # Typosquatting checks (only if not top 10k)
        if features['DomainTrancoRank'] == 0:
            # Normalize leetspeak (g00gle -> google)
            domain_norm = normalize_leetspeak(domain_name)
            
            for brand in high_value_brands_set:
                # Quick length check optimization
                if abs(len(domain_name) - len(brand)) > 3:
                     continue
                
                # CASE 1: Perfect normalized match (g00gle -> google == google)
                # This is the STRONGEST typosquatting signal
                if domain_norm == brand and domain_name != brand:
                    is_typo = 1
                    normalized_matches_brand = 1
                    max_sim = 1.0  # Perfect match after normalization
                    break
                
                # CASE 2: High similarity (paypa1 vs paypal = 0.83)
                sim = calculate_similarity(domain_name, brand)
                sim_norm = calculate_similarity(domain_norm, brand)
                best_sim = max(sim, sim_norm)
                
                if best_sim > 0.80 and domain_name != brand:
                    is_typo = 1
                    max_sim = max(max_sim, best_sim)
                    # Break early if strong match found
                    if best_sim > 0.95: break
        
        features['IsTyposquatting'] = is_typo
        features['MaxBrandSimilarity'] = max_sim
        features['NormalizedMatchesBrand'] = normalized_matches_brand
        
        # Interaction feature: not in Tranco but looks like a brand
        features['IsNotTrancoButSimilar'] = (1 - features['DomainTrancoRank']) * max_sim

        # --- D. SECURITY & SUSPICION ---
        suspicious_tlds = {'xyz', 'top', 'club', 'online', 'vip', 'cc', 'tk', 'ml', 'ga', 'cf'}
        features['HasSuspiciousTLD'] = 1 if suffix in suspicious_tlds else 0
        
        shorteners = {'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'is.gd'}
        features['IsShortenedURL'] = 1 if domain_full in shorteners else 0
        
        features['HasIPInURL'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
        features['HasDoubleSlashInPath'] = 1 if '//' in parsed.path else 0
        
        # Keywords
        phish_kw = {'login', 'verify', 'update', 'secure', 'account', 'banking', 'confirm'}
        if features.get('DomainTrancoRank', 0) == 1:
            features['HasPhishingKeywords'] = 0
        else:
            features['HasPhishingKeywords'] = sum(1 for kw in phish_kw if kw in url_lower)
        
        return features
        
    except Exception as e:
        return None

print("[OK] Feature extraction functions ready!")

# =============================================================================
# SECTION 5: DATA PROCESSING (LEAK-FREE)
# =============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    print("\n[DATA] Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"[OK] Loaded {len(df):,} rows")
    return df

def process_dataframe(df: pd.DataFrame, name: str = "Data", use_cache: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Process dataframe calculating ONLY engineered features on the fly.
    Ignores ALL CSV columns except 'URL' and 'label'.
    """
    from tqdm import tqdm
    
    # Cache logic kept simple
    cache_path_X = os.path.join(CACHE_DIR, f'{name}_X_strict.pkl')
    cache_path_y = os.path.join(CACHE_DIR, f'{name}_y_strict.pkl')
    
    if use_cache and os.path.exists(cache_path_X) and os.path.exists(cache_path_y):
        print(f"[LOAD] Loading {name} from cache (strict)...")
        return pd.read_pickle(cache_path_X), pd.read_pickle(cache_path_y)
    
    print(f"\n[WAIT] Processing {name}: {len(df):,} URLs (Strict Calculation)")
    
    features_list = []
    labels = []
    
    # Limit for testing speed if needed, but we do full here
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting features"):
        url = row['URL']
        label = row['label']
        
        feats = extract_engineered_features(url)
        if feats:
            features_list.append(feats)
            labels.append(label)
            
    X = pd.DataFrame(features_list)
    y = pd.Series(labels, name='label')
    
    # Fill any NaNs (rare)
    X = X.fillna(0)
    
    print(f"[OK] {name} complete. Shape: {X.shape}")
    
    if use_cache:
        X.to_pickle(cache_path_X)
        y.to_pickle(cache_path_y)
        
    return X, y


print("[OK] Data processing ready!")


# =============================================================================
# SECTION 6: FEATURE SELECTION
# =============================================================================

def select_features(X_train, y_train, X_test, min_feat=35, max_feat=50):
    """Intelligent feature selection."""
    print(f"\n[TARGET] Feature Selection")
    print(f"   Input: {X_train.shape[1]} features")
    
    # Remove NaN columns
    non_nan = X_train.columns[X_train.notna().any()].tolist()
    X_train = X_train[non_nan]
    X_test = X_test[non_nan]
    print(f"   After NaN removal: {len(non_nan)}")
    
    # Impute
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    # Variance threshold
    selector = VarianceThreshold(threshold=0.0001) # Lower threshold
    X_train_var = selector.fit_transform(X_train_imp)
    X_test_var = selector.transform(X_test_imp)
    
    kept_idx = selector.get_support(indices=True)
    kept_names = X_train_imp.columns[kept_idx].tolist()
    
    # Force keep critical security features
    critical_features = [
        'HasIPInURL', 'IsTyposquatting', 'HasSuspiciousTLD', 
        'HasPhishingKeywords', 'DomainTrancoRank', 'HasHexChars',
        'HasDoubleSlashInPath', 'IsShortenedURL', 'IsHTTPS',
        'HasLeetspeakChars', 'LeetspeakCharCount', 'DomainHasDigitLetterMix',
        'NormalizedMatchesBrand', 'IsNotTrancoButSimilar', 'MaxBrandSimilarity'
    ]
    
    for feat in critical_features:
        if feat in X_train.columns and feat not in kept_names:
            print(f"   [FORCE KEEP] {feat} (Low Variance but Critical)")
            kept_names.append(feat)
    
    # Re-build dataframes with forced features
    X_train_clean = X_train_imp[kept_names]
    X_test_clean = X_test_imp[kept_names]
    
    # XGBoost importance
    if len(kept_names) > max_feat:
        print(f"   Running XGBoost importance...")
        xgb = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(X_train_clean, y_train)
        
        importances = xgb.feature_importances_
        top_idx = np.argsort(importances)[-max_feat:][::-1]
        final_names = [kept_names[i] for i in top_idx]
        
        X_train_final = X_train_clean[final_names]
        X_test_final = X_test_clean[final_names]
    else:
        final_names = kept_names
        X_train_final = X_train_clean
        X_test_final = X_test_clean
    
    print(f"   [OK] Final: {len(final_names)} features")
    print(f"   Top 5: {final_names[:5]}")
    
    return X_train_final, X_test_final, final_names


print("[OK] Feature selection ready!")


# =============================================================================
# SECTION 7: MODEL TRAINING
# =============================================================================

class SoftVotingEnsemble:
    """Custom soft voting ensemble."""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1] * len(models)
    
    def predict_proba(self, X):
        probas = [m.predict_proba(X) for m in self.models]
        total = sum(self.weights)
        return sum(w * p for w, p in zip(self.weights, probas)) / total
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def train_model(X_train, y_train, X_test, y_test):
    """Train ensemble model."""
    print("\n[ML] Training Ensemble Model...")
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    print(f"   Class weights: {dict(zip(classes, weights))}")
    
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # XGBoost
    print("   Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=weights[1]/weights[0],
        random_state=42, eval_metric='logloss',
        early_stopping_rounds=30, verbosity=0
    )
    xgb.fit(X_train_sub, y_train_sub, eval_set=[(X_val, y_val)], verbose=False)
    
    # Random Forest
    print("   Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Logistic Regression
    print("   Training Logistic Regression...")
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42, C=0.1))
    ])
    lr.fit(X_train, y_train)
    
    # Ensemble
    ensemble = SoftVotingEnsemble([xgb, rf, lr], weights=[3, 2, 1])
    
    # Evaluate
    print("\n[DATA] Model Evaluation:")
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }
    
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"   MCC:       {metrics['mcc']:.4f}")
    
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    return ensemble, metrics


print("[OK] Model training ready!")


# =============================================================================
# SECTION 8: MAIN
# =============================================================================

def main():
    """Main execution."""
    print("\n" + "="*70)
    print("PHISHING DETECTION MODEL TRAINING")
    print("="*70)
    
    df = load_data(DATA_PATH)
    
    # Domain groups for leak-free splitting
    extract_fn = tldextract.TLDExtract(include_psl_private_domains=True)
    df['domain_group'] = df['URL'].apply(lambda x: extract_fn(str(x)).domain)
    
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df['domain_group']))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    print(f"\n[DATA] Split: Train={len(train_df):,}, Test={len(test_df):,}")
    
    # Process with caching enabled
    X_train, y_train = process_dataframe(train_df, "train", use_cache=True)
    X_test, y_test = process_dataframe(test_df, "test", use_cache=True)
    
    X_train_sel, X_test_sel, feature_names = select_features(X_train, y_train, X_test)
    
    model, metrics = train_model(X_train_sel, y_train, X_test_sel, y_test)
    
    print("\n[SAVE] Saving model...")
    joblib.dump(model, os.path.join(OUTPUT_DIR, 'final_ensemble.joblib'))
    joblib.dump(feature_names, os.path.join(OUTPUT_DIR, 'feature_names.joblib'))
    
    print(f"[OK] Model saved to {OUTPUT_DIR}")
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    return model, feature_names, metrics


if __name__ == "__main__":
    model, features, metrics = main()
