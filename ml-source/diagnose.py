import joblib, pandas as pd, sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'd:\FakingDetection\ml-source')
from phishing_detection_clean import extract_engineered_features

from phishing_detection_clean import SoftVotingEnsemble
model = joblib.load(r'd:\FakingDetection\ml-models\final_ensemble.joblib')
fn = joblib.load(r'd:\FakingDetection\ml-models\feature_names.joblib')

test_urls = [
    'https://github.com',
    'https://accounts.google.com', 
    'http://www.bbc.com',
    'https://www.paypal.com.security.alert.tk',
    'https://www.paypal.com.us',
]

for url in test_urls:
    feats = extract_engineered_features(url)
    fv = pd.DataFrame([feats])
    for f in fn:
        if f not in fv.columns:
            fv[f] = 0
    fv = fv[fn]
    
    pl = feats.get("PathLength", "?")
    sl = feats.get("NoOfSlashInURL", "?")
    print(f"\n=== {url} ===")
    print(f"  PathLength={pl}, NoOfSlashInURL={sl}")
    
    # Individual models
    for i, (m, w) in enumerate(zip(model.models, model.weights)):
        name = type(m).__name__
        if hasattr(m, 'steps'):
            name = 'LR_Pipeline'
        p = m.predict_proba(fv)[0]
        print(f"  Model {i} ({name}, w={w}): P(class0)={p[0]:.4f} P(class1)={p[1]:.4f}")
    
    # Ensemble
    ep = model.predict_proba(fv)[0]
    print(f"  Ensemble: P(class0)={ep[0]:.4f} P(class1)={ep[1]:.4f}")
    
    # Now with PathLength=0 override (as app.py does)
    fv2 = fv.copy()
    fv2['PathLength'] = 0
    fv2['NoOfSlashInURL'] = 2
    ep2 = model.predict_proba(fv2)[0]
    print(f"  Ensemble (w/ override): P(class0)={ep2[0]:.4f} P(class1)={ep2[1]:.4f}")
