import glob
import re

html_files = glob.glob('d:/FakingDetection/frontend/*.html')

for filepath in html_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    modified = False

    # 1. Change security check redirect
    if 'if (!username && !email) {' in content and 'TruthLens.html' in content:
        content, count = re.subn(r'(if \(!username && !email\) {\s*window\.location\.href =) \"TruthLens\.html\";', r'\1 "sign_in.html";', content)
        if count > 0: modified = True
        
    # Same check for API.isLoggedIn() if any
    if 'if (!API.isLoggedIn()) {' in content and 'TruthLens.html' in content:
        content, count = re.subn(r'(if \(!API\.isLoggedIn\(\)\) {[\s]*window\.location\.href =) [\'"]TruthLens\.html[\'"];', r'\1 "sign_in.html";', content)
        if count > 0: modified = True

    # 2. Fix technical AI model names
    replacements = {
        'DeBERTa NLI Model': 'AI Engine',
        'DeBERTa-v3-large': 'AI Engine',
        'LoRA Fine-tuned Adapter': 'Advanced Analysis Module',
        'Natural Language Inference (NLI)': 'Contextual Analysis',
        'XGBoost Soft-Voting Ensemble': 'AI Threat Detector'
    }
    
    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Modified: {filepath}')
