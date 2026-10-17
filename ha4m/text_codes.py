"""بيحسب أكواد MiniLM لوصف أفعال HA4M ويخزنها:  python ha4m/text_codes.py  (من غير -I، محتاج packages المستخدم)"""
import os, sys
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentence_transformers import SentenceTransformer
import importlib.util
src = (Path(__file__).parent / 'train_models.py').read_text(encoding='utf-8')
ns = {}
exec(src[src.index('DESC = ['):src.index('class Attn')], ns)
st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
from prepare import OUT
np.save(OUT / 'text_codes.npy', st.encode(ns['DESC'], normalize_embeddings=True))
print('saved', OUT / 'text_codes.npy')
