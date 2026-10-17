"""
بيخلّي `shared/` قابلة للاستيراد من الفولدر ده.

أي سكريبت هنا محتاج `paths` أو `ground_truth` أو `skeleton_norm` بيكتب
دي كأول استيراد محلي:

    import _bootstrap  # noqa: F401

ليه ملف مخصوص بدل ما كل سكريبت يعدّل sys.path بنفسه؟ عشان ده كان هيبقى
تلات سطور مكررة في 7 ملفات — نفس نوع التكرار اللي `paths.py` اتعمل عشان
يخلص عليه.
"""

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parent.parent / 'shared'

if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
