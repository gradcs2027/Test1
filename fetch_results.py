"""ينزّل ناتج آخر رن من Kaggle في مجلد نتايج.

الاستخدام:  python fetch_results.py results_vidtest2

بيستخدم الـ REST API مباشرة مش `kaggle kernels output` عشان ده بينزّل كل
حاجة في الـ /kaggle/working — ومنها ntu60_2d.pkl بحجم 705MB. هنا بنختار
الملفات اللي عايزينها بس.
"""
import base64
import io
import json
import os
import re
import sys
import urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else 'results'
KERNEL = 'abdallahhsamir/notebook520b8d324a'
WANT = ['final_timeline.png', 'confusion_matrix_skeleton.png', 'annotated_output.mp4']

cfg = json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))
AUTH = {'Authorization': 'Basic ' + base64.b64encode(
    f"{cfg['username']}:{cfg['key']}".encode()).decode()}
os.makedirs(OUT, exist_ok=True)

user, slug = KERNEL.split('/')
req = urllib.request.Request(
    f'https://www.kaggle.com/api/v1/kernels/output?user_name={user}&kernel_slug={slug}',
    headers=AUTH)
data = json.load(urllib.request.urlopen(req))

log = data.get('log', '')
if isinstance(log, str):
    try:
        log = json.loads(log)
    except Exception:
        pass
lines = [str(e.get('data', '')) for e in log] if isinstance(log, list) else str(log).split('\n')
txt = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', '\n'.join(lines))   # شيل ألوان ANSI

# اللوج فيه ميجابايتات من progress bars بتاعة Keras — بناخد السطور المهمة بس
KEYS = ['Test Accuracy', 'درجة الحرارة', 'اترفضت بالعتبة', 'الإجمالي النهائي other',
        'السيجمنتس', 'دقة إجمالية', 'خط الأساس', 'other recall', 'real recall',
        'الحجم', 'اتحفظ بنجاح', 'المستخدمة', 'مستبعدة', 'فريمات', 'FRAME_SKIP']
key = [l for l in txt.split('\n') if any(k in l for k in KEYS) and len(l) < 220]
sweep = [l for l in txt.split('\n') if re.match(r'\s+(0\.\d0|1\.01)\s+\d+\.\d%', l)]
seg = [l for l in txt.split('\n') if re.match(r'\s*\d+\.\d+s\s+\d+\.\d+s', l)]

io.open(f'{OUT}/key.txt', 'w', encoding='utf-8').write(
    '\n'.join(key) + '\n--- SWEEP ---\n' + '\n'.join(sweep))
io.open(f'{OUT}/seg.txt', 'w', encoding='utf-8').write('\n'.join(seg))
print(f'key {len(key)} | sweep {len(sweep)} | seg {len(seg)}')

for name in WANT:
    r = urllib.request.Request(
        f'https://www.kaggle.com/api/v1/kernels/output/download/{KERNEL}/{name}',
        headers=AUTH)
    blob = urllib.request.urlopen(r).read()
    open(os.path.join(OUT, name), 'wb').write(blob)
    print(name, round(len(blob) / 1e6, 1), 'MB', flush=True)
