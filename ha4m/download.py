"""
بيحمّل Skeletons.zip + Labels.txt لكل فيديو من HA4M (حوالي 2.7 جيجا لكل الـ 217 فيديو؛ مش الصور).

    python ha4m/download.py <فولدر_إخراج> ha4m/videos.txt

لو وقف في النص شغّله تاني: بيكمّل من غير ما يعيد اللي نزل.
"""
import base64, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
OUT = sys.argv[1]   # فولدر الإخراج الفاضي، مثلا ha4m_skel
TOK = "j5BZvG1sXPzh6x1"
vids = [l.strip() for l in open(sys.argv[2]) if l.strip()]
auth = "Basic " + base64.b64encode(f"{TOK}:".encode()).decode()
def get(url, dst, hdr=None):
    for k in range(5):
        try:
            req = urllib.request.Request(url, headers=hdr or {})
            data = urllib.request.urlopen(req, timeout=300).read()
            open(dst + '.part', 'wb').write(data); os.replace(dst + '.part', dst); return True
        except Exception as e:
            time.sleep(5 * (k + 1))
    return False
def job(v):
    d = os.path.join(OUT, v); os.makedirs(d, exist_ok=True)
    ok = True
    lb = os.path.join(d, 'Labels.txt')
    if not os.path.exists(lb): ok &= get(f"https://cloud.cnr.it/owncloud/public.php/webdav/{v}/Labels.txt", lb, {"Authorization": auth})
    sk = os.path.join(d, 'Skeletons.zip')
    if not os.path.exists(sk): ok &= get(f"https://cloud.cnr.it/owncloud/index.php/s/{TOK}/download?path=%2F{v}&files=Skeletons", sk)
    return v, ok
done = 0
with ThreadPoolExecutor(3) as ex:
    for v, ok in ex.map(job, vids):
        done += 1; print(done, len(vids), v, 'ok' if ok else 'FAILED', flush=True)
print('DONE', flush=True)
