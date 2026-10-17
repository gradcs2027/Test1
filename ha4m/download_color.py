"""
بيحمّل فريمات ColorAligned لفيديو واحد من HA4M، كل فريم تاني بس (15 فريم/ث زي التقييم).
حوالي 0.63 جيجا للفيديو بدل 8 جيجا لـ Color كامل.

    python -I ha4m/download_color.py <فولدر_إخراج_فاضي> IDU001V002 [كل_كام_فريم=2] [threads=4]

كل_كام_فريم = 6 يعني 5 فريم/ث (الفريمات اللي رقمها FrameID بيقبل القسمة على 6).
لو وقف شغّله تاني: بيكمّل من غير ما يعيد اللي نزل.
"""
import base64, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT, VID = sys.argv[1], sys.argv[2]
EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 2
THREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
TOK = "j5BZvG1sXPzh6x1"
BASE = "https://cloud.cnr.it/owncloud/public.php/webdav"
AUTH = {"Authorization": "Basic " + base64.b64encode(f"{TOK}:".encode()).decode()}


def fetch(url, method=None, hdr=None):
    req = urllib.request.Request(url, headers={**AUTH, **(hdr or {})}, method=method)
    return urllib.request.urlopen(req, timeout=300).read()


def ls(path):
    t = fetch(f"{BASE}/{path}/", 'PROPFIND', {"Depth": "1"}).decode()
    return [h for h in re.findall(r'<d:href>(.*?)</d:href>', t)][1:]


sub = ls(f"{VID}/ColorAligned")[0].rstrip('/').split('/')[-1]
names = sorted(h.split('/')[-1] for h in ls(f"{VID}/ColorAligned/{sub}") if h.endswith('.png'))
names = names[::2] if EVERY == 2 else [n for n in names if int(re.search(r'FrameID(\d+)', n).group(1)) % EVERY == 0]
os.makedirs(OUT, exist_ok=True)
print(len(names), 'frames', flush=True)


def job(n):
    dst = os.path.join(OUT, n)
    if os.path.exists(dst):
        return True
    for k in range(5):
        try:
            data = fetch(f"{BASE}/{VID}/ColorAligned/{sub}/{n}")
            open(dst + '.part', 'wb').write(data); os.replace(dst + '.part', dst)
            return True
        except Exception:
            time.sleep(3 * (k + 1))
    return False


done = bad = 0
with ThreadPoolExecutor(THREADS) as ex:
    for ok in ex.map(job, names):
        done += 1; bad += not ok
        if done % 100 == 0:
            print(done, len(names), 'failed', bad, flush=True)
print('DONE', done, 'failed', bad, flush=True)
