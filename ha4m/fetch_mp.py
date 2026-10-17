"""
الحل 2: فيديو واحد لكل شخص (أقل رقم فيديو)، صور ColorAligned بـ 5 فريم/ث، وبعدين MediaPipe على كل فيديو.

    python -I ha4m/fetch_mp.py [threads=4]

  - الشخص 1: IDU001V001 من الـ zip اللي عندنا (مفيش تحميل).
  - الباقي: download_color.py بـ كل_كام_فريم = 6، كل فيديو في فولدر لوحده جوه Downloads/HA4M_MP5.
  - الترتيب: أشخاص الـ test، بعدين val، بعدين train (لو وقف في النص، التقييم يبقى جاهز).
  - MediaPipe بيشتغل على الفيديو اللي خلص تحميله وإحنا بنحمّل اللي بعده (3 في نفس الوقت بالكتير).
  - لو وقف شغّله تاني: الفيديو اللي ليه npz بيتنط، والتحميل بيكمّل من مكانه.
المخرج: _scratch/ha4m_kp/mp5/<VID>.npz (زي extract_pose.py).
"""
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
SKEL = PROJ / '_scratch' / 'ha4m_skel'
KP = PROJ / '_scratch' / 'ha4m_kp' / 'mp5'
IMG = Path.home() / 'Downloads' / 'HA4M_MP5'
ZIP1 = Path.home() / 'Downloads' / 'IDU001V001.zip'
MP_PY = PROJ / '_scratch' / 'mp_env' / 'Scripts' / 'python.exe'
TASK = PROJ / '_scratch' / 'mp_models' / 'pose_landmarker_heavy.task'
EVERY, MAX_MP = 6, 3
THREADS = sys.argv[1] if len(sys.argv) > 1 else '4'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)


def videos():
    split = json.loads((HERE / 'results' / 'split.json').read_text())
    first = {}
    for d in sorted(p for p in SKEL.iterdir() if (p / 'Labels.txt').exists()):
        first.setdefault(int(d.name[3:6]), d.name)
    order = [s for part in ('test', 'val', 'train') for s in split[part]]
    return [first[s] for s in order if s in first]


def copy_from_zip(vid, out):
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP1)
    n = 0
    for name in z.namelist():
        m = re.search(r'FrameID(\d+)', name)
        if '/ColorAligned/' in name and name.endswith('.png') and m and int(m.group(1)) % EVERY == 0:
            dst = out / name.split('/')[-1]
            if not dst.exists():
                dst.write_bytes(z.read(name))
            n += 1
    print(vid, 'من الـ zip:', n, 'فريم', flush=True)


def main():
    KP.mkdir(parents=True, exist_ok=True)
    vids = videos()
    print(len(vids), 'فيديو:', ' '.join(vids), flush=True)
    running = []
    for i, vid in enumerate(vids, 1):
        npz = KP / f'{vid}.npz'
        if npz.exists():
            print(f'[{i}/{len(vids)}] {vid} موجود', flush=True)
            continue
        out = IMG / vid
        t0 = time.time()
        if vid == 'IDU001V001':
            copy_from_zip(vid, out)
        else:
            r = subprocess.run([sys.executable, '-I', str(HERE / 'download_color.py'), str(out), vid, str(EVERY), THREADS],
                               capture_output=True, text=True)
            last = (r.stdout.strip().splitlines() or ['?'])[-1]
            if r.returncode != 0 or 'failed 0' not in last:
                print(f'[{i}/{len(vids)}] {vid} التحميل فشل: {last} {r.stderr[-300:]}', flush=True)
                continue
        mb = sum(f.stat().st_size for f in out.glob('*.png')) / 1e6
        print(f'[{i}/{len(vids)}] {vid} اتحمّل {mb:.0f} ميجا في {time.time()-t0:.0f}s ({mb/max(1, time.time()-t0):.2f} ميجا/ث)', flush=True)
        while len([p for p in running if p.poll() is None]) >= MAX_MP:
            time.sleep(5)
        log = open(KP / f'{vid}.log', 'w')
        running.append(subprocess.Popen([str(MP_PY), '-I', str(HERE / 'extract_pose.py'), 'mp', str(out),
                                         str(SKEL / vid / 'Labels.txt'), str(npz), str(TASK)], stdout=log, stderr=subprocess.STDOUT))
    for p in running:
        p.wait()
    done = sorted(p.stem for p in KP.glob('*.npz'))
    print('DONE fetch_mp', len(done), '/', len(vids), flush=True)


if __name__ == '__main__':
    main()
