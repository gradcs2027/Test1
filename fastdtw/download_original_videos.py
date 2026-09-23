"""
تنزيل الفيديوهات الأصلية (vidtest1, 2, 3, 4) في فولدر جديد

الفيديوهات دي موجودة في:
- Kaggle dataset: gradcs2027/testvid
- أو: /kaggle/input/testvid_upload/ (لو متوصّلة)

بتنزّل الفيديوهات في فولدر:
./original_videos/
├─ vidtest1.mp4
├─ vidtest2.mp4
├─ vidtest3.mp4
└─ vidtest4.mp4

التشغيل: python download_original_videos.py
"""

import shutil
from pathlib import Path

# الفولدرات المحتملة للفيديوهات
VIDEO_SOURCES = [
    Path('/kaggle/input/testvid_upload'),
    Path('/kaggle/input/testvid'),
    Path('/kaggle/input/gradcs2027/testvid'),
    Path('testvid_upload'),
    Path('testvid'),
]

# الفيديوهات المطلوبة
VIDEOS = ['vidtest1', 'vidtest2', 'vidtest3', 'vidtest4']

# الفولدر الجديد
OUTPUT_DIR = Path('original_videos')


def find_video_source():
    """ادوّر على الفيديوهات في المسارات المختلفة"""
    for source in VIDEO_SOURCES:
        if source.exists():
            # تحقق إن الفيديوهات موجودة فيها
            video_files = list(source.glob('*.mp4'))
            if video_files:
                print(f"✅ لقيت فيديوهات في: {source}")
                return source

    print("❌ مالقتش الفيديوهات الأصلية في أي من المسارات:")
    for source in VIDEO_SOURCES:
        print(f"   - {source}")
    return None


def download_videos():
    """نزّل الفيديوهات في فولدر جديد"""

    print("="*70)
    print("🎬 تنزيل الفيديوهات الأصلية")
    print("="*70)

    # ادوّر على المصدر
    source = find_video_source()
    if not source:
        print("\n⚠️ الفيديوهات الأصلية مش موجودة على الجهاز")
        print("\nالحل: على Kaggle Notebook، اضغط + Add Data واختر:")
        print("   gradcs2027/testvid")
        return

    # اعمل الفولدر الجديد
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n✅ فولدر الإخراج: {OUTPUT_DIR.absolute()}")

    # نزّل الفيديوهات
    print(f"\n📥 بدء التنزيل من: {source}\n")

    for video_name in VIDEOS:
        # ادوّر عن الفيديو
        video_file = None
        for ext in ['.mp4', '.avi', '.mov']:
            candidate = source / f"{video_name}{ext}"
            if candidate.exists():
                video_file = candidate
                break

        if video_file:
            # نسخ الفيديو
            dest = OUTPUT_DIR / video_file.name
            try:
                size_mb = video_file.stat().st_size / 1024 / 1024
                print(f"📋 {video_name}:")
                print(f"   الحجم: {size_mb:.1f} MB")
                print(f"   بنقل من: {video_file}")

                shutil.copy2(video_file, dest)
                print(f"   ✅ نزل في: {dest}\n")
            except Exception as e:
                print(f"   ❌ خطأ: {e}\n")
        else:
            print(f"❌ {video_name} - مالقتش الملف\n")

    # اعرض النتايج
    downloaded = list(OUTPUT_DIR.glob('*.mp4'))

    print("="*70)
    print(f"📊 النتيجة: {len(downloaded)} فيديو")
    print("="*70)
    for v in sorted(downloaded):
        size_mb = v.stat().st_size / 1024 / 1024
        print(f"✅ {v.name} ({size_mb:.1f} MB)")

    print(f"\n📁 المجلد: {OUTPUT_DIR.absolute()}")
    print("\nالفيديوهات الأصلية جاهزة للمشاهدة! 🎬")


if __name__ == '__main__':
    download_videos()
