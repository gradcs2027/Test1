"""يحقن ملفات الخلايا في الـ notebook ويمسح الـ outputs القديمة.

كل خلية كودها في ملف .py لوحده عشان نعدّله بـ diff مفهوم بدل ما نعدّل
JSON. ده المصدر الوحيد للحقيقة — الـ .ipynb ناتج، ماتعدّلوش بالإيد.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
NB = HERE / "notebook520b8d324a.ipynb"

REPLACEMENTS = {
    2: HERE / "cell2_other_class.py",     # كل حركات الشخص الواحد (49 كلاس)
    3: HERE / "cell3_normalization.py",   # الـ normalization المشترك
    4: HERE / "cell4_split.py",           # مش بيعيد تعريف action_names
    5: HERE / "cell5_model.py",           # شيل Masking اللي بيكسر cuDNN LSTM
    6: HERE / "cell6_eval.py",            # تقييم يقرا مع 49 كلاس + top-k
    7: HERE / "cell7_video.py",           # VIDEO_PATH في مكان واحد + الفيديو الجديد
    8: HERE / "cell8_extract.py",         # يستخدم VIDEO_PATH + نسبة الاكتشاف
    9: HERE / "cell9_smoothing.py",       # الـ windows + smoothing
    10: HERE / "cell10_timeline.py",      # ground truth بتاع الفيديو المختبَر
}

# cell_diag.py خلية تشخيص لمرة واحدة كانت بتتحقن هنا (بـ tag domain-diag).
# خلصت شغلها (الحكم: التأطير مختلف، مش فرق بنيوي) واتشالت من الـ notebook.
# الملف لسه على الديسك لو احتجناه تاني. السطر ده بيمسح أي نسخة قديمة منها
# لو الـ notebook اتسحب من Kaggle وهي جواه.
DIAG_TAG = "domain-diag"

nb = json.loads(NB.read_text(encoding="utf-8"))
nb["cells"] = [c for c in nb["cells"]
               if c.get("metadata", {}).get("tag") != DIAG_TAG]

for idx, src_file in REPLACEMENTS.items():
    cell = nb["cells"][idx]
    assert cell["cell_type"] == "code", f"cell {idx} is not code"
    cell["source"] = src_file.read_text(encoding="utf-8").splitlines(keepends=True)

# ملاحظة: nbformat بيسمح لـ source يبقى list of lines أو string واحد،
# والـ notebook ده فيه الاتنين — فلازم نوحّدهم قبل أي معالجة نصية
def as_text(cell):
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


# خلية 11 (فيديو الإخراج) تعديلين سطر واحد، مش محتاجة ملف لوحدها
OLD_PATH_LINE = 'video_path = "/kaggle/input/datasets/abdallahhsamir/testvid/vid.mp4"'
NEW_PATH_LINE = "video_path = VIDEO_PATH"
# frame_indices بقى np.ndarray بعد ما خلية 8 اتصلّحت، و ndarray مالوش
# .index() — ده كسر رن 19 في آخر خلية. searchsorted بيشتغل مع الاتنين.
OLD_IDX_LINE = "pos = frame_indices.index(frame_idx)"
NEW_IDX_LINE = "pos = int(np.searchsorted(frame_indices, frame_idx))"
cell11 = nb["cells"][11]
text11 = as_text(cell11)
# السكربت بيتشغّل كذا مرة على نفس الـ notebook، فلازم يبقى idempotent
assert OLD_PATH_LINE in text11 or NEW_PATH_LINE in text11, "cell 11 path line not found"
assert OLD_IDX_LINE in text11 or NEW_IDX_LINE in text11, "cell 11 index line not found"
cell11["source"] = (text11.replace(OLD_PATH_LINE, NEW_PATH_LINE)
                          .replace(OLD_IDX_LINE, NEW_IDX_LINE).splitlines(keepends=True))

# ما يفضلش أي مسار مكتوب بالإيد في أي خلية
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        assert "testvid/vid.mp4" not in as_text(cell), f"old path left in cell {i}"

# الـ outputs القديمة مالهاش لازمة وبتكبّر الملف 600KB
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
# الطرفية على ويندوز cp1252 فمش بتاخد emoji/عربي — نخليها ASCII
print(f"OK: patched cells {sorted(REPLACEMENTS)}, "
      f"{len(nb['cells'])} cells total in {NB.name}")
