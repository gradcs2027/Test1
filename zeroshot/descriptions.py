"""
أسماء ووصف الحركات لتجربة الـ zero shot.

اتكتبوا قبل أي نتيجة وبيتثبّتوا: لو عدّلنا الوصف بعد ما نشوف النتايج
عشان الرقم يحلو، يبقى ده ضبط على الاختبار مش zero shot.
"""

# NTU60: الحركات الفردية A001..A049 (index 0..48) — نفس الـ 49 حركة اللي
# في lstm/lstm_ntu60.ipynb. الحركات اللي فيها شخصين (A050..A060) برّه لأن
# بنقرا هيكل الشخص الأول بس.
NTU49 = [
    ('drink water',        'a person lifts a cup to the mouth and drinks water'),
    ('eat meal',           'a person brings food to the mouth with a hand and eats'),
    ('brushing teeth',     'a person moves a toothbrush back and forth in front of the mouth'),
    ('brushing hair',      'a person moves a hand with a brush over the head to comb the hair'),
    ('drop',               'a person lets an object fall from the hand to the floor'),
    ('pick up',            'a person bends down and picks an object up from the floor'),
    ('throw',              'a person swings an arm and throws an object forward'),
    ('sitting down',       'a person lowers the body from standing and sits down on a chair'),
    ('standing up',        'a person rises from a chair to a standing position'),
    ('clapping',           'a person claps both hands together repeatedly in front of the chest'),
    ('reading',            'a person holds a book in front of the face and reads'),
    ('writing',            'a person writes with a pen using one hand on a surface'),
    ('tear up paper',      'a person pulls both hands apart to tear up a piece of paper'),
    ('wear jacket',        'a person puts the arms through the sleeves and puts on a jacket'),
    ('take off jacket',    'a person pulls a jacket off the shoulders and arms'),
    ('wear a shoe',        'a person lifts one foot and puts on a shoe'),
    ('take off a shoe',    'a person lifts one foot and takes off a shoe'),
    ('wear on glasses',    'a person raises both hands to the face and puts on glasses'),
    ('take off glasses',   'a person raises both hands to the face and takes off glasses'),
    ('put on a hat',       'a person raises both hands to the head and puts on a hat'),
    ('take off a hat',     'a person raises a hand to the head and takes off a hat'),
    ('cheer up',           'a person raises both arms up in the air to cheer'),
    ('hand waving',        'a person raises one hand and waves it from side to side'),
    ('kicking something',  'a person swings one leg forward to kick an object'),
    ('reach into pocket',  'a person puts a hand into the pocket of the pants'),
    ('hopping',            'a person hops on one foot'),
    ('jump up',            'a person jumps up in the air with both feet'),
    ('phone call',         'a person raises a hand holding a phone to the ear and talks'),
    ('playing with phone', 'a person holds a phone with both hands in front of the body and taps the screen'),
    ('typing on keyboard', 'a person moves both hands over a keyboard and types'),
    ('pointing',           'a person extends one arm and points with a finger'),
    ('taking a selfie',    'a person holds a phone at arm length in front of the face to take a selfie'),
    ('check time',         'a person raises the wrist in front of the chest and looks at a watch'),
    ('rub two hands',      'a person rubs the two hands together in front of the body'),
    ('nod head',           'a person nods the head or bows forward'),
    ('shake head',         'a person shakes the head from side to side'),
    ('wipe face',          'a person moves a hand across the face to wipe it'),
    ('salute',             'a person raises a hand to the forehead to salute'),
    ('palms together',     'a person brings the two palms together in front of the chest'),
    ('cross hands',        'a person crosses the two hands in front of the body'),
    ('sneeze cough',       'a person covers the mouth with a hand and sneezes or coughs'),
    ('staggering',         'a person staggers and loses balance while walking'),
    ('falling down',       'a person loses balance and falls down to the floor'),
    ('headache',           'a person holds a hand to the head as if having a headache'),
    ('chest pain',         'a person presses a hand on the chest as if in pain'),
    ('back pain',          'a person puts a hand on the lower back as if in pain'),
    ('neck pain',          'a person holds a hand on the neck as if in pain'),
    ('nausea vomiting',    'a person bends forward and covers the mouth as if feeling sick'),
    ('fan self',           'a person waves a hand in front of the face to fan themselves'),
]
assert len(NTU49) == 49

# الحركات اللي بنختبر عليها (الـ 18 حركة في فيديوهات الاختبار).
# اللي ليها مقابل في NTU بتاخد نفس نصّ NTU بالحرف.
OURS = {
    'brush_hair':    NTU49[3][1],
    'clapping':      NTU49[9][1],
    'drink_water':   NTU49[0][1],
    'hand_shake':    'a person extends one hand forward and shakes hands with someone',
    'hugging':       'a person opens both arms and wraps them around someone to hug',
    'lying':         'a person lies down flat on the floor or on a bed',
    'phone_call':    NTU49[27][1],
    'play_pingpong': 'a person swings a paddle with one arm to hit a ball over a table tennis table',
    'running':       'a person runs fast with quick alternating leg steps and swinging arms',
    'sitting':       NTU49[7][1],
    'spray_perfume': 'a person raises one arm holding a bottle and sprays perfume toward the neck',
    'stand_up':      NTU49[8][1],
    'touch_head':    'a person raises a hand and touches the top of the head',
    'turn_on_light': 'a person raises an arm and presses a light switch on the wall',
    'wake_up':       'a person wakes up in bed, sits up and stretches the arms',
    'walking':       'a person walks forward with alternating leg steps and swinging arms',
    'wave':          NTU49[22][1],
    'wear_glasses':  NTU49[17][1],
}

# الحركات اللي بنشيلها من تدريب NTU كلها (فيديوهات ونص): اسمنا → index في NTU49
HIDDEN = {'wave': 22, 'clapping': 9, 'sitting': 7}

# حركات NTU زيادة بتتشال من التدريب برضه، بس عشان نظبط عليها معامل التصحيح
# (gamma) — مش عشان نختبر عليها. اتختاروا عشوائي (seed 0) من الحركات اللي
# مالهاش أي علاقة بالـ 18 بتوعنا، فالحركات المخبّية فوق تفضل مخبّية لحد الاختبار.
CALIB = {'playing with phone': 28, 'rub two hands': 33, 'staggering': 41}

# «متشافة» = نصّها بالظبط موجود في نصوص التدريب. الباقي «مش متشافة».
_out = set(HIDDEN.values()) | set(CALIB.values())
_train_txt = {d for i, (_, d) in enumerate(NTU49) if i not in _out}
SEEN = sorted(k for k, v in OURS.items() if v in _train_txt)
UNSEEN = sorted(k for k in OURS if k not in SEEN)
