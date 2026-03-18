import json
from pathlib import Path

r = Path('/Users/wscholl/squish/dev/results')
BF16 = {'arc_easy': 0.750, 'hellaswag': 0.612, 'piqa': 0.772, 'winogrande': 0.630}
cfgs = [
    ('BF16 reference',        'accuracy_bf16_reference_squish_path_500.json'),
    ('Lossless (no INT4)',     'accuracy_lossless_500.json'),
    ('No AWQ (alpha=0)',       'accuracy_fp16attn_noawq_500.json'),
    ('v1 alpha=0.10 g=16 n=20','accuracy_mixed_precision_500.json'),
    ('v3 alpha=0.15 g=16 n=20','accuracy_mixed_v3_500.json'),
    ('v2 alpha=0.05 g=32 n=64','accuracy_mixed_v2_500.json'),
]
print('{:<32} {:>7} {:>7} {:>7} {:>7} {:>5}'.format('Config','arc','hella','piqa','wino','beats'))
print('-' * 72)
for name, fn in cfgs:
    d = json.loads((r / fn).read_text())['results']
    a = d['arc_easy']['acc']
    h = d['hellaswag']['acc']
    p = d['piqa']['acc']
    w = d['winogrande']['acc']
    marks = ['Y' if v >= BF16[k] else ' '
             for k, v in zip(['arc_easy','hellaswag','piqa','winogrande'], [a,h,p,w])]
    n = marks.count('Y')
    print('{:<32} {:.4f}{} {:.4f}{} {:.4f}{} {:.4f}{} {}/4'.format(
        name, a, marks[0], h, marks[1], p, marks[2], w, marks[3], n))
