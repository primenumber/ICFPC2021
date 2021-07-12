import glob
import json
import math
import re

def max_score(prob):
    v = len(prob["figure"]["vertices"])
    e = len(prob["figure"]["edges"])
    h = len(prob["hole"])
    return math.ceil(1000.0 * math.log2(v * e * h / 6.0))

result = []
total = 0
for path in glob.glob('problems/*.in'):
    prob_id = int(re.search(r'\d+', path).group())
    problem = json.load(open(path))
    result.append((prob_id, max_score(problem)))
    total += max_score(problem)

print("Total:", total)
for pid, score in sorted(result):
    print(pid, score)
