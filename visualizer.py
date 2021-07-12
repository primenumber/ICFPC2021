import numpy as np
import cv2 as cv
import json
import sys

from graphviz import Digraph

MAX_PROB_ID = 132

def get_prob_path(i):
    return f"problems/{i}.in"


def get_edges(i):
    problem = json.load(open(get_prob_path(i)))
    result = []
    for bonus in problem["bonuses"]:
        if bonus["bonus"] == "GLOBALIST":
            result.append(bonus["problem"])

    return result


def bonus_graph():
    engines = ['twopi', 'osage', 'sfdp', 'fdp', 'patchwork', 'neato', 'dot', 'circo']
    dot = Digraph(engine=engines[7])
    for i in range(1, MAX_PROB_ID + 1):
        for to in get_edges(i):
            print(to)
            dot.edge(str(i), str(to))

    dot.render('bonus.gv')


def visualize(prob_path, ans_path, img_path):
    problem = json.load(open(prob_path))
    answer = json.load(open(ans_path))

    hole = np.array(problem["hole"], np.int32) * 4
    hole.reshape((-1, 1, 2))

    size = np.amax(hole)

    img = np.zeros((size, size, 3), np.uint8)

    cv.polylines(img, [hole], True, (0, 255, 255))

    for (x, y) in problem["hole"]:
        cv.circle(img, (x * 4, y * 4), 2, (255, 255, 255), -1)

    pose = np.array(answer["vertices"], np.int32) * 4
    for (u, v) in problem["figure"]["edges"]:
        cv.line(img, pose[u], pose[v], (255, 255, 0), 1)

    for (x, y) in answer["vertices"]:
        cv.circle(img, (x * 4, y * 4), 2, (255, 0, 255), -1)

    cv.imwrite(img_path, img)

#for i in range(1, 133):
#    visualize(f"problems/{i}.in", f"best/{i}.out", f"vis/{i}.png")

#bonus_graph()
i = int(sys.argv[1])
visualize(f"problems/{i}.in", sys.argv[2], f"vis/{i}.png")
