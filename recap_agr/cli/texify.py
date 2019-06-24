#! /usr/bin/env python3

import json
import sys
import os
import subprocess

FILTER = [
    "Duration",
    "Precision",
    "Recall",
    "Average Precision",
    "NDCG",
    "Correctness",
    "Completeness",
]


def print_clipboard(output):
    process = subprocess.Popen(
        "pbcopy", env={"LANG": "en_US.UTF-8"}, stdin=subprocess.PIPE
    )
    process.communicate(output.encode("utf-8"))


def tex_tab(results_list):
    return " & " + " & ".join(results_list) + r" \\"


def texify_file(filename, print_tex):
    with open(filename, "r") as file:
        content = json.load(file)

        results = {**content["Results"]["unranked"], **content["Results"]["ranked"]}
        results["Duration"] = content["Duration"]

        results_tex = {key: f"\\({results[key]}\\)" for key in FILTER}

        if print_tex:
            print(tex_tab(results_tex.keys()))
            print(tex_tab(results_tex.values()))

        print_clipboard(tex_tab(results_tex.values()))


def texify(print_tex=False):
    files = sorted(os.listdir("data/results"))
    filename = os.path.join("data/results", files[-1])
    texify_file(filename, print_tex)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        texify_file(sys.argv[1], True)
    else:
        texify(True)
