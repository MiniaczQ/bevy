import numpy as np
import csv
import matplotlib.pyplot as plt

INPUT_FILES = [
    "godot_static",
    "godot_moving",
    "surfels_static",
    "surfels_moving",
    "ue5_static",
    "ue5_moving",
]

NAMES = [
    "Godot kamera statyczna",
    "Godot kamera ruchoma",
    "Surfele kamera statyczna",
    "Surfele kamera ruchoma",
    "Unreal Engine 5 kamera statyczna",
    "Unreal Engine 5 kamera ruchoma",
]

datas = []

for name, input_file in zip(NAMES, INPUT_FILES):
    with open(f"testing/performance/frame/data/{input_file}.csv", "r") as file:
        data = np.array([float(x[0]) for x in csv.reader(file)])
        data = data[:60]
        datas.append(data)
        mean = np.mean(data)
        std = np.std(data)
        print(f"Name: {name:<32}  Mean: {mean:1.3f}  Std: {std:1.3f}")

GRAPH_FILES = ["godot", "surfels", "ue5"]

for file_name, data_static, data_moving in zip(
    GRAPH_FILES,
    [datas[0], datas[2], datas[4]],
    [datas[1], datas[3], datas[5]],
):
    frames = data_static.shape[0]
    X = range(1, frames + 1)
    plt.plot(X, data_static)
    plt.plot(X, data_moving)
    # plt.legend(["Kamera statyczna", "Kamera ruchoma"])
    plt.ylim((0.0, 10.0))
    plt.xlabel("Klatka")
    # plt.title(name)
    plt.ylabel("Czas na obliczenie klatki [ms]")
    plt.tight_layout()
    plt.savefig(
        f"testing/performance/frame/imgs/{file_name}.png", dpi=600, transparent=True
    )
    plt.clf()
