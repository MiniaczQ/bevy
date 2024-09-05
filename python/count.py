import csv
import numpy as np
import matplotlib.pyplot as plt

TEST_FILES = [
    "count_static",
    "count_jump_sweep",
    "count_step_sweep",
    "count_jump_zoom_in",
    "count_step_zoom_in",
    "count_jump_zoom_out",
    "count_step_zoom_out",
]

NAMES = [
    "Nieruchoma kamera",
    "Przesunięcie skokowe",
    "Przesunięcie stopniowe",
    "Zbliżenie skokowe",
    "Zbliżenie stopniowe",
    "Oddalenie skokowe",
    "Oddalenie stopniowe",
]

for test_file, name in zip(TEST_FILES, NAMES):
    path = f"python/data/{test_file}.csv"
    with open(path, "r") as file:
        counts = np.array([[int(v) for v in vs] for vs in csv.reader(file)])
        total = np.sum(counts)
        mean = np.mean(counts)
        dev = np.std(counts)
        print(
            f"Test: {test_file:<32} | Visible: {total:>4} | Mean: {mean: 1.3f} | Dev: {dev: 1.3f}"
        )

        im = plt.imshow(counts, vmin=0, vmax=9)
        for i in range(16):
            for j in range(16):
                plt.text(j, i, counts[i, j], ha="center", va="center", color="w")

        #cbar = plt.colorbar(im)
        plt.title(name)
        plt.tick_params(
            top=False,
            bottom=False,
            labeltop=False,
            labelbottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
        )

        plt.tight_layout()
        plt.savefig(f"python/imgs/{test_file}_heatmap.png", dpi=600, transparent=True)
        plt.clf()
