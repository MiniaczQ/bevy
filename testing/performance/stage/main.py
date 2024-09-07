import matplotlib.pyplot as plt
import numpy as np
import csv

INPUT_FILES = ["static", "moving"]

NAMES = ["Kamera statyczna", "Kamera ruchoma"]

STAGES = [
    "Wypełnienie struktury akceleracyjnej jądrem 1x1",
    "Usunięcie surfeli na bazie zagęszczenia",
    "Usunięcie surfeli na bazie wykorzystania",
    "Wstawienie surfeli na bazie zagęszczenia",
    "Wypełnienie struktury akceleracyjnej jądrem 5x5",
    "Wstępne próbkowanie surfeli",
    "Współdzielenie próbek pomiędzy surfelami",
    "Zaktualizowanie jasności surfela",
    "Zapisanie oświetlenie pikseli w teksturze",
]

for name, input_file in zip(NAMES, INPUT_FILES):
    data = None
    with open(f"testing/performance/stage/data/{input_file}.csv", "r") as file:
        data = np.array([float(x[0]) for x in csv.reader(file)])
    Y = np.arange(8, -1, -1)
    plt.barh(Y, data)
    for y, x in zip(Y, data):
        plt.text(x * 1.2, y, f"{x:1.4f}", verticalalignment="center")
    plt.yticks(Y, STAGES)
    plt.xlim((0.001, 30))
    plt.xscale("log")
    plt.xlabel("Czas na obliczenie kroku [ms]")
    #plt.title(name)
    plt.tight_layout()
    plt.savefig(
        f"testing/performance/stage/imgs/{input_file}.png", dpi=600, transparent=True
    )
    plt.clf()
