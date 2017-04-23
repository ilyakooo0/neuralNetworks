import csv
import png
import numpy as np


with open("data/training.csv") as training:
    file = csv.reader(training)
    next(file)
    for i, line in enumerate(file):
        nums = list(map(int, line[-1].split()))
        png.from_array(np.reshape(nums, (96, 96)), "L", {
            "height": 96,
            "width": 96,
            "bitdepth": 8
        }).save("images/" + str(i) + ".png")