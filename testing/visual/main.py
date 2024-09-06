import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity

INPUT_FILES = ["blender", "godot_4", "unreal_engine_5", "surfels"]
NAMES = ["Blender", "Godot 4", "Unreal Engine 5", "Surfele"]

images = []
for input_file in INPUT_FILES:
    img = Image.open(f"testing/visual/imgs/{input_file}.png").convert("HSV")
    img = np.array(img, dtype="float")
    img = img[:, :, 2]  # Just value
    img /= 255.0
    images.append(img)
images = np.array(images)

ref_name = NAMES[0]
ref_img = images[0]
others = list(zip(NAMES[1:], images[1:]))

for name, image in others:
    mse = mean_squared_error(ref_img, image)
    ssim = structural_similarity(ref_img, image, data_range=1.0)
    print(f"Name: {name:<20}   MSE: {mse:1.5f}   SSIM: {ssim:1.5f}")
