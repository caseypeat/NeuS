import numpy as np
import json
import os


if __name__ == "__main__":
    dirpath = "../../NeRF-Reconstruction/outputs/processed_data/2"
    dirpath = os.path.join("generate_data", dirpath)

    cameras_path = os.path.join(dirpath, "cameras.npz")

    cameras = np.load(cameras_path)

    print(cameras)