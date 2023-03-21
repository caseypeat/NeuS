import numpy as np
import os
import json


if __name__ == "__main__":
    filepath = "../../data/plant_and_food/tonemapped/2022-08-31/jpeg_north/refined_5cm.json"
    filepath = os.path.join("generate_data", filepath)

    with open(filepath, 'r') as file:
        data = json.load(file)

    print(data)