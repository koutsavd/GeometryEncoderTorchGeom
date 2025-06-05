import json
import os
from pathlib import Path
import util
from joblib import Parallel, delayed


def get_dimensions_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    min_coords = data["min"]
    max_coords = data["max"]
    model_id = data.get("id", json_path.parent.name)

    dimensions = {
        "width": max_coords[0] - min_coords[0],  # x
        "height": max_coords[1] - min_coords[1],  # y
        "length": max_coords[2] - min_coords[2],  # z
    }
    return model_id, dimensions


def is_probably_a_car(dim):
    length, height, width = dim["length"], dim["height"], dim["width"]

    if length > 5.5 or height > 2.0 or width > 2.5:
        return False  # too big
    if length < 2.5:
        return False  # too small
    if height / length > 0.5:
        return False  # too tall
    return True

def filter_car_like_models(root_dir):
    root = Path(root_dir)
    accepted = []

    for json_file in root.rglob("model_normalized.json"):
        model_id, dims = get_dimensions_from_json(json_file)
        if is_probably_a_car(dims):
            accepted.append((model_id, dims))

    return accepted

if __name__ == "__main__":
    root_path = Path("/Volumes/M.2/02958343")
    car_models = filter_car_like_models(root_path)

    #files = (root_path / model_id / "models/model_normalized.obj" for model_id, _ in car_models)
    print(f"Filtered {len(car_models)} car-like models.")
    # for file in files:
    #     util.read_preproc_save(file,10)
    # tasks = [delayed(util.read_preproc_save)(root_path / model_id / "models/model_normalized.obj",model_id, 10) \
    #                  for model_id, _ in car_models]
    # Parallel(n_jobs=8)(tasks)
    for model_id, _ in car_models:
        file = root_path / model_id / "models/model_normalized.obj"
        util.read_preproc_save(file,  model_id ,10)
