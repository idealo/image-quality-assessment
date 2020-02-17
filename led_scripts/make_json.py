import json
import os

def get_params_from_image_id(image_id):
    vars_to_extract = {
        "color": str,
        "led-brightness": float,
        "exposure": int,
        "color-temp": int,
    }
    data = {}
    params = image_id.split("_")
    data["location"] = params[0]
    data["orientation"] = params[-1]
    for p in params[1:-1]:
        for k, converter in vars_to_extract.items():
            if p.startswith(k):
                val = p.replace(k, "")
                if val[0] == "-":
                    continue
                try:
                    data[k] = converter(val)
                except ValueError:
                    data[k] = val
    return data

def get_bad_params_from_manually_verified_file(path):
    all_data = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = [el.strip() for el in line.split(",")]
            if line[1] == "bad":
                image_id = os.path.splitext(os.path.split(line[0])[-1])[0]
                record = get_params_from_image_id(image_id)
                record["image_id"] = image_id
                all_data.append(record)
    return all_data

path = "/home/pi/image-quality-assessment/manually_verified_images.txt"
params = get_bad_params_from_manually_verified_file(path)
print(params)
with open("manually_verified.json", "w") as f:
    json.dump(params, f)
