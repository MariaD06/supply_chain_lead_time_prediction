from copy import deepcopy
from pathlib import Path

import yaml

BASE_CONFIG_PATH = Path("config/modeling.yaml")
OUT_DIR = Path("config/ablations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
    base_config = yaml.safe_load(f)

ablations = {
    "all_features": {
        "remove_numeric": [],
        "remove_categorical": [],
    },
    "without_distance": {
        "remove_numeric": ["distance_km"],
        "remove_categorical": [],
    },
    "without_transport_mode": {
        "remove_numeric": [],
        "remove_categorical": ["transport_mode"],
    },
    "without_weather": {
        "remove_numeric": [],
        "remove_categorical": ["weather_condition"],
    },
    "without_route_ports": {
        "remove_numeric": [],
        "remove_categorical": ["origin_port", "destination_port"],
    },
}

for ablation_name, remove in ablations.items():
    config = deepcopy(base_config)

    config["project_name"] = f"ablation_{ablation_name}"

    config["features"]["numeric"] = [
        col for col in config["features"]["numeric"]
        if col not in remove["remove_numeric"]
    ]

    config["features"]["categorical"] = [
        col for col in config["features"]["categorical"]
        if col not in remove["remove_categorical"]
    ]

    out_path = OUT_DIR / f"{ablation_name}.yaml"

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Saved {out_path}")