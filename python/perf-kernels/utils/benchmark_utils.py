import os
import json

# Base directory where configs are located
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


def get_model_configs(config_path='model_configs.json', model_families=["llama3"], model="all"):
    """
    Load model names from the configuration file.

    Args:
        config_path (str): User-provided path to the configuration JSON file.
        model_families (list): List of model family names to retrieve.

    Returns:
        dict: A dictionary of available models and their configurations for the specified families.
    """
    # Resolve config path relative to ./perf-kernels/
    config_path = os.path.join(BASE_DIR, config_path)

    with open(config_path, 'r') as f:
        configs = json.load(f)

    # Extract models and their configurations for the specified families
    filtered_configs = {}

    for family in model_families:
        if family in configs:
            # Check if model filtering is required
            if model == "all":
                # Include all models in the family
                for model_size, model_configs in configs[family].items():
                    filtered_configs[f"{family}-{model_size}"] = model_configs
            else:
                # Parse the model string (e.g., llama3_8B or llama3-8B)
                delimiter = "_" if "_" in model else "-"
                model_parts = model.split(delimiter)

                # Check if the family and size match
                if len(model_parts) == 2 and model_parts[0] == family:
                    model_size = model_parts[1]
                    if model_size in configs[family]:
                        filtered_configs[f"{family}-{model_size}"] = configs[family][model_size]

    if not filtered_configs:
        print(f"Warning: No models selected for families: {model_families} with filter: '{model}'")

    return filtered_configs


def get_available_models(config_file='model_configs.json', model_families=["llama3"]):
    """
    Load model names from the configuration file.

    Args:
        config_file (str): Path to the configuration JSON file.
        model_families (list): List of model family names to retrieve.

    Returns:
        list: A list of available models for the specified families.
    """
    # Resolve config path relative to ./perf-kernels/
    config_path = os.path.join(BASE_DIR, config_file)

    with open(config_path, 'r') as f:
        configs = json.load(f)

    models = [f"{family}-{model}" for family in model_families if family in configs for model in configs[family]]

    return models
