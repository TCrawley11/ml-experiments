import yaml

def load_yaml(path: str):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_yaml("model/config/model_config.yaml")
print(config['model'])