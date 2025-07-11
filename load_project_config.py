import yaml

def load_project_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    return config