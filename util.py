import os
import json
import yaml
import json
import hashlib

def hash_params(params_dict):
    s = json.dumps(params_dict, sort_keys=True)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]
    
def create_dirs(dirpath):
    """Creating directories."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def load_yaml(filepath):
    with open(filepath, 'r') as stream:
        return yaml.safe_load(stream)


def load_json(filepath):
    """Load a json file."""
    with open(filepath, "r", encoding='utf8') as fp:
        obj = json.load(fp)
    return obj


def save_json(obj, filepath):
    """Save a dictionary to a json file."""
    with open(filepath, "w") as fp:
        json.dump(obj, fp, indent=4)
