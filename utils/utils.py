import json
import os 

def save_dict_to_json(d: dict, json_path: str):
    """Saves dict of floats in json file

    Args:
        d: dict
        json_path: (string) path to json file
    """
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: v for k, v in d.items()}
            json.dump(d, f, indent=4)
    else:
        with open(json_path, 'r') as f:
            jdict = json.load(f)
        for key, val in d.items():
            jdict[key] = val
        with open(json_path, 'w') as f:
            json.dump(jdict, f, indent=4)

def save_argparser(parser, save_dir) -> dict:

    jsummary = {}
    for key, val in vars(parser).items():
        jsummary[key] = val

    save_dict_to_json(jsummary, os.path.join(save_dir, 'param-summary.json'))

    return jsummary