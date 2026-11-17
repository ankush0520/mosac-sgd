import copy, yaml


def load_config(path="configs/default.yaml", **overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        set_path(cfg, k, v)
    return cfg


def set_path(cfg, dotted, value):
    d = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
    return cfg


def with_overrides(cfg, **kv):
    c = copy.deepcopy(cfg)
    for k, v in kv.items():
        set_path(c, k.replace("__", "."), v)
    return c
