import os

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

TEMPPATH = os.path.join(WORKSPACE, "temp")
if not os.path.exists(TEMPPATH):
    os.makedirs(TEMPPATH)

def create_temp_folder(name):
    path = os.path.join(TEMPPATH, name)
    # delete
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    os.makedirs(path)
    return path

def save_pkl(obj, target_file, name):
    import pickle
    path = os.path.join(os.path.splitext(target_file)[0], name)
    if path[-4:] != ".pkl":
        path += ".pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(target_file, name, default=None):
    import pickle
    path = os.path.join(os.path.splitext(target_file)[0], name)
    if path[-4:] != ".pkl":
        path += ".pkl"
    if not os.path.exists(path):
        return default
    with open(path, 'rb') as f:
        return pickle.load(f)