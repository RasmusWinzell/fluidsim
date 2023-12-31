import glob
import inspect
from importlib.machinery import SourceFileLoader
from os.path import basename, dirname, isfile, join

sim_format = "sim{}"
result_format = "result{}"


def save_sim(sim, result=None):
    source = inspect.getsource(sim)
    hash = calc_hash(source)
    if hash in hashes:
        print("Simulation already saved as", hashes[hash].__name__)
        return False

    file_name = sim_format.format(len(sims))
    file_path = join(dirname(__file__), file_name + ".py")
    with open(file_path, "w+") as f:
        f.write(source)
        print("Saved as", file_name)
    res_path = join(dirname(__file__), result_format.format(len(sims)) + ".txt")
    if result is not None:
        with open(res_path, "w+") as f:
            f.write(result)


def get_sims():
    file_names = glob.glob(join(dirname(__file__), "*.py"))
    __all__ = [
        f
        for f in file_names
        if isfile(f) and not f.endswith("__init__.py") and not f.endswith(__file__)
    ]
    modules = [
        SourceFileLoader(basename(module)[:-3], module).load_module()
        for module in __all__
    ]
    sim_modules = [module for module in modules if hasattr(module, "Simulation")]
    return sim_modules


def load_results(sims):
    file_names = glob.glob(join(dirname(__file__), "*.txt"))
    results = []
    for file_name in file_names:
        with open(file_name, "r") as f:
            results.append(f.read())
    return results


def calc_hashes(sims):
    return {calc_hash(inspect.getsource(sim)): sim for sim in sims}


def calc_hash(source):
    return sum([ord(char) * 2141 for char in source])


sims = get_sims()
results = load_results(sims)
hashes = calc_hashes(sims)
