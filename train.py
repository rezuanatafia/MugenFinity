"""Run the notebook's code cells to create the model bundle in data/models/."""
import builtins, json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
cells = json.loads((ROOT / "dengue_preprocessing.ipynb").read_text(encoding="utf-8"))["cells"]
builtins.display = print          # notebooks provide display(); scripts do not
os.chdir(ROOT / "data")           # the notebook looks for the CSV in the current folder
scope = {}
for cell in cells:
    if cell["cell_type"] == "code":
        exec("".join(cell["source"]), scope)
