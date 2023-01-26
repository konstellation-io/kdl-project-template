
## Importing library functions

Reusable functions can be imported from the library (`lib` directory) to avoid code duplication and to permit a more organized structuring of the repository.

**In Jupyter:**
To import library code in notebooks, you may need to add the `lab` directory to PYTHONPATH, for example as follows:

```python
import sys
from pathlib import Path

DIR_REPO = Path.cwd().parent.parent
DIR_LAB = DIR_REPO / "lab"

sys.path.append(str(DIR_LAB))

from lib.viz import plot_confusion_matrix
```

**In Vscode**:
Imports from `lab` directory subdirectories are recognized correctly by code linters
thanks to the defined `PYTHONPATH=lab` in the .env environment file.
However, they are not recognized by the terminal,
so in order to run code with imports from Vscode terminal,
prepend your calls to Python scripts with `PYTHONPATH=lab` as follows:
`PYTHONPATH=lab python {filename.py}`.

**On Github Actions:**
To be able to run imports from the `lib` directory on Github Actions, you may add it to PYTHONPATH in [github_actions_pipeline.yml](.github/workflows/github_actions_pipeline.yml) as indicated:

```yaml
env:
  PYTHONPATH:  ${{ github.workspace }}
```

`github.workspace` is the root on the github runner that the repository is cloned to,  which includes `lib`.
This then allows importing library functions directly from the Python script that is being executed on the runner, for instance:

```python
from lib.viz import plot_confusion_matrix
```