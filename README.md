# KDL Project Template

## Project structure

The project repository has the following directory structure:

```
├── lab
│   │
│   ├── analysis  <- Analyses of data, models etc. (typically notebooks)
│   │
│   ├── docs      <- High-level reports, executive summaries at each milestone (typically .md)
│   │
│   ├── lib       <- Importable functions shared between analysis notebooks and processes scripts
│   │                (including unit tests)
│   │
│   └── processes           <- Source code for reproducible workflow steps. For example:
│       ├── process_data   
│       │   ├── main.py      
│       │   ├── process_data.py  
|       │   └── test_process_data.py
|       ├── train_model
│       │   ├── main.py      
│       │   ├── train_model.py  
|       │   └── test_train_model.py
│       └── ...
│   
├── goals         <- Acceptance criteria (TBD)
│   
├── runtimes      <- Code for generating deployment runtimes (.krt)
│   
├── .drone.yml
├── .flake8     
├── .gitignore
|
└── README.md
```


## Example project pipelines

KDL contains various components that need to be correctly orchestrated and connected. 
To illustrate their intended usage, we provide two example machine learning pipelines already implemented in KDL. 
The first example pipeline is a simple classification problem with standard ML models from scikit-learn.
The second example pipeline is a slightly more complex image classification problem using PyTorch.

### Scikit-learn example: Wine classification

The first example pipeline is a simple classification task. 
Based on the [Wine Recognition Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset), the aim is to classify three different types of wines based on their physicochemical characteristics (alcohol content, malic acid, etc.).

The code for wine classification is in [lab/processes/sklearn_example/main.py](lab/processes/sklearn_example/main.py).

The execution of wine classification pipeline on Drone agents is specified in [.drone.yml](.drone.yml) (for simplicity, we are omitting various additional components, such as the environment variables and the AWS secrets):

```yaml
kind: pipeline
type: kubernetes
name: application-examples

trigger:
  ref:
  - refs/tags/run-examples-*

steps:
  - name: sklearn-example
    image: terminus7/sci-toolkit-runner:1.1.2
    commands:
      - python3 lab/processes/sklearn_example/main.py
```

To trigger pipeline execution on Drone runners, push a tag containing the name matching the trigger (e.g. `run-examples-v1`) to the remote repository.
For more information, see the section Launching experiment runs (Drone) below.

The results of executions are stored in MLflow: 
in this simple example we are only tracking one parameter (name of the classifier), and one metric (the obtained validation accuracy). 
The connection to MLflow to log these parameters and metrics is established via the code in the [main.py](lab/processes/sklearn_example/main.py) and with the environment variables in [.drone.yml](.drone.yml). For more information, see the section Logging experiment results (MLflow) below.
To see the tracked experiments, visit the MLflow tool UI.


### PyTorch example: digit classification

The second example pipeline is based on an image classification problem, with the aim of classifying digits from the [Optical Recognition of Handwritten Digits](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset) dataset. This is _not_ the standard MNIST dataset (over 20,000 images of 28x28 pixels), it is a considerably smaller dataset (1797 images of 8x8 pixels), with the advantage that it does not require downloading from the internet as it is already distributed with the scikit-learn library in the `sklearn.datasets` package.

In this example pipeline, defined in [lab/processes/pytorch_example/main.py](lab/processes/pytorch_example/main.py), the dataset images are split into train, validation and test sets, and are classified as digits 0-9 with a simple convolutional neural network. The training history (accuracy and loss per epoch on both training and validation data) is stored as an artifact in MLflow (`training_history.csv` and visualized in `.png`). The model with the highest validation accuracy is saved as a .joblib file in MLflow artifacts, and is used to produce an assessment of model performance on the validation dataset (e.g. saving the loss and accuracy metrics, and the confusion matrix of the validation set, `confusion_matrix.png`, all logged to MLflow).


## Importing library functions

Reusable functions can be imported from the library (`lib` subdirectory) to avoid code duplication and permit a more organized structuring of the repository.

To import library code in notebooks, you may need to add the `lab` directory to PYTHONPATH, for example as follows:

```python
import sys
from pathlib import Path

DIR_REPO = Path.cwd().parent.parent
DIR_LAB = DIR_REPO / "lab"

sys.path.append(str(DIR_LAB))

from lib.viz import plot_confusion_matrix
```

To be able to run imports from the `lib` directory on Drone, you may add it to PYTHONPATH in .drone.yml as indicated:

```yaml
environment:
  PYTHONPATH: /drone/src/lab
```

`/drone/src` is the location on the Drone runner that the repository is cloned to, and `lab` is the name of the laboratory section of our repository which includes `lib`. 
This then allows importing library functions directly from the Python script that is being executed on the runner, for instance:

```python
from lib.viz import plot_confusion_matrix
```

To see a working example, refer to the existing `application-examples` pipeline defined in .drone.yml 
(the PyTorch example pipeline uses library imports in `processes/pytorch_example/main.py`).


## Launching experiment runs (Drone)

To enable full tracability and reproducibility, all executions that generate results or artifacts 
(e.g. processed datasets, trained models, validation metrics, graphics of model validation, etc.) 
are run on Drone runners instead of the user's Jupyter or Vscode tools. 

This way, any past execution can be traced to the exact version of the code that was run (`VIEW SOURCE </>`),
and the runs can be reproduced with a click of the button in the Drone UI (`RESTART`).

To launch a pipeline, use a trigger in .drone.yml as shown below:

```yaml
trigger:
  ref:
  - refs/tags/process-data-*
```

With this trigger in place, the pipeline will be executed on Drone agents when pushing a tag (matching the pattern specified in the trigger) to the remote repository:

```bash
git tag process-data-v0
git push origin process-data-v0 
```

If using an external repository (e.g. hosted on Github), a delay in synchronization between Gitea and the mirrored external repo may cause a delay in launching the pipeline on the Drone runners. 
This delay can be overcome by manually forcing a synchronization of the repository in the Gitea UI Settings.

## Logging experiment results (MLflow)



