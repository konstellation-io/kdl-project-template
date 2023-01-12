# KDL Project Template

## Project structure

The project repository has the following directory structure:

```
├── .vscode
│   └── launch.json <- Configuration for test executions in Vscode
│   └── config.json <- Base configuration for VSCode
├── .github
│   └── workflows
│       │   └── experiments.yml  <- Pipeline to be run by github
├── goals         <- Acceptance criteria (typically as automated tests describing desired behaviour)
├── lab
│   ├── analysis  <- Analyses of data, models etc. (typically notebooks)
│   ├── docs      <- High-level reports, executive summaries at each milestone (typically .md)
│   └── processes           <- Source code for reproducible workflow steps.
│       ├── prepare_data
│       │   ├── main.py
│       │   ├── cancer_data.py
│       │   └── cancer_data_test.py
│       │   └── Pipfile                 <- Custom dependencies for prepare_data process
│       │   └── Pipfile.lock
|       ├── train_dnn_pytorch
│       │   ├── main.py
│       │   ├── densenet.py
│       │   └── densenet_test.py
│       │   └── Pipfile                 <- Custom dependencies for train_dnn_pytorch process
│       │   └── Pipfile.lock
│       └── train_standard_classifiers
│       │   ├── main.py
│       │   ├── classifiers.py
│       │   └── classifiers_test.py
│       │   └── Pipfile                 <- Custom dependencies for train_standard_classifiers process
│       │   └── Pipfile.lock
│       └── conftest.py        <- Pytest fixtures
├── lib           <- Importable functions used by analysis notebooks and processes scripts
├── runtimes      <- Code for generating deployment runtimes (.krt)
├── .gitignore
├── dvc.yml       <- Instructions for dvc repro and experiments
├── params.yml    <- Configuration file
├── README.md     <- Main README
└── pytest.ini    <- Pytest configuration
└── Pipfile       <- Global dependencies
└── Pipfile.lock
```

The `processes` subdirectory contains as its subdirectories the various separate processes (`prepare_data`, etc.),
which can be tought of as nodes of an analysis graph.
Each of these processes contains:

- `main.py`, a clearly identifiable main script for running on CI/CD (Drone)
- `{process}.py`, containing importable functions and classes specific to that process,
- `{process}_test.py`, containing automated unit or integration tests for this process, and

The process names from the template are not likely to generalize to other projects, so here is another example for clarity:

```
└── processes
    ├── prepare_data
    │   ├── main.py
    │   ├── (image_data).py         <- importable functions
    │   └── (image_data)_test.py    <- for automated testing
    ├── train_model
    │   ├── main.py
    │   ├── (convnet).py
    │   └── (convnet)_test.py
    └── ...
```

In the examples shown, all processes files are Python `.py` files.
However, the idea of modularizing the analysis into separate processes facilitates changing any of those processes to a different language as may be required, for example R or Julia.

## First steps

In order to start making use of this repository, certain steps need to be taken in order to have our CD running.

### Github secrets
In the github repository we will need to add the following secrets:
- MINIO_ACCESS_KEY_ID: this may change depending on your S3. Consult with the konstellation team if unclear which value this secret should have
- MINIO_SECRET_KEY_ID: same as with MINIO_ACCESS_KEY_ID

### Install dependencies
In order to start our project we will need to install the dependencies. 
These dependencies will allow us to run the template's example as well as initiate our data tracking.
To get the dependencies from the Pipfile.lock we run

```bash
pipenv sync --dev
```
If we need to modify the dependencies (either to remove, update or add dependencies),
we instead would update the Pipfile and run 
``bash
pipenv install --dev
```
Once our dependencies are installed we can start our virtual environment
```bash
pipenv shell
```

### Track data
Among our dependencies we have installed dvc. 
Dvc is a data tracking tool which will allow us to track modifications 
and control the versions of our data through git (for more information got to [dvc.org](https://dvc.org/).

To start tracking our data we first need to initiate dvc by running

```bash
dvc init
```
Now our repository is dvc tracked too!
last thing is to set our remote dvc repository to share our data

```bash
dvc remote add minio s3://<bucket_namet>/dvc -d
dvc remote modify minio endpointurl https://minio.kdl-dell.konstellation.io
dvc remote modify --local minio access_key_id <access_key_id>
dvc remote modify --local minio secret_access_key <secret_access_key>
```

### Optional: add pre commiting hooks

Optionally we may want to install dvc pre commiting hooks which automatizes common actions need when git commiting
and pushing.To do so we install the pre-commit-tool
```bash
dvc install --use-pre-commit-tool
```
If we do not take this option we must remember that:
- After any git commit, it is recommended to run dvc status to visualize if your data version also needs to be committed
- After any git push, we should run dvc push to update the remote
- After any git checkout, we must dvc checkout to update artifacts in that revision of code

### Test installation
To make sure our project is good to go we will first need to run the tests

``` bash
pytest
```
If test are run correctly locally we can now see if our actions are also set.
We will first need to modify our experiments.yml adding our mlflow url

With this modfications we can now commit, tag and push with git to start our run!

If the job has been run correctly we should see that a new commit has been made by our CD. 

This new commit will mantain the code it was used to execute it, 
with the addition that now it will have updated our dvc tracked artifacts
To visualize these changes we need to git pull and dvc pull the changes.

## Example project pipeline

KDL contains various components that need to be correctly orchestrated and connected.
To illustrate their intended usage, we provide an example machine learning pipeline already implemented in KDL.

The example pipeline is a simple classification problem based on the [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset).
The dataset contains 30 numeric features and the binary target class (benign/malignant).

The code illustrating the implementation of a machine learning pipeline in KDL is composed of three parts:

- Data preparation
- Traditional ML models (in scikit-learn)
- Neural network models (in PyTorch)

More information on each of these steps:

- **Data preparation**
  (code in [lab/processes/prepare_data/main.py](lab/processes/prepare_data/main.py)):
  the dataset is loaded from sklearn datasets and normalized;
  the transformed data are split into train, validation and test sets;
  and the processed data are stored on the shared volume.
- **Traditional ML models (in scikit-learn)**
  (code in [lab/processes/train_standard_classifiers/main.py](lab/processes/train_standard_classifiers/main.py)):
  the processed datasets are loaded from the shared volume as arrays;
  the script iterates through a number of classification algorithms,
  including logistic regression, naïve Bayes, random forest, gradient boosting, etc.;
  validation accuracy is computed and logged to MLflow.
- **Neural network models (in PyTorch)**
  (code in [lab/processes/train_dnn_pytorch/main.py](lab/processes/train_dnn_pytorch/main.py)):
  the processed datasets are loaded from the shared volume as torch DataLoaders;
  the script initiates a densely connected neural network for binary classification
  and launches its training and validation;
  the training history (accuracy and loss per epoch on both training and validation data) are stored as an artifact in MLflow (`training_history.csv` and visualized in `.png`).
  The model with the highest validation accuracy is saved as a .joblib file in MLflow artifacts, and is used to produce an assessment of model performance on the validation dataset (e.g. saving the loss and accuracy metrics, and the confusion matrix of the validation set, `confusion_matrix.png`, all logged to MLflow).

The full definition of the pipeline is defined in [dvc.yaml](dvc.yaml).
We will see the components of this pipeline later on the section [Track pipelines](##Track-pielines)

### Continuous development execution
The execution of the example classification pipeline on github actions is specified in [.github/workflows/experiments.yml](.github/workflows/experiments.yml).
We will see hat each block of code does later on the [Launching experiment runs (Github Actions)](##Launching-experiment-runs-(Github-Actions)). 
But for now we will focus on:

**Execution trigger**
```yaml
on:
  push:
    tags:
      - "run-example*"
```
To **launch the execution** of this pipeline on Github Actions, push a tag containing the name matching the defined trigger to the remote repository.
In this case, the tag pattern is `run-example-*`,
therefore to launch the execution run the following commands in the Terminal:
`git tag run-example-v0 && git push origin run-example-v0`.
For more information and examples, see the section Launching experiment runs below.

The **results of executions** will generate a new commit with the results of the execution as well as store it in MLflow.
In the example of training traditional ML models, we are only tracking one parameter (the name of the classifier) and one metric (the obtained validation accuracy). In the PyTorch neural network training example, we are tracking the same metric (validation accuracy) for comparisons, but a different set of hyperparameters, such as learning rate, batch size, number of epochs etc.
In a real-world project, you are likely to be tracking many more parameters and metrics of interest.
The connection to MLflow to log these parameters and metrics is established via the code in the [main.py](lab/processes/train_standard_classifiers/main.py) and with the environment variables in [.drone.yml](.drone.yml).
For more information on MLflow tracking, see the section "Logging experiment results (MLflow)" below.
To see the tracked experiments, visit the MLflow tool UI.

### Handling Process Dependencies

The recommended way to handle specific dependencies and versions across different processes is to have a custom `Pipfile`
inside each process folder. Only the necessary dependencies for each process need to be specified in each `Pipfile`. In this way the time of execution and preparation of the environment for each process is limited as much as possible, avoiding installing dependencies that are not necessary.

In the [dvc.yaml](dvc.yaml), dependencies can be installed as follow:

```yaml
step_name:
  cmd:  
  - cd lab/processes/prepare_data/
  - pipenv install --system
  - python main.py
```

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
To be able to run imports from the `lib` directory on Github Actions, you may add it to PYTHONPATH in experiments.yml as indicated:

```yaml
env:
  PYTHONPATH:  ${{ github.workspace }}
```

`github.workspace` is the root on the github runner that the repository is cloned to,  which includes `lib`.
This then allows importing library functions directly from the Python script that is being executed on the runner, for instance:

```python
from lib.viz import plot_confusion_matrix
```
## Data tracking and Pipeline tracking

By using dvc, we are capable of tracking data, processes and their outputs
in order to keep the status of the project in its intirety on our git history.

### Track datasets

In order to start tracking a new dataset. We must download the dataset on our user-tools' vscode. 
We can then add the data o be tracked by dvc by running
```bash
dvc add data/file.txt
```
We will see that this commands generates a new file `data/file.txt.dvc`. This new file will be tracked by our git
while a .gitignore will indicate git to not track the original file.
We can then dvc commit the new dataset and push it to our remote.
```bash
dvc commit
dvc push
```
Now our data is tracked and shared to our S3.

### Track pipelines

Tracking data is useful, but in some cases we do not only want to track the data but how the data is created.
This would be the case when we get our raw data and process it. We would like to not only track the processed data, 
but what code created it. 
To do so, we will use dvc pipelines.

We create our pipeline in `dvc.yaml`, defining all the steps required from data preprocessing, experiment training, evaluation, etc. 
This file shows the steps our pipeline needs to take in order to complete.
Each step is composed of:
 - Name: which can be use to make reference in dvc commands
 - cmd: the command(s) to be run in the step. Usually it will be a single command to run one of our scripts
 - deps: dependencies of our step. In this section we should include input data for our scripts as well as the code in which the step is dependent (The code being runned as well as any local file it requires). 
 - params: similar to deps but to the variable level. Our parameters are define in [params.yaml](params.yaml). In here we can define which parameters are relevant to our step. 
 - outs: The outputs expected for this step. This can be any directory, dataset or artifact expected from our code.
 - always_changed (optional): If set to True, dvc will always consider this step has been modified from the last execution. This makes it so that dvc always runs this step. We use this flag specially to query an untracked dataset that could be modified since the last execution (such as in the template example, where the raw dataset is hosted by scikit-learn)

It is important to define all dependencies, params and outs for each step.
When executing the pipeline, dvc will check if any dependency or parameter has been changed since last execution. 
If none has changed, dvc will just checkout the tracked output, skipping the execution of the step.
By doing so, we do not need to partition our pipeline to be executed independently. 
We can make sure our whole pipeline is sound without having to execute its entirety every time.

## Launching experiment runs (Github Actions)

To enable full tracability and reproducibility, all executions that generate results or artifacts
(e.g. processed datasets, trained models, validation metrics, plots of model validation, etc.)
are run on Github runners instead of the user's Jupyter or Vscode tools.

This way, any past execution can always be traced to the exact version of the code that was run (`Triggered` in the UI of the Action run)
and the runs can be reproduced with a click of the button in the UI of the Drone run (`Re-run jobs`).

The event that launches a pipeline execution is defined by the trigger specified in `.github/workflows/experiments.yml`
This file is divided in blocks of codes with dedicated responisibilities

**Execution trigger**
```yaml
on:
  push:
    tags:
      - "run-example*"
```

With this trigger in place, the pipeline will be executed on Github runner whenever a tag matching the pattern
specified in the trigger is pushed to the remote repository, for example:

```bash
git tag run-example-v0
git push origin run-example-v0
```

**Environment definition**
```yaml
env:
  ACCESS_TOKEN: ${{ secrets.ACCESS_TOKEN }}
  PYTHONPATH: ${{ github.workspace }}
```
Several environment variables needed in the step of the execution pipeline.

**Pipeline setup**
```yaml
runs-on: ["self-hosted", "igz", "cpu"]
steps:
  - uses: actions/checkout@v3
    ...
  - name: Pipenv setup
    run: |
      pipenv sync --system
```
These steps should not be modified since they are needed for dvc usage as well as commiting the results of our pipeline.
In case you `dvc.yaml`contemplates setting individual environments per step, you may want to skip the Pipenv setup step.

**Run pipeline**
```yaml
- name: Run Experiment
  env:
    MLFLOW_S3_ENDPOINT_URL: https://minio.kdl-dell.konstellation.io
    MLFLOW_URL: https://<bucket-name>-mlflow.kdl-dell.konstellation.io
    AWS_ACCESS_KEY_ID: ${{ secrets.MINIO_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.MINIO_SECRET_ACCESS_KEY }}
  run: |
    dvc repro --pull
```

This step is the one that executes our entire pipeline. 
dvc requires the remote information to be able to get the tracked data (as our CD will not have any).
If your project is based on an input dataset/artifact rather than the query used for the example.
You may need to add a `dvc pull data/`before the exectuion of the pipeline.

**Tracking Modifications**
```yaml
- name: Share experiment
  run: |
    raw=$(git branch -r --contains ${{ github.ref }})
    branch=${raw#*/}
    commit_message=$(git show -s --format=%B $branch)
    commit_message="${commit_message}-results"
    git add dvc.lock
    git commit -m "$commit_message"
    dvc push
    git push origin "HEAD:$branch"
```
This step will commit the updated dvc.lock so that we can view the results in our user-tools
and keep the state of our pipeline tracked.
A detail explanation of what the step is doing would be:
 - Finding the branch belonging to this commit. Since we are using a tag we need to find to which branch this tag corresponds to
 - Get the original commit message and add a _-results_ to be able to track what commit the results are from
 - Add and commit the changes done, which should only be dvc.lock. Since keeps track of our datafiles
 - Push changes both to git and dvc to the corresponding branch

TODO: Check with konstellation team on this section

### Docker images for experiments & trainings 

In the `drone.yml` file you can specify the image that is going to be used for each pipeline step.

```yml
steps:
  - name: prepare-data
    image: konstellation/kdl-py:3.9-1.1.0
  ...
```

There are two recommendations regarding which image to use:

1. Using an official runtime image. These images are used for running the KDL Usertools and have everything you need to run your code. If using one of these images take into account that the first thing you would need to do in the drone pipeline is to install your custom dependencies (`pipenv install`). You can find info about runtimes and their docker images inside KDL in the Usertools Settings section.
2. Using a custom image. For this case it is recommended to build a new layer on top of the official runtime images adding whatever you need to run your experiments/trainings.

## Logging experiment results (MLflow)

To compare various experiments, and to inspect the effect of the model hyperparameters on the results obtained, you can use MLflow experiment tracking.
Experiment tracking with MLflow enables logging the parameters with which every run was executed and the metrics of interest, as well as any artifacts produced by the run.

The experiments are only tracked from the executions on Github Actions.
In local test runs, mlflow tracking is disabled (through the use of a mock object replacing mlflow in the process code).

The environment variables for connecting to MLflow server are provided in .experiments.yml:

```yaml
env:
  MLFLOW_S3_ENDPOINT_URL: https://minio.kdl-dell.konstellation.io
  MLFLOW_URL: https://<bucket-name>-mlflow.kdl-dell.konstellation.io
```

The use of MLflow for experiment tracking is illustrated by the scikit-learn example pipeline in [lab/processes/train_standard_classifiers/main.py](lab/processes/train_standard_classifiers/main.py).

```python
import mlflow

mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

with mlflow.start_run(run_name=MLFLOW_RUN_NAME, tags=MLFLOW_TAGS):

    # (... experiment code ...)

    # Log to MLflow
    mlflow.log_param("classifier", model_name)
    mlflow.log_metric("validation_accuracy", val_acc)
```

For more information on logging data to runs, see [MLflow documentation on logging](https://www.mlflow.org/docs/latest/tracking.html#logging-data-to-runs).

Whenever one script execution trains various models (e.g. in hyperparameter search, where a model is trained with many different combinations of hyperparameters), it is helpful to use nested runs. This way, the sub-runs will appear grouped under the parent run in the MLflow UI:

```python
import mlflow

with mlflow.start_run(run_name=MLFLOW_RUN_NAME, tags=MLFLOW_TAGS):

    # (... experiment setup code shared between subruns ... )

    with mlflow.start_run(run_name=SUBRUN_NAME, nested=True, tags=MLFLOW_TAGS):

        # (... model training code ...)

        mlflow.log_metric("classifier", model_name)
        mlflow.log_param("validation_accuracy", val_acc)
```

To compare the executions and vizualise the effect of logged parameters on the logged metrics,
you can select the runs you wish to compare in the MLflow UI, select "Compare" and add the desired parameters and metrics to the visualizations provided through the UI.
Alternatively, the results can also be queried with the MLflow API. For more information on the latter, see [MLflow documentation on querying runs](https://www.mlflow.org/docs/latest/tracking.html#querying-runs-programmatically).

## Testing

To run the automated tests, you have two options: via command line or via the Vscode UI.

### Running tests from command line

You can use the command `pytest` directly from the terminal line as follows:

```
PYTHONPATH=lab pytest -v                              # Run all tests (verbose)
PYTHONPATH=lab pytest -v lab/processes/prepare_data   # Run only tests in prepare_data
PYTHONPATH=lab pytest -v -m unittest                  # Run only unit tests
PYTHONPATH=lab pytest -v -m integration               # Run only integration tests
```

You may add other optional pytest arguments as needed
(see [pytest usage documentation](https://docs.pytest.org/en/6.2.x/usage.html)).

### Running tests from Vscode UI

It is also possible to run the tests using the Vscode user interface.
To run all tests, select `Ctrl+Shift+P`, then search for `Python: Run All Tests`.

You may **run tests individually**
by clicking on the `Run Test` option next to the name of the test in the editor.
If this option does not appear next to the test,
check that your file name and test name both include the string "test\_" or "\_test",
then run `Ctrl+Shift+P` and search for `Python: Discover Tests`.

**Interactive debugging:**
Unlike the command line option, the UI option also permits the use of the interactive debugging tool in Vscode.

- First, place breakpoints in your code (by placing a marker, clicking to the left of the code line number).
- Next, select `Debug Test` next to the test (if launching individually), or `Ctrl+Shift+D` (`Python: Debug All Tests`).
- Select your test configuration "All tests" / "Integration tests" / "Unit tests" and click on the Run icon (these configurations can be edited in `.vscode/launch.json`)
- Use the Debug Console to explore the variables at the breakpoints, and the debug controls to pass between breakpoints

### Data for testing

Thanks to the use of dvc. We will always have our latest datasets in our vscode.
However if we were to need a mock dataset to avoid running a preprocessing or such we can still create a mock dataset.
This temporary dataset is provided to such tests
through the use of a test fixture defined in `conftest.py`,
and is eliminated by the same fixture after the test is executed.
The fixture is passed to any test as a function argument,
as seen in the following example (from KDL Project Template):

```python
# In conftest.py

@pytest.fixture(name="temp_data_dir", scope="module")
def temporary_cancer_data_directory(dir_temp="temp"):
    """
    Pytest fixture for those tests that require a data directory containing the cancer dataset arrays.
    As part of setup, the fixture creates those arrays in the temporary location specified by dir_temp

    Keyword Arguments:
        dir_temp {str} -- Path where the files will be temporarily generated; the directory is cleared up after
            running the test (default: {"temp"})
    """

    # Setup:
    prepare_cancer_data(dir_output=dir_temp)

    yield dir_temp

    # Teardown:
    shutil.rmtree(dir_temp)


# In test file:

def test_load_data_splits_as_npy_array(self, temp_data_dir):
    """
    Test that data splits can be loaded as numpy arrays.
    Note: requires dir_temp populated with .npy files as generated by prepare_cancer_data, prepared by
    test fixture temp_cancer_data_dir (in conftest.py)
    """
    result = load_data_splits(dir_processed=temp_data_dir, as_type="array")
    for array in result:
        assert isinstance(array, np.ndarray)

```

In the example above, the fixture `temporary_cancer_data_directory` (abbreviated with the name `temp_data_dir`)
defines what happens before and after executing a test that uses that fixture:

- Before running the test, this fixture runs through the setup code,
  creating our dataset locally using `prepare_cancer_data`.
- Next, the fixture yields the location of that directory to the test function.
- After the test function has terminated running, the fixture executes the teardown code.

If we drop the `temp_data_dir` parameter from this test function,
the test will run without the fixture,
and will fail because the required data directory does not exist.

To learn more, see the documentation on [pytest fixtures](https://docs.pytest.org/en/6.2.x/fixture.html).
