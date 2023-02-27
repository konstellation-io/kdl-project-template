# KDL Project Template

---

|  Component  | Coverage  |  Bugs  |  Maintainability Rating  |
| :---------: | :-----:   |  :---: |  :--------------------:  |
|  KDL Project Template  | [![coverage][coverage-badge]][coverage-link] | [![bugs][bugs-badge]][bugs-link] | [![mr][mr-badge]][mr-link] |

[coverage-badge]: https://sonarcloud.io/api/project_badges/measure?project=konstellation-io_kdl-project-template&metric=coverage
[coverage-link]: https://sonarcloud.io/api/project_badges/measure?project=konstellation-io_kdl-project-template&metric=coverage

[bugs-badge]: https://sonarcloud.io/api/project_badges/measure?project=konstellation-io_kdl-project-template&metric=bugs
[bugs-link]: https://sonarcloud.io/component_measures?id=konstellation-io_kdl-project-template&metric=Reliability

[mr-badge]: https://sonarcloud.io/api/project_badges/measure?project=konstellation-io_kdl-project-template&metric=sqale_rating
[mr-link]: https://sonarcloud.io/component_measures?id=konstellation-io_kdl-project-template&metric=Maintainability

## Table of contents

- [KDL Project Template](#kdl-project-template)
  - [Table of contents](#table-of-contents)
  - [Project structure](#project-structure)
  - [First steps](#first-steps)
    - [Github secrets (One team member)](#github-secrets-one-team-member)
    - [Install dependencies (All team members)](#install-dependencies-all-team-members)
    - [Initialize dvc (One team member)](#initialize-dvc-one-team-member)
    - [Set local dvc configurations (All team members)](#set-local-dvc-configurations-all-team-members)
    - [Assign your MLFLOW URL and git repo to your workflows (One team member)](#assign-your-mlflow-url-and-git-repo-to-your-workflows-one-team-member)
    - [Test installation (Optional)](#test-installation-optional)
    - [First workflow (Optional)](#first-workflow-optional)
    - [Aplying experiment (Optional)](#aplying-experiment-optional)
  - [Example project pipeline](#example-project-pipeline)
  - [Handling Process Dependencies](#handling-process-dependencies)
  - [Importing library functions](#importing-library-functions)
  - [Testing](#testing)
    - [Running tests from command line](#running-tests-from-command-line)
    - [Running tests from Vscode UI](#running-tests-from-vscode-ui)
    - [Data for testing](#data-for-testing)
  - [Logging experiment results (MLflow)](#logging-experiment-results-mlflow)
  - [Docker images for experiments \& trainings](#docker-images-for-experiments--trainings)
  - [Optional - installing pre-commit](#optional---installing-pre-commit)
  - [Information on dvc and further reading](#information-on-dvc-and-further-reading)

---

## Project structure

The project repository has the following directory structure:

```
├── .vscode
│   ├── launch.json <- Configuration for test executions in Vscode
│   └── config.json <- Base configuration for VSCode
├── .github
│   └── workflows
│       └── github_actions_pipeline.yml  <- Pipeline to be run by github
├── goals         <- Acceptance criteria (typically as automated tests describing desired behaviour)
├── lab
│   ├── analysis  <- Analyses of data, models etc. (typically notebooks)
│   ├── data  <- Directory to save data
│   │   ├── raw <- Directory for raw data
│   │   └── processed <- Directory for processed data
│   ├── docs      <- High-level reports, executive summaries at each milestone (typically .md)
│   └── processes           <- Source code for reproducible workflow steps.
│       ├── prepare_data
│       │   ├── main.py
│       │   ├── cancer_data.py
│       │   ├── cancer_data_test.py
│       │   ├── Pipfile                 <- Custom dependencies for prepare_data process
│       │   └── Pipfile.lock
|       ├── train_dnn_pytorch
│       │   ├── main.py
│       │   ├── densenet.py
│       │   ├── densenet_test.py
│       │   ├── Pipfile                 <- Custom dependencies for train_dnn_pytorch process
│       │   └── Pipfile.lock
│       └── train_standard_classifiers
│       │   ├── main.py
│       │   ├── classifiers.py
│       │   ├── classifiers_test.py
│       │   ├── Pipfile                 <- Custom dependencies for train_standard_classifiers process
│       │   └── Pipfile.lock
│       └── conftest.py        <- Pytest fixtures
├── lib           <- Importable functions used by analysis notebooks and processes scripts
├── runtimes      <- Code for generating deployment runtimes (.krt)
├── .gitignore
├── dvc.yml       <- Instructions for dvc repro and experiments
├── params.yml    <- Configuration file
├── README.md     <- Main README
├── pytest.ini    <- Pytest configuration
├── Pipfile       <- Global dependencies
└── Pipfile.lock
```

The `processes` subdirectory contains as its subdirectories the various separate processes (`prepare_data`, etc.),
which can be tought of as nodes of an analysis graph.
Each of these processes contains:

- `main.py`, a clearly identifiable main script for running on CI/CD (Github Actions)
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

In order to start making use of this repository,
certain steps need to be taken in order to have our CD running and our data tracked.
Most of the step just required one team member to execute.
After which, the rest of team members just need to make sure to be up to date with the last git commit.

### Github secrets (One team member)

In the github repository we will need to add the following secrets:
- AWS_ACCESS_KEY_ID: this may change depending on your S3. Consult with the konstellation team if unclear which value this secret should have
- AWS_SECRET_ACCESS_KEY: same as with MINIO_ACCESS_KEY_ID
- REPO_TOKEN: A Personal token from a member of the project taken from github

To add secrets to your github repository go to your github repository -> Settings -> Secrets and variables -> Actions.
In there selecet `New repository secret` add the Name of your secret and is value.

### Install dependencies (All team members)

In order to start our project we will need to install the required dependencies.
These dependencies will allow us to run the template's example as well as initiate our data tracking.
To get the dependencies from the Pipfile.lock we run

```bash
pipenv sync --dev
```

Once our dependencies are installed we can start our virtual environment

```bash
pipenv shell
```

### Initialize dvc (One team member)

Among our dependencies we have installed dvc.
Dvc is a data tracking tool which will allow us to track modifications
and control the versions of our data through git (for more information got to [dvc.org](https://dvc.org/).

To start tracking our data we first need to initiate a dvc repository by running

```bash
dvc init
```

Now our repository is dvc tracked too. We can now locally track our data and pipeline executions.
However, to be able to share these updates with our teammates as well as Github Actions we need to add a remote
To do so we run the following commands:

```bash
dvc remote add minio s3://<project_name>/dvc --default
dvc remote modify minio endpointurl https://minio.kdl-dell.konstellation.io
```

Remember to update your <project_name>.

### Set local dvc configurations (All team members)
In order to get access to our minio,
we will need to set our access keys.
These access keys are common for the entire team (usually).
However it is not secure to share them in git, which means that each team member must set them locally.
To do so we run the following commands:

```bash
dvc remote modify --local minio access_key_id <access_key_id>
dvc remote modify --local minio secret_access_key <secret_access_key>
```

### Assign your MLFLOW URL and git repo to your workflows (One team member)

Our experiment will be tracked by mlflow when run on Github Actions.
In order for Github to know where to send the new information we need to modify the environment variable in [github_actions_pipeline.yml](.github/workflows/github_actions_pipeline.yml)
and [create_pr.yml](.github/workflows/create_pr.yml).
A `TODO` mark has been left to indicate where to make the modification to our project_name's name

### Test installation (Optional)

To make sure our project is good to go we will first need to run the tests

``` bash
pytest
```

If tests run correctly on our user-tools, we can now see if our actions are also set.

### First workflow (Optional)

With these modfications we can now commit and push with git to start our run!
In order to trigger a github run we need to add a commit message with the estructure "experiment: <commit_message>".
By pushing that commit we should see in our github actions the run being executed.
The results of the experiment will be save as usual in MlFlow

### Aplying experiment (Optional)

Once we are happy with the results of one of our experiments, we can merge them to main.
To do so, we must go to the Actions section of our github repository and look for the workflow create PR.

In there, we will see a button that says Run workflow:

A prompt will ask us to give the name of the experiment.
This name should be recorded in mlflow as the experiments tag
and should coincide to the first 7 characters of the commit's SHA that launched the experiment
(otherwise known as the short-SHA).

By giving the name,
the workflow will run, creating a new branch for our experiment from its original commit.
It will then reproduce the experiment and create a PR to main.
We can then, visit the branch to add any additional files or open discussion with our colleagues.

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


## Handling Process Dependencies

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
To be able to run imports from the `lib` directory on Github Actions, you may add it to PYTHONPATH in your [github workflows](.github/workflows) as indicated:

```yaml
env:
  PYTHONPATH:  ${{ github.workspace }}
```

`github.workspace` is the root on the github runner that the repository is cloned to, which includes `lib`.
This then allows importing library functions directly from the Python script that is being executed on the runner, for instance:

```python
from lib.viz import plot_confusion_matrix
```

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

If we drop the `temp_data_dir` parameter from this test function, the test will run without the fixture, and will fail because the required data directory does not exist.

To learn more, see the documentation on [pytest fixtures](https://docs.pytest.org/en/6.2.x/fixture.html).

## Logging experiment results (MLflow)

To compare various experiments, and to inspect the effect of the model hyperparameters on the results obtained, you can use MLflow experiment tracking.
Experiment tracking with MLflow enables logging the parameters with which every run was executed and the metrics of interest, as well as any artifacts produced by the run.

The experiments are only tracked from the executions on Github Actions.
In local test runs, mlflow tracking is disabled (through the use of a mock object replacing mlflow in the process code).

The environment variables for connecting to MLflow server are provided in [github workflows](.github/workflows):

```yaml
env:
  MLFLOW_S3_ENDPOINT_URL: https://minio.kdl-dell.konstellation.io
  MLFLOW_URL: https://<project-name>-mlflow.kdl-dell.konstellation.io
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

## Docker images for experiments & trainings

In the  [github workflows](.github/workflows) files you can specify the image that is going to be used for each pipeline step.

```yml
job_name:
    runs-on: ["igz", "dell", "cpu"]
    container:
      image: konstellation/kdl-py:3.9-1.5.0
  ...
```

There are two recommendations regarding which image to use:

1. Using an official runtime image. These images are used for running the KDL Usertools and have everything you need to run your code. If using one of these images take into account that the first thing you would need to do in the drone pipeline is to install your custom dependencies (`pipenv install`). You can find info about runtimes and their docker images inside KDL in the Usertools Settings section.
2. Using a custom image. For this case it is recommended to build a new layer on top of the official runtime images adding whatever you need to run your experiments/trainings.

## Optional - installing pre-commit

Code quality and security are important aspects of software development. To help with this, we have included a [pre-commit](https://pre-commit.com/index.html) configuration file that will run a series of checks on your code before you commit it. This will help you to catch issues before they are committed to the repository.

First of all, we have provided pre-commit as a dev package in the Pipfile, so you don't need to worry about having it installed. However, you will need to install the pre-commit hooks into your local repository. It is not mandatory to use pre-commit, but we encourage you to do so since it establishes a good baseline.

Secondly, you will need to install the pre-commit hooks into your local repository. Pre-commit hooks are scripts that are run before you commit your code. They can check for things like linting errors, security issues, etc. To install the pre-commit hooks, run the following command:

```bash
pre-commit install --install-hooks -t pre-commit -t pre-push -t post-checkout
```

You can learn more about the different ways of installing hooks in your repository clone in this [Github issue](https://github.com/pre-commit/pre-commit.com/issues/255).

Now you're ready to begin checking your code. Pre-commit can check your code in two different ways:

1. Adding files to the Git staging area and then committing your changes will execute pre-commit only on those files that have changed since the last commit.
1. Running `pre-commit run -a` will execute all hooks on every single file of your repository without needing to perform Git commands.

Independent of how you choose to run pre-commit, you will see a list of all the checks being performed. If any of the checks fail, depending on the hook, files will be modified, or you will see a warning. For example:

- If you have forgotten to add a blank line at the end of a file, the `end-of-file-fixer` hook will add it for you. The commit command will fail, and if you run `git status`, you'll find that your file has been modified; therefore, you will need to add it to the staging area again. Performing the same commit again will effectively create a commit in your repository.

```bash
check for added large files..............................................Passed
trim trailing whitespace.................................................Passed
check for merge conflicts................................................Passed
check for broken symlinks............................(no files to check)Skipped
check yaml...............................................................Passed
fix end of files.........................................................Failed
- hook id: end-of-file-fixer
- exit code: 1
- files were modified by this hook

Fixing README.md

don't commit to branch...................................................Passed
Detect hardcoded secrets.................................................Passed
```

- If your code includes some sort of secret you have forgotten to ignore, the `gitleaks` hook will detect a high entropy string and warn you about it. This time no automatic action will be done on your behalf; you will need to fix the issue before you can commit your code.

```bash
check for added large files..............................................Passed
trim trailing whitespace.................................................Passed
check for merge conflicts................................................Passed
check for broken symlinks............................(no files to check)Skipped
check yaml...............................................................Passed
fix end of files.........................................................Failed
- hook id: end-of-file-fixer
- exit code: 1
- files were modified by this hook

Fixing .nada

don't commit to branch...................................................Passed
Detect hardcoded secrets.................................................Failed
- hook id: gitleaks
- exit code: 1

○
    │╲
    │ ○
    ○ ░
    ░    gitleaks

Finding:     AWS_ACCESS_KEY_ID=REDACTED
Secret:      REDACTED
RuleID:      aws-access-token
Entropy:     3.684184
File:        .nada
Line:        1
Fingerprint: .nada:aws-access-token:1

7:11PM INF 1 commits scanned.
7:11PM INF scan completed in 75ms
7:11PM WRN leaks found: 1
```

Check these links for a complete list of the [configured](.pre-commit-config.yaml) and [available](https://pre-commit.com/hooks.html) pre-commit hooks, some of which we make up a common-sense baseline and others that, depending on your project's nature, could make sense to add.

Finally, you also can completely avoid using pre-commit by adding the `--no-verify` flag to your commit command. This will skip all pre-commit checks and commit your code as usual. Also, there could be some situations where you would desire to apply pre-commit rules only to a portion of your code. For example, say you want to run pre-commit on your code but don't apply the changes made to a specific file. To achieve this behavior, you can run pre-commit and let it modify your files, which will be removed from Git's staging area. Then, `git add` those files whose change you want to commit, and `git checkout <filename>` the ones whose modification you wish to override. Then, create the commit using `--no-verify`, as explained above.

If we do not take this option we must remember that:
- After any git commit, it is recommended to run dvc status to visualize if your data version also needs to be committed
- After any git push, we should run dvc push to update the remote
- After any git checkout, we must dvc checkout to update artifacts in that revision of code

## Information on dvc and further reading

For more information on dvc and its usage,
as well as any other information on how to use the KDL template,
please refer to our (confluence page)[https://intelygenz.atlassian.net/wiki/spaces/K/pages/81362945/Introduction+to+dvc]
