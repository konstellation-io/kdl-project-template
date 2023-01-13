# KDL Project Template

---

## Table of contents

- [KDL Project Template](#kdl-project-template)
  - [Table of contents](#table-of-contents)
  - [Project structure](#project-structure)
  - [First steps](#first-steps)
    - [Github secrets](#github-secrets)
    - [Install dependencies](#install-dependencies)
    - [Track data](#track-data)
    - [Assign your MLFLOW URL](#assign-your-mlflow-url)
    - [Optional - pre-commit](#optional---pre-commit)
    - [Test installation](#test-installation)
  - [Example project pipeline](#example-project-pipeline)
    - [Continuous development execution](#continuous-development-execution)
    - [Handling Process Dependencies](#handling-process-dependencies)
  - [Importing library functions](#importing-library-functions)
  - [Data Version Control (DVC)](#data-version-control-dvc)
    - [Adding data](#adding-data)
      - [Local Dataset](#local-dataset)
      - [External database](#external-database)
    - [Pipeline (dvc.yaml)](#pipeline-dvcyaml)
    - [Comparing verisons](#comparing-verisons)
  - [Launching experiment runs (Github Actions)](#launching-experiment-runs-github-actions)
    - [Docker images for experiments \& trainings](#docker-images-for-experiments--trainings)
  - [Logging experiment results (MLflow)](#logging-experiment-results-mlflow)
  - [Model registry and going to production](#model-registry-and-going-to-production)
  - [Testing](#testing)
    - [Running tests from command line](#running-tests-from-command-line)
    - [Running tests from Vscode UI](#running-tests-from-vscode-ui)
    - [Data for testing](#data-for-testing)

---

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

In order to start making use of this repository, certain steps need to be taken in order to have our CD running and our data tracked.
Only one team member is required to follow these steps. After which, the rest of team members just need to make sure to be up to date with the last git commit.

### Github secrets

In the github repository we will need to add the following secrets:
- MINIO_ACCESS_KEY_ID: this may change depending on your S3. Consult with the konstellation team if unclear which value this secret should have
- MINIO_SECRET_KEY_ID: same as with MINIO_ACCESS_KEY_ID

To add secrets to your github repository go to your github repository -> Settings -> Secrets and variables -> Actions. In there selecet `New repository secret` add the Name of your secret and is value.

### Install dependencies

In order to start our project we will need to install the required dependencies. 
These dependencies will allow us to run the template's example as well as initiate our data tracking.
To get the dependencies from the Pipfile.lock we run

```bash
pipenv sync --dev
```

If we need to modify the dependencies (either to remove, update or add dependencies),
we instead would update the Pipfile and run 

```bash
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

To start tracking our data we first need to initiate a dvc repository by running

```bash
dvc init
```
Now our repository is dvc tracked too. We can now locally track our data and pipeline executions.
However, to be able to share these updates with our teammates as well as Github Actions we need to add a remote
To do so we run the following commands:

```bash
dvc remote add minio s3://<bucket_namet>/dvc -d
dvc remote modify minio endpointurl https://minio.kdl-dell.konstellation.io
dvc remote modify --local minio access_key_id <access_key_id>
dvc remote modify --local minio secret_access_key <secret_access_key>
```
Remember to update your <bucket_name> as well as  <access_key_id> and <secret_access_key>

### Assign your MLFLOW URL

Our experiment will be tracked by mlflow when run on Github Actions.
In order for Github to know where to send the new information we need to modify the environment variable in [experiments.yml](.github/workflows/experiments.yml). A `TODO` mark has been left to indicate where to make the modification to our bucket's name

### Optional - pre-commit

Code quality and security are important aspects of software development. To help with this, we have included a [pre-commit](https://pre-commit.com/index.html) configuration file that will run a series of checks on your code before you commit it. This will help you to catch issues before they are committed to the repository.

First of all, we have provided pre-commit as a dev package in the Pipfile, so you don't need to worry about having it installed. However, you will need to install the pre-commit hooks into your local repository. It is not mandatory to use pre-commit, but we encourage you to do so since it establishes a good baseline.

Secondly, you will need to install the pre-commit hooks into your local repository. Pre-commit hooks are scripts that are run before you commit your code. They can check for things like linting errors, security issues, etc. To install the pre-commit hooks, run the following command:

```bash
pre-commit install --install-hooks
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

### Test installation

To make sure our project is good to go we will first need to run the tests

``` bash
pytest
```
If tests are run correctly locally we can now see if our actions are also set.
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
`git tag run-example-v0.0.0 && git push origin run-example-v0.0.0`.
For more information and examples, see the section Launching experiment runs below.

The **results of executions** will generate a new commit with the results of the execution as well as store it in MLflow.
In the example of training traditional ML models, we are only tracking one parameter (the name of the classifier) and one metric (the obtained validation accuracy). In the PyTorch neural network training example, we are tracking the same metric (validation accuracy) for comparisons, but a different set of hyperparameters, such as learning rate, batch size, number of epochs etc.
In a real-world project, you are likely to be tracking many more parameters and metrics of interest.
The connection to MLflow to log these parameters and metrics is established via the code in the [main.py](lab/processes/train_standard_classifiers/main.py) and with the environment variables in [experiments.yml](github/workflows/experiments.yml).
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

## Data Version Control (DVC)

### Adding data

Depending on our problem we may have one of two types of datasets: local and external. 
Local datasets are those that can be hosted within KDL. These datasets are usually in the form of parquet files, images or videos, we can track this data directly with dvc. 
External datasets, on the other hand, are those that reside in an external database and need to be queried/downloaded (sklearn dataset, Hadoop, Big Table, Snowflake, etc.). For these cases, we will need to develop a query script that saves the data locally and track changes on this script.

#### Local Dataset

To make use of a local dataset we are going to follow a two step process. First we are going to send our data to minio in order to have a safe storage of the raw data received. We then are going to bring that data to our user tools and track them with dvc. By doing so we are going to have two copies of the same data: one will reside in Minio as an insurance in case git/dvc breaks down or there is a need to restart the project, the other is going to reside in our usertools and is the one we are going to use on a daily basis.

To start we will need our data to be stored in our minio’s bucket in the directory data/raw, this copy will be untouchable. If new data were to come and needed to substitute previous data we will not overwrite the original dataset, this will be done in our user tools.

Once the data is in the bucket’s minio `<bucket_name>/data/raw` directory we will download it to our usertools. To do so, we will use our minio client to download our data. For that we will use the `cp` (copy) command indicating from where we are copying the data and the destination directory; we also add the recursive flag to ensure data within subdirectories is downloaded:

```bash
mc cp minio/{bucket_name}/data/raw data/raw –recursive
```

Once the command has been executed, we should be able to see it in our working space.

Now we need to start tracking its modifications. To do so, we are going to use dvc. Dvc commands are very similar to the git commands, although their functionality are slightly different. To start tracking a data file we need to add it with dvc. We can do this file, by file or in a glob:

```bash
dvc add data/raw/** –recursive
```

The recursive flag indicates to dvc to track each individual file. Dvc can track full directories, however this is NOT recommended, since it would not be clear what its contents are and can induce errors. If a new user were to access our repository, they would be able to understand the type and quantity of data our repository needs without having to pull the data.

With the command runned we should see that for each of our file a file_name.dvc is created. This .dvc file is the connection between git and dvc. This file will be tracked by git, and dvc will be able to interpret it as a version of our data that stores in our minio’s dvc directory. Now, any modification to the data will be recorded by dvc, dvc will then update our .dvc file and git will track this last file.

To be able to commit changes in files tracked by dvc, we will use the same command as git:

```bash
dvc commit
```

And to be able to share it with other user, we will push this update to the repository:

```bash
dvc push
```

This last command can be hooked to our git push in order to automate that every time a new code version is updated on origin, the corresponding dvc is also updated. (see [pre-commit](#optional---pre-commit))

#### External database

In the cases where our data resides in an external database we will need to query the versions of our data. To do so, we will need to develop a query script or method. This query should save the data in our directory `data/raw` within our repository. We could then track this data with dvc as with the static method. Nevertheless, in the cases where our data may change (because there is an update in the database or we want to collect different data) we may want to add the query method as part of our dvc pipeline.

An example on how to use external databases is given by the template-example in the step prepare_data

```yaml
prepare_data:
  cmd: python3 lab/processes/prepare_data/main.py
  deps:
    - lab/processes/prepare_data/main.py
    - lab/processes/prepare_data/cancer_data.py
    - lib/pytorch.py
  outs:
    - ${paths.dir_processed}
  always_changed: true
```

We will go into the details on the components of `dvc.yaml` in [dvc.yaml](#dvc.yaml).
For now, we just need to understand that this stage of the pipeline will execute our query and track the versioning of the data that would be saved in data/raw.
We set the always_changed flag to indicate that this stage should always be run since we do not know when the data at the source may have change.
If the results of the query does not change from execution to execution, dvc will realize and will NOT generate a new version of our data.

### Pipeline (dvc.yaml)

Dvc not only tracks versions of our files, it can also track the pipelines that lead to other data, code, models, metrics, etc. This is done through the dvc pipelines.

The default pipeline file for dvc is the dvc.yaml.
Whenever we execute a dvc command without any targets, it will search for this file.
If our project only requires one pipeline we recommend saving this pipeline on the `dvc.yaml` file.
If we require more than one pipeline we  will have to generate a seperate directory for each pipeline, but the yaml file must always be named `dvc.yaml`. We will then need to add as a target when running dvc repro.

dvc.yaml files are divided in steps. These steps are define by the following fields:
 - cmd (command): the command (or commands) to run on this step.This is where we will indicate which script to execute
 - deps (Dependencies): such as the input data and the scripts that are needed for this step. These dependencies determine if the command needs to be executed or not, if no changes have been made to the dependencies, the command will be skipped and the outputs will be cached
 - outs (outputs): any outputs of the step such as processed data or models. These are particularly important if the following step depends on this output (if this is the case, the artifact should appear in this step’s outs and the following step’s deps). If the step is skipped, these artifacts will be cached.
 - params (parameters): tracked variables in our code (such as epochs, tree depth, layer sized, etc.). The parameters themselves can reside in a config file (either .py, .yaml or .init).
 - always_changed flag: An optional flag to indicate to dvc to always run the step. Default is False

It is important to define all dependencies, params and outs for each step.

This pipeline can now be executed by dvc. To do so we just need to run the command:

```bash
dvc repro
```

In doing so, dvc will reproduce the pipeline. 
When dvc reproduces a pipeline it will go step by step. 
In each step it will first verify if the dependencies have been changes since the last reproduction 
(if this is the first time reproducing , it will always be considered changed). 
If modifications have been made, the step’s cmd will be executed and the outputs will be tracked. 
If a steps dependencies and parameters have not changed since the last execution, 
the step will be skipped and instead its outputs will be taken from cached.

### Comparing verisons

Since our data, models, artifacts are now tracked by dvc through our git history we will always have access to prior versions of those elements.
We can access them with python through the dvc.api.
To compare any element tracked by dvc we can use the following code snipet:

```python
import dvc.api

data_prior_commit = dvc.api.read(
  'data/my_data.txt',
  rev='tag-prior-commit'
)

current_data = dvc.api.read(
  'data/my_data.txt'
)
```

In this example we access a tracked file within the `data` folder however,
any element tracked by dvc can be accessed the same way (whether it be a model, image, pre-processed data, etc.)

## Launching experiment runs (Github Actions)

To enable full tracability and reproducibility, all executions that generate results or artifacts
(e.g. processed datasets, trained models, validation metrics, plots of model validation, etc.)
are run on Github runners instead of the user's Jupyter or Vscode tools.

This way, any past execution can always be traced to the exact version of the code that was run (`Triggered` in the UI of the Action run)
and the runs can be reproduced with a click of the button in the UI of Github Actions (`Re-run jobs`).

The execution pipeline that github is going to reference is find in [`.github/workflows/experiments.yml`](.github/workflows/experiments.yml). 
This execution pipeline is NOT the training pipeline,
it just prepares github to be able to run our experiments and share them.
Thus, little to no modifications are needed in order to use it.

This file is divided in blocks of codes with dedicated responisibilities

**Execution trigger**
```yaml
on:
  push:
    tags:
      - "run-example*"
```

This block defines the type of tag that would trigger our CD. 
You may change the tag as needed for your project. 

**Environment definition**

```yaml
env:
  ACCESS_TOKEN: ${{ secrets.ACCESS_TOKEN }}
  PYTHONPATH: ${{ github.workspace }}
  AWS_ACCESS_KEY_ID: ${{ secrets.MINIO_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.MINIO_SECRET_ACCESS_KEY }}
  MLFLOW_S3_ENDPOINT_URL: https://minio.kdl-dell.konstellation.io
  MLFLOW_URL: https://<bucket-name>-mlflow.kdl-dell.konstellation.io
```

Several environment variables needed in the step of the execution pipeline.
As mentioned in [first steps](#Assign-your-MLFLOW-URL), remember to modify the `MLFLOW_URL` variable

**Machine selection**

```yaml
runs-on: ["self-hosted", "igz", "cpu"]
```

In here we define the machine where to run our Github Action. 
Modify as needed according to your requirements (hosting machine, cpu/gpu, etc.)

**Library installation**

```yaml
steps:
  - uses: actions/checkout@v3
    ...
  - name: Pipenv setup
    run: |
      pipenv sync --system
```

These steps should not be modified since they are needed for dvc usage as well as commiting the results of our pipeline.
In case your `dvc.yaml`contemplates setting individual environments per step, you may want to skip the Pipenv setup step.

**Run pipeline**

```yaml
- name: Run Experiment
  run: |
    dvc repro --pull
```

This step is the one that executes our entire pipeline define at [`dvc.yaml`](dvc.yaml). 
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

If the process has been runned correctly, we should see a new commit on our branch in the cloud repository.

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

The environment variables for connecting to MLflow server are provided in [experiments.yml](.github/workflows/experiments.yml):

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

## Model registry and going to production

To make a smooth transition between the KAI LAB and the KRE it is important that each of our experiments have their models ready for production.

In order to do so, we are going to use dvc pipeline tracking.
To register models, and other artifacts, 
we should include them as part of our step's outs as well as save them through our code.
To register through our pipeline we just need to add them as our outs:

```yaml
train_dnn_pytorch:
  ...
  outs:
    - artifacts/dnn_pytorch/
```

Now any element saved in our `artifacts/dnn_pytorch` directory will be tracked by dvc.
The only element missing is to make sure our scripts save our artifacts and models to that directory

```python
torch.save(model.state_dict(), 'artifacts/dnn_pytorch/densenet.pt')
```

By ensuring these elements are in place we can now run experiments modifying our code.
Once one of our experiments reaches our desired metrics, we can merge the commit dedicated to that experiment to main.
In that commit we will find the code for the entire training pipeline, 
the data used and the model needed for production.

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

