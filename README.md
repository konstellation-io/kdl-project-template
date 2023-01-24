# KDL Project Template

---

## Table of contents

- [KDL Project Template](#kdl-project-template)
  - [Table of contents](#table-of-contents)
  - [Project structure](#project-structure)
  - [First steps](#first-steps)
    - [Github secrets](#github-secrets)
    - [Install dependencies](#install-dependencies)
    - [Initialize dvc](#initialize-dvc)
    - [Assign your MLFLOW URL](#assign-your-mlflow-url)
    - [Test installation](#test-installation)
    - [First CD run](#first-cd-run)
  - [Example project pipeline](#example-project-pipeline)
    - [Continuous development execution](#continuous-development-execution)
    - [Handling Process Dependencies](#handling-process-dependencies)
    - [Docker images for experiments \& trainings](#docker-images-for-experiments--trainings)

---

## Project structure

The project repository has the following directory structure:

```
├── .vscode
│   └── launch.json <- Configuration for test executions in Vscode
│   └── config.json <- Base configuration for VSCode
├── .github
│   └── workflows
│       │   └── github_actions_pipeline.yml  <- Pipeline to be run by github
├── goals         <- Acceptance criteria (typically as automated tests describing desired behaviour)
├── lab
│   ├── analysis  <- Analyses of data, models etc. (typically notebooks)
│   ├── artifacts  <- Directory to save models/training artifacts
│   ├── data  <- Directory to save data
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
- AWS_ACCESS_KEY_ID: this may change depending on your S3. Consult with the konstellation team if unclear which value this secret should have
- AWS_SECRET_ACCESS_KEY: same as with MINIO_ACCESS_KEY_ID

To add secrets to your github repository go to your github repository -> Settings -> Secrets and variables -> Actions. In there selecet `New repository secret` add the Name of your secret and is value.

### Install dependencies

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

### Initialize dvc

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
dvc remote add minio s3://<bucket_namet>/dvc --default
dvc remote modify minio endpointurl https://minio.kdl-dell.konstellation.io
dvc remote modify --local minio access_key_id <access_key_id>
dvc remote modify --local minio secret_access_key <secret_access_key>
```

Remember to update your <bucket_name> as well as  <access_key_id> and <secret_access_key>

### Assign your MLFLOW URL

Our experiment will be tracked by mlflow when run on Github Actions.
In order for Github to know where to send the new information we need to modify the environment variable in [github_actions_pipeline.yml](.github/workflows/github_actions_pipeline.yml). 
A `TODO` mark has been left to indicate where to make the modification to our bucket's name

### Test installation

To make sure our project is good to go we will first need to run the tests

``` bash
pytest
```

If tests run correctly on our user-tools, we can now see if our actions are also set.

### First CD run

With this modfications we can now commit, tag and push with git to start our run!

If the job has been run correctly we should see that a new commit has been made by our CD. 

This new commit will mantain the code it was used to execute it, 
with the addition that now it will have updated our dvc tracked artifacts
To visualize these changes we need to run:

```bash
git pull # Get our lates code versions, .dvc files and dvc.lock update
dvc pull # Get the dvc tracked data corresonding to this code version
```

We should now see that [data/processed](lab/data/processed) has new files, corresponding to the output of our `prepare_data` step.

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

The execution of the example classification pipeline on github actions is specified in [.github/workflows/github_actions_pipeline.yml](.github/workflows/github_actions_pipeline.yml).
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
The connection to MLflow to log these parameters and metrics is established via the code in the [main.py](lab/processes/train_standard_classifiers/main.py) and with the environment variables in [github_actions_pipeline.yml](github/workflows/github_actions_pipeline.yml).
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

