
# Pipeline (dvc.yaml)

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