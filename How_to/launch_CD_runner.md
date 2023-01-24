
## Launching experiment runs (Github Actions)

To enable full tracability and reproducibility, all executions that generate results or artifacts
(e.g. processed datasets, trained models, validation metrics, plots of model validation, etc.)
are run on Github runners instead of the user's Jupyter or Vscode tools.

This way, any past execution can always be traced to the exact version of the code that was run (`Triggered` in the UI of the Action run)
and the runs can be reproduced with a click of the button in the UI of Github Actions (`Re-run jobs`).

The execution pipeline that github is going to reference is find in [`.github/workflows/github_actions_pipeline.yml`](.github/workflows/github_actions_pipeline.yml). 
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
