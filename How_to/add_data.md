# Adding data to the project

Depending on our problem we may have one of two types of datasets: local and external. 
Local datasets are those that can be hosted within KDL. These datasets are usually in the form of parquet files, images or videos, we can track this data directly with dvc. 
External datasets, on the other hand, are those that reside in an external database and need to be queried/downloaded (sklearn dataset, Hadoop, Big Table, Snowflake, etc.). For these cases, we will need to develop a query script that saves the data locally and track changes on this script.

## Local Dataset

To make use of a local dataset we are going to follow a two step process. First we are going to send our data to minio in order to have a safe storage of the raw data received. We then are going to bring that data to our user tools and track them with dvc. By doing so we are going to have two copies of the same data: one will reside in Minio as an insurance in case git/dvc breaks down or there is a need to restart the project, the other is going to reside in our usertools and is the one we are going to use on a daily basis.

To start we will need our data to be stored in our minio’s bucket in the directory data/raw, this copy will be untouchable. If new data were to come and needed to substitute previous data we will not overwrite the original dataset, this will be done in our user tools.

Once the data is in the bucket’s minio `<bucket_name>/data/raw` directory we will download it to our usertools. To do so, we will use our minio client to download our data. For that we will use the `cp` (copy) command indicating from where we are copying the data and the destination directory; we also add the recursive flag to ensure data within subdirectories is downloaded:

```bash
mc cp minio/{bucket_name}/data/raw data/raw –recursive
```

Once the command has been executed, we should be able to see it in our working space.

Now we need to start tracking its modifications. To do so, we are going to use dvc. Dvc commands are very similar to the git commands, although their functionality are slightly different. To start tracking a data file we need to add it with dvc. We can do this file, by file or in a glob:

```bash
dvc add lab/data/raw –recursive
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
