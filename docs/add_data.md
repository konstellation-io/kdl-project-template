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

Now we need to start tracking its modifications. To do so, we are going to use dvc. Dvc commands are very similar to the git commands, although their functionality are slightly different. To start tracking a data file we need to add it with dvc. We can track single files or entire directories:

```bash
dvc add lab/data/raw
```

It is preferable to track full directories as opposed to every single file, especially when there are several of them (more than 30). In order to keep information what the dataset holds we can add a description and/or metadata

```bash
dvc add lab/data/raw --desc "Two parquet files, one with the input (X.csv) and one with expected output (y.csv)" --meta extension=.gzip
```
This would allow for future developers to understand better what to expect from our data.

After running the command we should see one or two files generated: a .dvc file and a .gitignore file.

The .gitignore file is only generated if our file was not previously ignored, 
if you do not see the .gitignore, is because it was already ignored.

The .dvc file is always generated, this file is the key for dvc.
This .dvc file is going to tracked with git, the file itself has an md5 address that dvc can interpret to collect our data.
This way git tracks our .dvc file and dvc interprets it to collect our data.

However, this tracked file only exists in our local repository. 
(Kind of having a branch that has never been pushed by git)
In order to share it with other users, we need push this update to our minio:

```bash
dvc push
```

Now, any person that access this git commit can pull the data by running

```bash
dvc pull lab/data/raw
```

#### External database

In the cases where our data resides in an external database we will need to query the versions of our data. 
To do so, we will need to develop a query script. 
This query should save the data in our directory `data/raw` within our repository. 
We could then track this data with dvc as with the static method. Nevertheless, 
in the cases where our data may change (because there is an update in the database or we want to collect different data) we may want to add the query method as part of our dvc pipeline.

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
