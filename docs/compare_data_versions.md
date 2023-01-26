# Comparing verisons data versions

Since our data are now tracked by dvc through our git history we will always have access to prior versions of those elements.
We can access them with python through the dvc.api.
To compare any element tracked by dvc, we can reference it through its git tag/branch/commit. 
This would allow us to compare any revision of our dvc tracked element. 
To do so, we can use the following code snippet:

```python
import dvc.api

# Read data from git tag tag-prior-commit
data_prior_commit = dvc.api.read(
  'lab/data/preprocess/my_data.txt',
  rev='tag-prior-commit' 
)

# Read same data but from current workingspace
current_data = dvc.api.read(
  'lab/data/preprocessed/my_data.txt' 
)
```

In this example we access a tracked file within the `lab/data/preprocess/` directory however,
any element tracked by dvc can be accessed the same way (whether it be a model, image, pre-processed data, etc.)
