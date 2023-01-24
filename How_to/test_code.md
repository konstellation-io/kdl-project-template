

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