
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
