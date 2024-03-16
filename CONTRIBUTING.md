# Contributing

When contributing to this repository, please first [create an issue](https://github.com/finsberg/gotranx/issues/new/choose) containing information about the missing feature or the bug that you would like to fix. Here you can discuss the change you want to make with the maintainers of the repository.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## New contributor guide

To get an overview of the project, read the [documentation](https://finsberg.github.io/gotranx/). Here are some resources to help you get started with open source contributions:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

## Pull Request Process


### Pull Request

- When you're finished with the changes, create a pull request, also known as a PR. It is also OK to create a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/) from the very beginning. Once you are done you can click on the ["Ready for review"] button. You can also [request a review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review) from one of the maintainers.
- Don't forget to [link PR to the issue that you opened ](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).
- Enable the checkbox to [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a merge.
Once you submit your PR, a team member will review your proposal. We may ask questions or request for additional information.
- We may ask for changes to be made before a PR can be merged, either using [suggested changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request) or pull request comments. You can apply suggested changes directly through the UI. You can make any other changes in your fork, then commit them to your branch.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this [git tutorial](https://lab.github.com/githubtraining/managing-merge-conflicts) to help you resolve merge conflicts and other issues.
- Please make sure that all tests are passing, github pages renders nicely, and code coverage are are not lower than before your contribution. You see the different github action workflows by clicking the "Action" tab in the GitHub repository.


### Enforced style guide using pre-commit hooks

We want to have a consistent style on all the contributions to the repository. The way we enforce this is through pre-commit hooks and contributors are encouraged to install the pre-commit hooks locally when developing. You can install the pre commit hooks by first install `pre-commit`
```
python3 -m pip install pre-commit
```
and then install the pre-commit hooks using the command
```
pre-commit install
```
at the root of the repository. This will install all the hooks listed in the file called `.pre-commit-config.yaml` in the root of the repository.

Every time you make a commit to the repository a set of tests will run to make sure that the changes you made are following the style guide. Usually, the hooks will autoformat your code so that you only need to do a `git add` again and then redo the `git commit`.

Note that when you make a push to the repo, the pre-commit hooks will be run on all the files in the repository. You can also run the pre-commit hooks on all the files using the command
```
pre-commit run --all
```
To learn more about pre-commit you can check out https://pre-commit.com

## Test suite
For every new features of bugfix you should also make sure to not lower the code coverage for the test suite. This means that if you for example add a new function then you should also make sure that the function is properly tested (at a minimum it should be covered by the test suite).

To run the test suite, please install the package with the optional dependencies `test`, i.e
```
python3 -m pip install -e ".[test]"
```
in the root of the repository. To run the tests you can execute the command
```
python3 -m pytest
```
You can read more about using pytest in the [official documentation of pytest](https://docs.pytest.org/).

## Documentation
The documentation is hosted at GitHub pages and created with [`MkDocs`](https://www.mkdocs.org). Contributions to the documentation both is very welcomed.

To build the documentation locally you can installed the `docs` optional dependencies, i.e
```
python3 -m pip install -e ".[docs]"
```
in the root of the repository. Now you can build the documentation by first copying the markdown files from the root directory to the `docs` folder
```
cp *.md docs/
``
and then running
```
mkdocs build
```
For reference, please see the [github workflow](https://github.com/finsberg/gotranx/blob/main/.github/workflows/pages.yml) that is used for building the pages.
