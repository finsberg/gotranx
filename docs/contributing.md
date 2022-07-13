# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/finsberg/gotran-parser/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in troubleshooting.
-   Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

Action Potential features could always use more documentation, whether as part of the official Action Potential features docs, in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/finsberg/gotran-parser/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to
    implement.
-   Remember that this is a volunteer-driven project, and that
    contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `gotran-parser` for local development.

* Fork the `gotran-parser` repo on GitHub.

* Clone your fork locally:

```
git clone git@github.com:your_name_here/gotran-parser.git
```

* Install your local copy into a virtual environment.

```
cd gotran-parser/
python -m venv venv
. venv/bin/activate  # Unix
# . venv\Scripts\activate # Windows
python -m pip install -e ".[dev,test,docs]"
```

* Install the pre-commit hooks

```
pre-commit install
```

* Create a branch for local development:

```
git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.


* Commit your changes

```
git add .
git commit -m "Your detailed description of your changes."
```

This will also run the pre-commit hooks. Please try to fix these issues pointed out by the pre-commit hooks before pushing. However, if you are unable to do so, that is also fine. To ignore the pre-commit hooks you can add the `--no-verify` flag to the `git commit` command, e.g

```
git commit -m "Your detailed description of your changes." --no-verify
```


* Push your branch to GitHub:

```
git push origin name-of-your-bugfix-or-feature
```

* Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring.
3. Make sure all tests are passing for all supported python versions by checking out the CI at <https://github.com/finsberg/gotran-parser/actions>

## Tips

To run a subset of tests starting with `test_something` do:

```
python -m pytest -k test_something
```


## Deploying

A reminder for the maintainers on how to deploy. Make sure all your changes are committed. Then run:

```
bump2version patch # possible: major / minor / patch
git push
git push --tags
```

Github actions will then make sure that a new version is uploaded to PyPi.
