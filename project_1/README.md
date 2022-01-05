# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In the first project of the ND we try to predict the churn of credit card customers. The data we use is stemming from
[Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code). Our starting point is the
[churn notebook](churn_notebook.ipynb), the aim is to use that as base for refactoring it into a python project which
follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested).

The project folder has the following structure:

```bash
└── project_1
    ├── README.md
    ├── churn_library_solution.py
    ├── churn_library_orig.py
    ├── churn_notebook.ipynb
    ├── churn_script_logging_and_tests.py
    ├── churn_script_logging_and_tests_orig.py
    ├── data
    │   └── bank_data.csv
    ├── images
    │   ├── eda
    │   └── results
    ├── logs
    ├── models
    │   ├── logistic_model.pkl
    │   └── rfc_model.pkl
    ├── poetry.lock
    └── pyproject.toml
```

Note the files with the `_orig.py` prefixes, these are the original project files, the ones without the prefix contains 
the solutions.


## Running Files
We use pyenv and poetry for dependency management. Thus pyenv should be available on your system (e.g. via Homebrew).
We use thy python version 3.8.12 for this project. 
- install the needed python version via virtualenv ` pyenv install 3.8.12 `
- create the virtualenv for the project as `pyenv virtualenv 3.8.12 ml_devops_p1`
- activate the virtualenv as `pyenv activate ml_devops_p1`
- make sure poetry is available e.g. `pip install poetry`
- install all the projects dependencies as `poetry install` in the project root

In case you need additional packages you can add them as `poetry add package-name==version`, optionally with exact 
versions. Packages can be updated also with poetry as `poetry update`.  






