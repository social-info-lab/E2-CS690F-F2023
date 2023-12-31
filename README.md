# Learning to Query Social Media via Interpretable ML
In this project we trained sparse decision trees using [GOSDT-guesses](https://github.com/ubc-systopia/gosdt-guesses/tree/main) to generate optimal Boolean search queries based on a set of relevant data items. To evaluate our automatic query generation in a real-world application, we examine the querying of political Twitter polls, specifically on the U.S. presidential election.

GitHub Folder Structure and Content
- /data/ --> Folder for all data, generated or raw.
  - /data/attr-lists - jsons of lists of attributes to be inputted to decision trees.
  - /data/decision-trees - text outputs of GOSDT training
  - /data/generated-queries - jsons of boolean queries
  - /data/query-data - labeled CSVs of raw data. You will need to re-import the data into your local installation.
  - /data/query-results - txt files containing metrics of best generated queries
- /out/ --> Unity sbatch script outputs
- /query_gen_scripts/ --> Main scripts for generating features for decision trees, boolean queries, and testing queries
- /sbatch/ --> Bash scripts to be used with sbatch
- /utils/ --> various utility functions
- /GOSDT/ --> GOSDT model files from the [GOSDT-guesses Repository](https://github.com/ubc-systopia/gosdt-guesses/tree/main)
- /initial_experiments/ --> initial experiments with preprocessing and GOSDT

## Installation
(We encountered several issues with the installation of gosdt as it relies on an deprecated sklearn package. We therefore recommend using Python 3.9.7 and setting the environment variable SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True)
```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install autocorrect
pip install nltk
pip install scipy
pip install attrs packaging editables sklearn sortedcontainers gmpy2
pip install gosdt

pip install -e .
```