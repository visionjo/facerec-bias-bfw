# Source code for producing the results and figures

The code is divided between Python modules in `facebias` and Jupyter notebooks
in `notebooks`. The modules implement the methodology and code that is reused
in different applications. This code is tested using `pytest` with the test
code in `tests`. The notebooks perform the data analysis and processing and
generate the figures for the paper.

The `Makefile` automates all processes related to executing code.
Run the following to perform all actions from building the software to
generating the final figures:

    make all


## Python `facebias`

Analyze, both quantitatively and qualitatively, the bias present in face verification. Specifically, focusing on different subgroups making up the Balanced Faces in the Wild (BFW) dataset, such as split by gender (i.e., Male / Female), ethnicity (Asian, Black, Indian, White), or both (i.e., ethnicities split by gender).

`facebias` enables the reproducing of results from paper, [_Face Recognition: Too Bias, or Not Too Bias_](../manuscript/latest-version.pdf). Furthermore, the aim of this package is to allow for all experiments to be generalizable to other related problem and data domains. Check out list of [Notebooks](#notebooks) for details of features and capabilities of `facebias`. 

PR, bug reports, issues, etc., are welcome and would be appreciated :) 


## Building, testing, and linting

Use the `Makefile` to build, test, and lint the software:

* Build and install:

        make build

* Run the static checks using flake8 and pylint:

        make check

* Run the tests in `tests` and doctests in docstrings:

        make test

* Calculate the test coverage of the main Python code (not including the
  notebooks):

        make coverage


## Generating results and figures

The Jupyter notebooks produce most of the results and figures. The `Makefile`
can execute the notebooks to generate these outputs. This is better than
executing the notebooks by hand because it ensures that cells are run
sequentially in a way that can be reproduced.

* Generate all results files specified in the `Makefile`:

        make results

* Create all figure files specified in the `Makefile`:

        make figures

## To Do

- [x] create demo 1: notebook to compare features, i.e., create 'score' column
- [x] facebias.io: add function overwrite datatable with option to append columns to existing dataframe if file exists; else, save as is
- [x] facebias.io: add function to load features as dictionary
- [x] create demo 2: generate SDM curves
- [x] create demo 3: generate ROC Curves
- [x] create demo 4: generate confusion matrix (Rank-1 analysis) curves
- [ ] create demos 6: determine optimal threshold per fold for each subgroup
- [ ] create demos 7: output TAR@FAR table, latex formatteds
- [ ] add notebooks to nbviewer and include in proceednig section (once public)

<a name="notebooks"></a>
## Notebooks
* [0-prepare-datatable.pdf](notebooks/pdf/0_prepare_datatable.pdf):
  Prepare datatable for other experiments.
* [0-generate-mean-faces.pdf](notebooks/pdf/1a_generate_mean_faces.pdf):
  Mean faces of BFW w.r.t. different subgroups.
* [1-compare-features.pdf](notebooks/pdf/1_compare_features.pdf):
  Determine scores of feature pairss.
* [2-create-sdm.pdf](notebooks/pdf/2_create_sdm_curves.pdf):
  Experiments using signal detection theory.
* [3-calculate-det-curves.pdf](notebooks/pdf/3_calculate_and_display_det_curves.pdf):
  DET curves using various subsets of subgroups.
* [4-create-confusion.pdf](notebooks/pdf/4_det_curve_multinet_analysis.pdf):
  Experiment in (3) done for different models (i.e., VGG16, ResNet50, SENet50)
* [5-create-confusion.pdf](notebooks/pdf/5-rank1-error-confusion-analysis.pdf):
  Rank 1 NN analysis.
* [COMING SOON]():
  Determine optimal thresholds.
* [COMING SOON]():
  Generate TAR@FAR table.
                            
                                

