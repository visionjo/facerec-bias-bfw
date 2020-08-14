# Face Recognition: Too Bias, or Not Too Bias?
<div>
<blockquote>
     Robinson, Joseph P., Gennady Livitz, Yann Henon, Can Qin, Yun Fu, and Samson Timoner. 
     "<a href="https://arxiv.org/pdf/2002.06483.pdf">Face recognition: too bias, or not too bias?
     </a>" 
     <i>In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 
     Workshops</i>, pp. 0-1. 2020.
 </blockquote>
</div>
<div>
    <div>
      <a href="https://forms.office.com/Pages/ResponsePage.aspx?id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAAMAAMDJhXxUMElHQ0tVSDFSNDZTMVBPSVpXMkxJTkY4Ny4u">Download Data</a> 
     </div>
    <div style="display: none;" id="robinsonfacebias2020">
      <pre class="bibtex">@inproceedings{robinson2020face,
               title={Face recognition: too bias, or not too bias?},
               author={Robinson, Joseph P and Livitz, Gennady and Henon, Yann and Qin, Can and Fu, Yun and Timoner, Samson},
               booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
               pages={0--1},
               year={2020}
             }
    </pre>
  </div>
  <br>
</div>

![Teaser](docs/bfw-logo.png)


## Overview
This project investigates bias in automatic facial recognition (FR). Speccifically, subbjects are grouped into predefined subgroups that are based on gender, ethnicity, and soon to be age. For this, we prpose a novel image collection called Balanced Faces in the Wild (BFW), which is balanced across eigth subgroups (i.e., 800 face images of 100 subjects, each with 25 face samples). Thus, along with name (i.e., identification) labels and task protocols (e.g., list of pairs for face verifiication, pre-packaged data-table with additional metadata and labels, etc.), BFW clearly groups into ethnicities (i.e., Asian (A), Black (B), Indian (I), and White (W)) and genders (i.e., Females (F) and Males (M)). The motivation and, thus, the intent is that BFW will provide a proxy to characterize FR systems with demographic-specific analysis now possible. For instance, various confusion metrics, along with the pre-defined criteria (i.e., score threshold) that are fundamental when characterizing performance ratings of FR systems. The following visualizatiion summarizes the confusion metrics in a way that relates the different measurements.

![metrics](docs/metric-summary.png)

As discusssed up to this point has been the motivation of designing, building, and releasing BFW for research purposes. We expecct the data, all-in-all, will continue to evolve. Nonetheless, as is, there are vast options on ways to advance technology and our understanding thereof. Let us know focus on the contents of the repo (i.e., code-base) for which was created to support the data of BFW (i.e., data proxy), making all experiments in paper easily reproduciable and, thus, the work more friendly for getting started.

## Experimental-based contributions and findings
Several observations were made that widened our understanding of bias in FR. Obsservations were demonstrated experimentally, with all code used in experiments added as a part of thiss repo.

### Score sensitivity
For instance, it is shown that the scoring sensitivity within different subgroups verifies. That is, faces of the same identity tend to shift in expected values (e.g., given a true pair of Black faces, on average, have similarity scores smaller than a true pair of White, and thethe average range of scores for Males compared to Females). This is demonstrated using fundamental signal detection models (SDM), along with detection error trade-off (DET) curves.

### Global threshold
Once a FR system is deployed, a criteria (i.e., theshold) is set (or tunable) such that similarity scores that do not pass are assumed false matchess, and are filtered out of candidate pool for potential true pairs. In other words, thresholds act as decision boundaries that map scores (or distances) to nominal values such as *genuine* or *imposter*. Considering the variable sensitivty found prior, intuition tells us that a variable threshold is optimal. Thus, returning to the fundamental concepts of signal detection theory, we show that using a single, global threshold yields skewed performance ratings across different subgroups. For this, we demonstrate that subgroup-specific thresholds are optimal in terms of overall performance and balance across subgroups. 

### All-in-all
All of this and more (i.e., evaluation and analysis of FR systems on BFW data, along with data structures and implementation schemes that optimized for the problems at hand are included in modules making up the project and demonstrated in notebook tutorials). We will continue to add tools for a fair analysis of FR systems. Thus, not only the experiments, but also the data we expect to grow. All contributions not only welcome, but are entirely encouraged.


Here are quick links to key aspects of this resource.

* Check out research paper, [https://arxiv.org/pdf/2002.06483.pdf](https://arxiv.org/pdf/2002.06483.pdf)
* See [data/README.md](data/README.md) for more on BFW.
* See [code/README.md](code/README.md) for more on 'facebias' package and experiments contained within.
* See [results/README.md](results/README.md) for summary of figures and results.

Register and download via this <a href="https://forms.office.com/Pages/ResponsePage.aspx?id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAAMAAMDJhXxUMElHQ0tVSDFSNDZTMVBPSVpXMkxJTkY4Ny4u">form</a>.

**Final note.** Thee repo is a work-in-progress. Certainly, it is ready to be cloned and used; however, expect regular improvements, both in the implementation and documentation (i.e., *getting started* instructions will be enhanced). For now, it is recommended to start with README files listed just above, along with the tutorial notebooks found in `code->notebooks`, with brief descriptions in README and more detail inline of each notebook. Again, PRs are more than welocme :)

## Paper abstract
We reveal critical insights into problems of bias in state-of-the-art facial recognition (FR) systems using a novel Balanced Faces In the Wild (BFW) dataset: data balanced for gender and ethnic groups. We show variations in the optimal scoring threshold for face-pairs across different subgroups. Thus, the conventional approach of learning a global threshold for all pairs results in performance gaps between subgroups. By learning subgroup-specific thresholds, we reduce performance gaps, and also show a notable boost in overall performance. Furthermore, we do a human evaluation to measure bias in humans, which supports the hypothesis that an analogous bias exists in human perception. For the BFW database, source code, and more, visit <a href="https://github.com/visionjo/facerec-bias-bfw">https://github.com/visionjo/facerec-bias-bfw</a>.

## Software implementation
All source code used to generate the results and figures in the paper are in the `code` folder. The calculations and figure generation are all run inside [Jupyter notebooks](http://jupyter.org/) which are store [code/notebooks](code/notebooks). The data used in this study should be downloaded via completed the registration form and saved in the `data` directory, while the sources for the manuscript text and figures are in `manuscript`. Results generated by the code are saved in `results`. See the `README.md` files in each directory for a full description (READMEs also listed above).

## The data

Most processes and experiments depend on a <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html">pandas dataframe</a>. A [demo notebook](code/notebooks/pdf/0_prepare_datatable.pdf) is provided to show the steps of populating the data structure included in the data download in a csv and pickle file. Furthermore, documentation in [data/README](data/README.md) summarize the details about data as described in the [paper](https://arxiv.org/pdf/2002.06483.pdf) and used in the [notebooks](https://github.com/visionjo/facerec-bias-bfw/blob/master/code/README.md#notebooks).
  

## Getting the code

You can download a copy of all the files in this repository by cloning the [git](https://git-scm.com/) repository:

    git clone https://github.com/visionjo/facerec-bias-bfw.git

or [download a zip archive](https://github.com/visionjo/facerec-bias-bfw/archive/master.zip).


## Dependencies

You'll need a working Python environment to run the code. The recommended way to set up your environment is through the [Anaconda Python distribution](https://www.anaconda.com/download/) which provides the `conda` package manager. Anaconda can be installed in your user directory and does not interfere with the system Python installation. The required dependencies are specified in the file `environment.yml`.

We use `conda` virtual environments to manage the project dependencies in isolation. Thus, you can install our dependencies without causing conflicts with your setup (even with different Python versions).

Run the following command in the repository folder (where `environment.yml` is located) to create a separate environment and install all required dependencies in it:

    conda env create


## Reproducing the results

Before running any code you must activate the conda environment:

    source activate ENVIRONMENT_NAME

or, if you're on Windows:

    activate ENVIRONMENT_NAME

This will enable the environment for your current terminal session. Any subsequent commands will use software that is installed in the environment.

To build and test the software, produce all results and figures, and compile the manuscript PDF, run this in the top level of the repository:

    make all

If all goes well, the manuscript PDF will be placed in `manuscript/output`.

You can also run individual steps in the process using the `Makefile`s from the `code` and `manuscript` folders. See the respective `README.md` files for instructions.

Another way of exploring the code results is to execute the Jupyter notebooks individually. To do this, you must first start the notebook server by going into the repository top level and running:

    jupyter notebook

This will start the server and open your default web browser to the Jupyter interface. In the page, go into the `code/notebooks` folder and select the notebook that you wish to view/run.

The notebook is divided into cells (some have text while other have code). Each cell can be executed using `Shift + Enter`.
Executing text cells does nothing and executing code cells runs the code and produces it's output. To execute the whole notebook, run all cells in order.

## To Do
- [x] Begin Template
- [x] Create demo notebooks
- [x] Add manuscript
- [ ] Documentation (sphinx)
- [ ] Update README (this)
- [x] Pre-commit, formatter (Black) and .gitignore
- [ ] Complete test harness
- [x] Modulate (refactor) code
- [ ] Complete datatable (i.e., extend pandas.DataFrame)
- [ ] Add scripts and CLI

## License

All source code is made available under a BSD 3-clause license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See `LICENSE.md` ([LICENSE](LICENSE.md)) for the full license text.

The manuscript text is not open source. The authors reserve the rights to the article content, which is currently submitted for publication in the 2020 IEEE Conference on AMFG.

## Acknowledgement
We would like to thank the [PINGA](https://github.com/pinga-lab?type=source) organization on Github for the [project template](https://github.com/pinga-lab/paper-template) used to structure this project.
