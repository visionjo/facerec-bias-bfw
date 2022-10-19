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
      <a href="https://forms.gle/3HDBikmz36i9DnFf7">Download Data</a> 
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

<div>
<blockquote>
     Robinson, Joseph P., Can Qin, Yann Henon, Samson Timoner, and Yun Fu. 
     "<a href="https://arxiv.org/pdf/2103.09118.pdf">Balancing Biases and Preserving Privacy on
Balanced Faces in the Wild.</a>" <i>In CoRR arXiv:2103.09118</i>, (2021).
 </blockquote>
</div>
<div>
    <div style="display: none;" id="robinson2021balancing">
      <pre class="bibtex">@article{robinson2021balancing,
        title={Balancing Biases and Preserving Privacy on Balanced Faces in the Wild},
        author={Robinson, Joseph P and Qin, Can and Henon, Yann and Timoner, Samson and Fu, Yun},
        journal={arXiv preprint arXiv:2103.09118},
        year={2021}
       }
    </pre>
  </div>
  <br>
</div>


![Teaser](docs/bfw-logo.png)

## Balanced Faces in the Wild (BFW): Data, Code, Evaluations

__version__: 0.4.5 (following Semantic Versioning Scheme-- learn more here, https://semver.org)

Intended to address problems of bias in facial recognition, we built BFW as a labeled data resource made available for evaluating recognition systems on a corpus of facial imagery made-up of EQUAL face count for all subjects: EQUAL across demographics, and, thus, face data balanced in faces per subject, individuals per ethnicity, and ethnicities per gender or vise versa.


Data can be accessed via <a href="https://forms.gle/PAKbxgUxCSUbM29q9">Google form</a>. Do not hesitate to report an issue for any and all inquiries.

Also, the same data has been added to IEEEDataPort, which can be accessed at <a href="https://ieee-dataport.org/documents/balanced-faces-wild">https://ieee-dataport.org/documents/balanced-faces-wild</a>.

## Project Overview
This project investigates bias in automatic facial recognition (FR). Specifically, subjects are grouped into predefined subgroups based on gender, ethnicity, and soon-to-be age. For this, we propose a novel image collection called Balanced Faces in the Wild (BFW), which is balanced across eight subgroups (i.e., 800 face images of 100 subjects, each with 25 face samples). Thus, along with the name (i.e., identification) labels and task protocols (e.g., list of pairs for face verification, pre-packaged data-table with additional metadata and labels, etc.), BFW clearly groups into ethnicities (i.e., Asian (A), Black (B), Indian (I), and White (W)) and genders (i.e., Females (F) and Males (M)). Thus, the motivation and intent are that BFW will provide a proxy to characterize FR systems with demographic-specific analysis now possible. For instance, various confusion metrics, along with the predefined criteria (i.e., score threshold), are fundamental when characterizing performance ratings of FR systems. The following visualization summarizes the confusion metrics in a way that relates to the different measurements.

![metrics](docs/metric-summary.png)

As discussed, the motivation for designing, building, and releasing BFW for research purposes has been discussed. We expect the data, all-in-all, will continue to evolve. Nonetheless, as is, there are vast options on ways to advance technology and our understanding thereof. Let us now focus on the contents of the repo (i.e., code-base) for which was created to support the data of BFW (i.e., data proxy), making all experiments in paper easily reproducible and, thus, the work more friendly for getting started.

## Experimental-based contributions and findings
Several observations were made that widened our understanding of bias in FR. Views were demonstrated experimentally, with all code used in experiments added as a part of this repo.

### Score sensitivity
For instance, it is shown that the scoring sensitivity within different subgroups verifies. That is, faces of the same identity tend to shift in expected values (e.g., given a correct pair of Black faces, on average, have similarity scores smaller than a true pair of White, and the middle range of scores for Males compared to Females). This is demonstrated using fundamental signal detection models (SDM), along with detection error trade-off (DET) curves.

### Global threshold
Once an FR system is deployed, a criterion (i.e., threshold) is set (or tunable) such that similarity scores that do not pass are assumed false matches and are filtered out of the candidate pool for potential true pairs. In other words, thresholds act as decision boundaries that map scores (or distances) to nominal values such as *genuine* or *imposter*. Considering the variable sensitivity found prior, intuition tells us that a variable threshold is optimal. Thus, returning to the fundamental concepts of signal detection theory, we show that using a single, global threshold yields skewed performance ratings across different subgroups. For this, we demonstrate that subgroup-specific thresholds are optimal in terms of overall performance and balance across subgroups. 

### All-in-all
All of this and more (i.e., evaluation and analysis of FR systems on BFW data, along with data structures and implementation schemes optimized for the problems at hand, are included in modules making up the project and demonstrated in notebook tutorials). We will continue to add tools for a fair analysis of FR systems. Thus, not only the experiments but also the data we expect to grow. All contributions are not only welcome but are entirely encouraged.


Here are quick links to key aspects of this resource.

* Check out research paper, [https://arxiv.org/pdf/2002.06483.pdf](https://arxiv.org/pdf/2002.06483.pdf)
* See [data/README.md](data/README.md) for more on BFW.
* See [code/README.md](code/README.md) for more on 'facebias' package and experiments contained within.
* See [results/README.md](results/README.md) for summary of figures and results.

Register and download via this <a href="https://docs.google.com/forms/d/e/1FAIpQLSdZQLrKJnX5ZuE2oBa9FH_W9dm_wiptAHXTGyQ0-CjW-KLjUA/viewform">form</a>.

**Final note.** Thee repo is a work-in-progress. Certainly, it is ready to be cloned and used; however, expect regular improvements, both in the implementation and documentation (i.e., *getting started* instructions will be enhanced). For now, it is recommended to begin with README files listed just above, along with the tutorial notebooks found in `code-> notebooks` with brief descriptions in README and more detail inline of each notebook. Again, PRs are more than welcome :)

## Paper abstract
We reveal critical insights into bias problems in state-of-the-art facial recognition (FR) systems using a novel Balanced Faces In the Wild (BFW) dataset: data balanced for gender and ethnic groups. We show variations in the optimal scoring threshold for face pairs across different subgroups. Thus, the conventional approach of learning a global threshold for all pairs results in performance gaps between subgroups. By learning subgroup-specific thresholds, we reduce performance gaps and show a notable boost in overall performance. Furthermore, we do a human evaluation to measure human bias, which supports the hypothesis that an analogous bias exists in human perception. For the BFW database, source code, and more, visit <a href="https://github.com/visionjo/facerec-bias-bfw">https://github.com/visionjo/facerec-bias-bfw</a>.


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

All source code is made available under a BSD 3-clause license. You can freely use and modify the code without warranty, so long as you provide attribution to the authors. See `LICENSE.md` ([LICENSE](LICENSE.md)) for the full license text.

The manuscript text is not open source. The authors reserve the rights to the article content, which is currently submitted for publication in the 2020 IEEE Conference on AMFG.

## Acknowledgement
We would like to thank the [PINGA](https://github.com/pinga-lab?type=source) organization on Github for the [project template](https://github.com/pinga-lab/paper-template) used to structure this project.
