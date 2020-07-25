# Data and related files
### Balanced Faces _in the Wild_ (BFW)

BFW, proposed in [<span style="color:blue">**1**</span>], provides balance in data (i.e., different subgroups) for face verification. Specifically, Table I compares BFW to related datasets (i.e., bias in faces), Table II characterizes BFW highlighting relevant stats, and Figure 1 shows a sample montage for each of the eight subgroups in BFW.

<p>
  <img src=../docs/facemontage.jpg alt="facemontage.png" width="700"/>

  **Fig 1. Subgroups of BFW.** Rows depict different genders, Female (top) and Male (bottom). Columns are grouped by ethnicity (i.e., Asian, Black, Indian, and White, respectfully).
</p>
  
<p>
  
**Table 1. Database stats and nomenclature.** Header: Subgroup definitions. Top: Statistics of Balanced Faces in the Wild (BFW). Bottom: Number of pairs for each partition. Columns grouped by ethnicity and then further split by gender.
<br>

<img src=../docs/table1.png alt="table1" width="700"/>
</p>

<p>
  
**Table 2. BFW and related datasets.** BFW is balanced across ID, gender, and ethnicity (Table 1). Compared with DemogPairs, BFW provides more samples per subject and subgroups per set, while using a single resource, VGG2. RFW, on the other hand, supports domain adaptation, and focuses on race-distribution - not the distribution of identities.

<img src=../docs/table2.png alt="table2" width="700"/>
</p>

### Data Files
This folder contains the raw data files used in the paper:

* `bfw-<version>-datatable.pkl`: List of pairs with corresponding tags for class labels (1/0), subgroups, and scores.
Download link: <a href="https://forms.office.com/Pages/ResponsePage.aspx?id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAAMAAMDJhXxUMElHQ0tVSDFSNDZTMVBPSVpXMkxJTkY4Ny4u">form</a>.

### Data structure
Paired faces and all corresponding metadata is organized as a <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html"> pandas dataframe</a> formatted as follows.

| ID |  fold | p1  | p2  | label  | id1  | id2	| att1  | att2  | vgg16  | resnet50   | senet50   | a1   | a2   | g1   | g2 | e1   | e2   | sphereface   |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0  | 1  |  asian\_females/n000009/0010\_01.jpg | asian\_females/n000009/0043\_01.jpg | 1     | 0   | 0   | asian\_females | asian\_females | 0.820 | 0.703 | 0.679 | AF | AF | F  | F  | A  | A  | 0.393   |
| 1  | 1  | asian\_females/n000009/0010\_01.jpg | asian\_females/n000009/0120\_01.jpg | 1     | 0   | 0   | asian\_females | asian\_females | 0.719 | 0.524 | 0.594 | AF | AF | F  | F  | A  | A  | 0.354  |
| 2  | 1  |  asian\_females/n000009/0010\_01.jpg | asian\_females/n000009/0122\_02.jpg | 1     | 0   | 0   | asian\_females | asian\_females | 0.732 | 0.528 | 0.644  | AF | AF | F  | F  | A  | A  | 0.302  |
| 3 | 1    | asian\_females/n000009/0010\_01.jpg | asian\_females/n000009/0188\_01.jpg | 1     | 0   | 0   | asian\_females | asian\_females | 0.607 | 0.348 | 0.459 | AF | AF | F  | F  | A  | A  | \-0.009 |
| 4 | 1    | asian\_females/n000009/0010\_01.jpg | asian\_females/n000009/0205\_01.jpg | 1     | 0   | 0   | asian\_females | asian\_females | 0.629 | 0.384 | 0.495 | AF | AF | F  | F  | A  | A  | 0.133  |
<br>

* **ID** : index (i.e., row number) of dataframe ([0, *N*], where *N* is pair count).
* **fold** : fold number of five-fold experiment [1, 5].
* **p1**  and **p2** : relative image path of face
* **label** : ground-truth ([0, 1] for non-match and match, respectively)
* **id1** and **id2** : subject ID for faces in pair ([0, *M*], where *M* is number of unique subjects)
* **att1** and **att2** : attributee of subjects in pair.
* **vgg16**, **resnet50**, **senet50**, and **sphereface** : cosine similarity score for respective model.
* **a1** and **a2** : abbreviated attribute tag of subjects in pair [AF, AM, BF, BM, IF, IM, WF, WM].
* **g1** and **g2** : abbreviated gender tag of subjects in pair [F, M].
* **e1** and **e2** : abbreviate ethnicity tag of subjects in pair [A, B, I, W].


## Reported bugs
Here listed are the bugs in the data.<sup><a href="#fn1" id="ref1">1</a></sup> Each bug is listed per the date first reported, along with a brief description proceeding the item in parentheses. Future versions of data will incorporate bug fixes based on the following:

__24 July 2020__,
* asian_females/n002509/0139_03.jpg (incorrect identity)

__19 July 2020__
* white_females/n003391/0017_02.jpg (cartoon face)
* asian_males/n006741/0275_02.jpg (cartoon face)


### References
[1] Robinson, Joseph P., Gennady Livitz, Yann Henon, Can Qin, Yun Fu, and Samson Timoner. "<a href="https://arxiv.org/pdf/2002.06483.pdf">Face recognition: too bias, or not too bias?</a>" <i>In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops</i>, pp. 0-1. 2020.


<sup id="fn1">1. The list is for contributors to post any errors found in the data. Besides, all are welcome to report bugs by directly contacting <a href = "mailto: robinson.jo@northeastern.edu">Joseph Robinson</a>.
  <a href="#ref1" title="Footnote 1.">↩</a></sup>
