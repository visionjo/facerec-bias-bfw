# Data and related files
### Balanced Faces _in the Wild_ (BFW)

BFW, proposed in [<span style="color:blue">**1**</span>], provides balance in data (i.e., different subgroups) for face verification. Specifically, Table I compares BFW to related datasets (i.e., bias in faces), Table II characterizes BFW highlighting relevant stats, and Figure 1 shows a sample montage for each of the eight subgroups in BFW.

<img src=../docs/table1.png alt="table1" width="600"/>

**Subgroups of BFW.** Each row depicts a different gender, Female (F) (top) and Male (M) (bottom). Columns are grouped by ethnicity (i.e., Asian (A), Black (B), Indian (I), and White (W), respectfully).

<img src=../docs/facemontage.png alt="facemontage.png" width="600"/>

<img src=../docs/table2.png alt="table2" width="600"/>


### Data Files
This folder contains the raw data files used in the paper:

* `bfw-<version>-datatable.pkl`: List of pairs with corresponding tags for class labels (1/0), subgroups, and scores.
Download link: ADD URL

### References
[1] Joseph P. Robinson, Yann Henon, Gennady Livitz, Can Qin, Yun Fu, and Samson Timoner. Face Recognition: Too Bias, or Not Too Bias? _CoRR, abs/2002.06483_, 2020.
