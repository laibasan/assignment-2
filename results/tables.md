# Required results tables

# Table 1 - Data preparation

| Item | Value |
| --- | --- |
| Raw rows / raw columns | 12001 / 9 |
| Columns removed | Unnamed: 7, Unnamed: 8 |
| Missing values after cleaning | fault: 0, Ia: 0, Ib: 0, Ic: 0, Va: 0, Vb: 0, Vc: 0 |
| Fault rows / no-fault rows | 5496 / 6505 |
| Train / validation / test rows | 7200 / 2400 / 2401 |
| Fault fractions (train / validation / test) | 0.460694 / 0.451250 / 0.456476 |


# Table 2 - Logistic regression

| Feature set | Chosen alpha | Final J | Training accuracy | Validation accuracy |
| --- | --- | --- | --- | --- |
| R | 1.0 | 0.67716117 | 72.583333% | 73.083333% |
| Q | 10.0 | 0.05343427 | 97.875000% | 97.833333% |


# Table 3 - SVM validation

| Kernel | C | gamma | Validation accuracy | Support vectors |
| --- | --- | --- | --- | --- |
| linear | 0.1 | - | 72.583333% | 5849 |
| linear | 1 | - | 72.583333% | 5836 |
| linear | 10 | - | 72.583333% | 5835 |
| linear | 100 | - | 72.583333% | 5836 |
| rbf | 0.1 | 0.1 | 95.000000% | 1347 |
| rbf | 0.1 | 1 | 98.041667% | 848 |
| rbf | 1 | 0.1 | 97.500000% | 723 |
| rbf | 1 | 1 | 99.166667% | 482 |
| rbf | 10 | 0.1 | 99.208333% | 381 |
| rbf | 10 | 1 | 99.625000% | 342 |
| rbf | 100 | 0.1 | 99.541667% | 174 |
| rbf | 100 | 1 | 99.666667% | 266 |


# Table 4 - Once-only test results

| Model | Accuracy | TN | False alarms (FP) | Missed faults (FN) | TP |
| --- | --- | --- | --- | --- | --- |
| Logistic regression, Set R | 72.636401% | 1305 | 0 | 657 | 439 |
| Logistic regression, Set Q | 97.875885% | 1299 | 6 | 45 | 1051 |
| Linear SVM (C=0.1) | 73.594336% | 1305 | 0 | 634 | 462 |
| RBF SVM (C=100, gamma=1) | 99.583507% | 1301 | 4 | 6 | 1090 |

Confusion matrices use actual classes as rows and predicted classes as columns: [[TN, FP], [FN, TP]].


# Table 5 - Learned Set-Q weights

| Weight | Value |
| --- | --- |
| theta_0 (intercept) | 24.9209141224 |
| Ia | 2.3272915931 |
| Ib | -5.3580838672 |
| Ic | -1.7365031310 |
| Va | -0.5335687130 |
| Vb | 0.4167264098 |
| Vc | 0.0985796277 |
| Ia^2 | 18.8912382991 |
| Ib^2 | 4.7542559116 |
| Ic^2 | 13.9569180396 |
| Va^2 | -6.0276364341 |
| Vb^2 | -6.4642583654 |
| Vc^2 | -6.1407454567 |
