# Table 4 - Once-only test results

| Model | Accuracy | TN | False alarms (FP) | Missed faults (FN) | TP |
| --- | --- | --- | --- | --- | --- |
| Logistic regression, Set R | 72.636401% | 1305 | 0 | 657 | 439 |
| Logistic regression, Set Q | 97.875885% | 1299 | 6 | 45 | 1051 |
| Linear SVM (C=0.1) | 73.594336% | 1305 | 0 | 634 | 462 |
| RBF SVM (C=100, gamma=1) | 99.583507% | 1301 | 4 | 6 | 1090 |

Confusion matrices use actual classes as rows and predicted classes as columns: [[TN, FP], [FN, TP]].
