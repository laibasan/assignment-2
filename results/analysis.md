# Task 6 - Analysis

## 6.1 Why do the raw linear models perform similarly?

Set-R logistic regression achieves 72.6364%, while the linear SVM achieves 73.5943% on the prescribed test split. Both put a hyperplane in the six-dimensional standardised raw-feature space. Logistic regression minimises cross-entropy and the SVM minimises a regularised hinge objective, so their exact predictions need not match, but neither can represent a magnitude-based curved boundary. Their similarly limited scores suggest a feature-geometry limitation, not that the two optimisation procedures are identical. Large positive and large negative instantaneous currents can both belong to faults, with normal samples near the centre.

## 6.2 What do the squared-feature weight signs mean?

The current-square weights are all positive: Ia^2 = 18.89123830, Ib^2 = 4.75425591, Ic^2 = 13.95691804. The voltage-square weights are all negative: Va^2 = -6.02763643, Vb^2 = -6.46425837, Vc^2 = -6.14074546. Holding the other transformed features fixed, larger squared currents increase the log-odds of a fault, and larger squared voltages decrease them. This matches a fault's high current magnitude and voltage sag. The scaler standard deviations are positive, so standardisation preserves these derivative signs. These correlated features describe a fitted association, not a separate causal estimate for each phase; instantaneous voltage signs also reverse over an AC cycle.

## 6.3 What does the number of support vectors mean?

The chosen linear SVM uses 5849 of the 7,200 training samples (81.24%); the chosen RBF SVM uses 266 (3.69%). Many linear-model samples are on or inside the soft margin, including wrongly classified samples, because the raw geometry is poorly suited to a hyperplane. A large support-vector count means the solution depends on many training points. For kernel prediction that also increases memory and kernel computations, although a linear model can collapse its expansion into one weight vector. It does not by itself prove overfitting or a defective model.

## 6.4 What happens as C increases?

The linear validation accuracy stays at 72.583333% for every tested C; support-vector counts are 5,849, 5,836, 5,835 and 5,836. Thus increasing C does not fix the linear geometry, and the count is not strictly monotonic. For RBF gamma=0.1, validation accuracy rises from 95.00% to 99.541667% and the count falls from 1,347 to 174. For gamma=1, accuracy rises from 98.041667% to 99.666667% and the count falls from 848 to 266. Larger C penalises training margin violations more heavily, allowing a less regularised boundary. These trends describe this grid, not a theorem that accuracy always improves or support counts always fall. Ties in validation accuracy are resolved by smaller C and then smaller gamma, so the linear choice is C=0.1.

## 6.5 Is the test score optimistic for a real relay?

Likely optimistic. Rows are consecutive simulated time samples and adjacent waveform values are strongly related. Randomly distributing them can put near neighbours and parts of the same fault event in both training and test sets. There is no scaling or label leakage in this pipeline, but there is a dependence problem: a new physical event, line, fault resistance, loading condition or sensor noise distribution could be harder. A stronger follow-up would split by complete simulation runs or events, fit the scaler only on those training runs and assess separate unseen scenarios. The prescribed split is retained here so the assignment remains comparable.

## 6.6 Which model would be the deployment candidate?

Among these four fixed models, the RBF SVM is the best candidate for further relay validation: it misses 6 faults and raises 4 false alarms, versus 45 and 6 for Set Q. It improves both error counts on this test set; its 266 kernel terms would need a measured latency budget. Raw logistic regression and the linear SVM miss 657 and 634 faults respectively, which is an unacceptable trade-off for this experiment despite zero false alarms. Squared-feature logistic regression is much cheaper to compute and easy to inspect, but has more dangerous misses here. Actual deployment would require independent event-level validation, hardware timing, robustness checks and coordination with existing protection logic; the test set must not be reused to tune thresholds.

## Measured comparison

| Model | Test accuracy | TN | False alarms | Missed faults | TP |
| --- | --- | --- | --- | --- | --- |
| Logistic regression, Set R | 72.64% | 1305 | 0 | 657 | 439 |
| Logistic regression, Set Q | 97.88% | 1299 | 6 | 45 | 1051 |
| Linear SVM (C=0.1) | 73.59% | 1305 | 0 | 634 | 462 |
| RBF SVM (C=100, gamma=1) | 99.58% | 1301 | 4 | 6 | 1090 |

Every confusion matrix sums to 2,401. Hyperparameters were fixed using training/validation data before the single test evaluation. Full-precision values are in the CSV files; the discussion rounds values only for readability.
