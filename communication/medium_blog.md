# Detecting transmission-line faults: fix the features or change the classifier?

*AI-assisted working draft for review and rewriting before publication. No Medium publication has been performed.*

A transmission-line fault can begin when a phase touches ground or another phase. Current magnitudes rise and voltages sag, but the protective relay sees alternating signals: a fault current can be positive, negative, or close to zero at an instant. That makes the learning problem more interesting than a simple rule saying that a large positive reading means trouble. This experiment asks whether a straight decision boundary can identify faults from six instantaneous measurements, and compares two remedies when it cannot: squared features and a nonlinear kernel.

The input contains three currents in amperes and three voltages in per unit; the output is fault or no fault. E. Sathya Prakash's Kaggle dataset was simulated using MATLAB/Simulink. Its detection CSV has 12,001 rows. Two empty columns are removed and Output (S) becomes fault; the data table below records the preparation details.

## Keep the experiment fair

The prescribed seed-42 NumPy permutation produces 7,200 training, 2,400 validation and 2,401 test rows. Training means and population standard deviations scale every split; using evaluation rows for those statistics would leak information into model fitting.

Set R contains six raw features; Set Q adds their squares. Squares are computed before standardisation, then all twelve columns use training statistics. Squaring an already centred measurement creates a different feature. Neither matrix contains the label.

## The plots explain the problem

![Raw currents with fault intervals shaded](../results/figures/currents_first_2000.png)

The first 2,000 consecutive samples show oscillating phase currents and larger excursions during labelled fault intervals. The shading comes from the fault labels, not an estimated threshold. Some fault samples remain near zero because instantaneous readings pass through waveform zero crossings.

![Ia versus Ib, coloured by fault](../results/figures/ia_ib_scatter.png)

The scatter plot puts many normal samples near the centre, while fault samples extend in different directions. A straight line divides the plane into two half-planes, so it cannot surround a central normal region. This is a useful visual explanation, though the six-dimensional classifier also receives voltage information.

![Ia histogram by class](../results/figures/ia_histogram.png)

The histogram uses the same bins for both labels. Normal samples concentrate at relatively small current magnitudes, while fault rows produce broad positive and negative tails. The overlap warns against relying on one signed measurement. Squared currents preserve both tails as evidence of high magnitude, regardless of sign.

## Logistic regression from the equations

NumPy implements the sigmoid and batch gradient descent, with an explicit intercept column. Binary cross-entropy clips probabilities to [10^-12, 1 - 10^-12] before taking logarithms. Zero weights give initial cost log(2), approximately 0.693147. Coefficients update simultaneously on all training rows, and prediction uses threshold 0.5.

For each feature set, learning rates 0.01, 0.1, 1, and 10 run for 2,000 iterations. The largest rate with strictly decreasing cost is selected, including the change from the initial zero-weight cost. Set R selects alpha=1; alpha=10 is unstable. Set Q selects alpha=10. Their final costs are 0.677161 and 0.053434, with validation accuracies 73.08% and 97.83%. Both final models are then independently fitted from zero with their chosen rates.

## Compare the kernels using validation only

The SVM experiment uses Set R throughout. A linear SVC tests C values 0.1, 1, 10, and 100. The RBF SVC tests the same C values with gamma 0.1 and 1, giving eight kernel configurations. Accuracy is calculated directly; scikit-learn supplies only SVC. The best validation settings are linear C=0.1 and RBF C=100, gamma=1. Equal scores are resolved in favour of smaller C and then smaller gamma.

Linear validation accuracy stays at 72.58% across the C grid. Increasing C therefore cannot repair the missing nonlinear geometry. For RBF gamma=0.1, accuracy rises from 95.00% to 99.54%; for gamma=1, from 98.04% to 99.67%. Support counts decrease across each RBF sequence. Larger C makes margin violations more costly and reduces regularisation, but improving accuracy and decreasing support counts are observations here, not universal guarantees.

## Open the test set once

Only after all choices are fixed are the four final models evaluated on the test rows. The evaluation is cached with fingerprints of the data and models, so rerunning the pipeline does not silently reuse the holdout to select a new setting.

| Model | Test accuracy | TN | False alarms | Missed faults | TP |
| --- | --- | --- | --- | --- | --- |
| Logistic regression, Set R | 72.64% | 1305 | 0 | 657 | 439 |
| Logistic regression, Set Q | 97.88% | 1299 | 6 | 45 | 1051 |
| Linear SVM (C=0.1) | 73.59% | 1305 | 0 | 634 | 462 |
| RBF SVM (C=100, gamma=1) | 99.58% | 1301 | 4 | 6 | 1090 |

The raw logistic model and linear SVM have similar scores because both learn hyperplanes, even though their loss functions differ. Their hundreds of missed faults point to a representation problem. Set Q substantially reduces those misses by introducing a curved boundary in raw measurement space. The RBF model improves further, with six missed faults and four false alarms.

## Interpreting the models

All three squared-current weights are positive: approximately 18.89, 4.75, and 13.96. All three squared-voltage weights are negative: approximately -6.03, -6.46, and -6.14. Larger current magnitudes raise the fitted fault score, while larger voltage magnitudes lower it, consistent with current surges and voltage sag. Because phase measurements are correlated, these weights should not be treated as independent causal effects.

The selected linear SVM needs 5,849 support vectors, compared with 266 for the RBF model. Many points constrain the inadequate linear margin or lie inside it. A large support count means many training observations influence the solution, and kernel inference becomes more expensive. A linear SVM can instead compress its expansion into a single weight vector, so counts alone do not determine runtime or prove overfitting.

## A relay needs more than accuracy

The RBF SVM is the strongest candidate for further testing because it improves both missed faults and false alarms compared with Set Q. The squared-feature logistic model is simpler and cheaper, but misses 45 faults here. A real decision would also consider worst-case latency, sensor noise and coordination with other protection functions; this simulated comparison does not certify a relay.

The random split probably makes the scores optimistic for unseen physical events. Consecutive samples can send nearly identical neighbours, or parts of the same fault event, to different sets. Training-only scaling prevents preprocessing leakage but does not remove that dependence. A follow-up should hold out complete events or simulation runs and test new operating conditions. Thresholds must not be tuned against this already examined test set.

The accompanying PhaseGuard interface compares the saved Set-Q logistic model and RBF SVM side by side. It accepts all six measurements, displays the logistic probability and signed SVM margin, and clearly distinguishes those scores. Browser calculations use exported trained coefficients and support vectors, verified against Python on validation samples. The main lesson is practical: represent current magnitude before assuming that a more complicated classifier is the only solution.

**Source:** [Electrical Fault Detection and Classification, E. Sathya Prakash](https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification). **Code:** [https://github.com/laibasan/assignment-2](https://github.com/laibasan/assignment-2).

## Complete experiment tables

The held-out comparison above is Table 4. The remaining required tables are reproduced here.

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
