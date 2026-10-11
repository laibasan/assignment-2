"""Build analysis, the 800-1500-word blog draft, and the README from measured results."""
import json
import re
import pandas as pd
from src.common import ROOT, RESULTS, markdown_table

REPO_URL = "https://github.com/laibasan/assignment-2"


def main():
    test = json.loads((RESULTS / "test_results.json").read_text())["models"]
    selected = json.loads((RESULTS / "svm_selected.json").read_text())
    lr = json.loads((RESULTS / "logistic_results.json").read_text())
    svm = pd.read_csv(RESULTS / "svm_validation.csv")
    table = markdown_table(["Model", "Test accuracy", "TN", "False alarms", "Missed faults", "TP"],
                           [[r["model"], f"{r['accuracy']:.2%}", r["TN"], r["false_alarms"], r["missed_faults"], r["TP"]] for r in test])
    analysis = f"""# Task 6 - Analysis

## 6.1 Why do the raw linear models perform similarly?

Set-R logistic regression achieves {test[0]['accuracy']:.4%}, while the linear SVM achieves {test[2]['accuracy']:.4%} on the prescribed test split. Both put a hyperplane in the six-dimensional standardised raw-feature space. Logistic regression minimises cross-entropy and the SVM minimises a regularised hinge objective, so their exact predictions need not match, but neither can represent a magnitude-based curved boundary. Their similarly limited scores suggest a feature-geometry limitation, not that the two optimisation procedures are identical. Large positive and large negative instantaneous currents can both belong to faults, with normal samples near the centre.

## 6.2 What do the squared-feature weight signs mean?

The current-square weights are all positive: Ia^2 = 18.89123830, Ib^2 = 4.75425591, Ic^2 = 13.95691804. The voltage-square weights are all negative: Va^2 = -6.02763643, Vb^2 = -6.46425837, Vc^2 = -6.14074546. Holding the other transformed features fixed, larger squared currents increase the log-odds of a fault, and larger squared voltages decrease them. This matches a fault's high current magnitude and voltage sag. The scaler standard deviations are positive, so standardisation preserves these derivative signs. These correlated features describe a fitted association, not a separate causal estimate for each phase; instantaneous voltage signs also reverse over an AC cycle.

## 6.3 What does the number of support vectors mean?

The chosen linear SVM uses {selected['linear']['support_vectors']} of the 7,200 training samples ({selected['linear']['support_vectors']/7200:.2%}); the chosen RBF SVM uses {selected['rbf']['support_vectors']} ({selected['rbf']['support_vectors']/7200:.2%}). Many linear-model samples are on or inside the soft margin, including wrongly classified samples, because the raw geometry is poorly suited to a hyperplane. A large support-vector count means the solution depends on many training points. For kernel prediction that also increases memory and kernel computations, although a linear model can collapse its expansion into one weight vector. It does not by itself prove overfitting or a defective model.

## 6.4 What happens as C increases?

The linear validation accuracy stays at 72.583333% for every tested C; support-vector counts are 5,849, 5,836, 5,835 and 5,836. Thus increasing C does not fix the linear geometry, and the count is not strictly monotonic. For RBF gamma=0.1, validation accuracy rises from 95.00% to 99.541667% and the count falls from 1,347 to 174. For gamma=1, accuracy rises from 98.041667% to 99.666667% and the count falls from 848 to 266. Larger C penalises training margin violations more heavily, allowing a less regularised boundary. These trends describe this grid, not a theorem that accuracy always improves or support counts always fall. Ties in validation accuracy are resolved by smaller C and then smaller gamma, so the linear choice is C=0.1.

## 6.5 Is the test score optimistic for a real relay?

Likely optimistic. Rows are consecutive simulated time samples and adjacent waveform values are strongly related. Randomly distributing them can put near neighbours and parts of the same fault event in both training and test sets. There is no scaling or label leakage in this pipeline, but there is a dependence problem: a new physical event, line, fault resistance, loading condition or sensor noise distribution could be harder. A stronger follow-up would split by complete simulation runs or events, fit the scaler only on those training runs and assess separate unseen scenarios. The prescribed split is retained here so the assignment remains comparable.

## 6.6 Which model would be the deployment candidate?

Among these four fixed models, the RBF SVM is the best candidate for further relay validation: it misses {test[3]['missed_faults']} faults and raises {test[3]['false_alarms']} false alarms, versus {test[1]['missed_faults']} and {test[1]['false_alarms']} for Set Q. It improves both error counts on this test set; its {selected['rbf']['support_vectors']} kernel terms would need a measured latency budget. Raw logistic regression and the linear SVM miss {test[0]['missed_faults']} and {test[2]['missed_faults']} faults respectively, which is an unacceptable trade-off for this experiment despite zero false alarms. Squared-feature logistic regression is much cheaper to compute and easy to inspect, but has more dangerous misses here. Actual deployment would require independent event-level validation, hardware timing, robustness checks and coordination with existing protection logic; the test set must not be reused to tune thresholds.

## Measured comparison

{table}

Every confusion matrix sums to 2,401. Hyperparameters were fixed using training/validation data before the single test evaluation. Full-precision values are in the CSV files; the discussion rounds values only for readability.
"""
    (RESULTS / "analysis.md").write_text(analysis)
    blog = f"""# Detecting transmission-line faults: fix the features or change the classifier?

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

For each feature set, learning rates 0.01, 0.1, 1, and 10 run for 2,000 iterations. The largest rate with strictly decreasing cost is selected, including the change from the initial zero-weight cost. Set R selects alpha=1; alpha=10 is unstable. Set Q selects alpha=10. Their final costs are {lr['R']['final_cost']:.6f} and {lr['Q']['final_cost']:.6f}, with validation accuracies {lr['R']['validation_accuracy']:.2%} and {lr['Q']['validation_accuracy']:.2%}. Both final models are then independently fitted from zero with their chosen rates.

## Compare the kernels using validation only

The SVM experiment uses Set R throughout. A linear SVC tests C values 0.1, 1, 10, and 100. The RBF SVC tests the same C values with gamma 0.1 and 1, giving eight kernel configurations. Accuracy is calculated directly; scikit-learn supplies only SVC. The best validation settings are linear C=0.1 and RBF C=100, gamma=1. Equal scores are resolved in favour of smaller C and then smaller gamma.

Linear validation accuracy stays at 72.58% across the C grid. Increasing C therefore cannot repair the missing nonlinear geometry. For RBF gamma=0.1, accuracy rises from 95.00% to 99.54%; for gamma=1, from 98.04% to 99.67%. Support counts decrease across each RBF sequence. Larger C makes margin violations more costly and reduces regularisation, but improving accuracy and decreasing support counts are observations here, not universal guarantees.

## Open the test set once

Only after all choices are fixed are the four final models evaluated on the test rows. The evaluation is cached with fingerprints of the data and models, so rerunning the pipeline does not silently reuse the holdout to select a new setting.

{table}

The raw logistic model and linear SVM have similar scores because both learn hyperplanes, even though their loss functions differ. Their hundreds of missed faults point to a representation problem. Set Q substantially reduces those misses by introducing a curved boundary in raw measurement space. The RBF model improves further, with six missed faults and four false alarms.

## Interpreting the models

All three squared-current weights are positive: approximately 18.89, 4.75, and 13.96. All three squared-voltage weights are negative: approximately -6.03, -6.46, and -6.14. Larger current magnitudes raise the fitted fault score, while larger voltage magnitudes lower it, consistent with current surges and voltage sag. Because phase measurements are correlated, these weights should not be treated as independent causal effects.

The selected linear SVM needs 5,849 support vectors, compared with 266 for the RBF model. Many points constrain the inadequate linear margin or lie inside it. A large support count means many training observations influence the solution, and kernel inference becomes more expensive. A linear SVM can instead compress its expansion into a single weight vector, so counts alone do not determine runtime or prove overfitting.

## A relay needs more than accuracy

The RBF SVM is the strongest candidate for further testing because it improves both missed faults and false alarms compared with Set Q. The squared-feature logistic model is simpler and cheaper, but misses 45 faults here. A real decision would also consider worst-case latency, sensor noise and coordination with other protection functions; this simulated comparison does not certify a relay.

The random split probably makes the scores optimistic for unseen physical events. Consecutive samples can send nearly identical neighbours, or parts of the same fault event, to different sets. Training-only scaling prevents preprocessing leakage but does not remove that dependence. A follow-up should hold out complete events or simulation runs and test new operating conditions. Thresholds must not be tuned against this already examined test set.

The accompanying PhaseGuard interface compares the saved Set-Q logistic model and RBF SVM side by side. It accepts all six measurements, displays the logistic probability and signed SVM margin, and clearly distinguishes those scores. Browser calculations use exported trained coefficients and support vectors, verified against Python on validation samples. The main lesson is practical: represent current magnitude before assuming that a more complicated classifier is the only solution.

**Source:** [Electrical Fault Detection and Classification, E. Sathya Prakash](https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification). **Code:** [{REPO_URL}]({REPO_URL}).
"""
    blog += "\n## Complete experiment tables\n\nThe held-out comparison above is Table 4. The remaining required tables are reproduced here.\n\n"
    blog += "\n\n".join((RESULTS / file).read_text() for file in [
        "table_1_data.md", "table_2_logistic.md", "table_3_svm.md", "table_5_theta_Q.md"])
    communication = ROOT / "communication"; communication.mkdir(exist_ok=True)
    (communication / "medium_blog.md").write_text(blog)
    words = len(re.findall(r"\b[\w'-]+\b", blog))
    if not 800 <= words <= 1500:
        raise ValueError(f"Blog has {words} words, outside 800-1500")
    (communication / "linkedin_post.md").write_text(f"""# LinkedIn post draft

AI-assisted draft. Review and personalise before posting. Publication is pending your Medium URL.

Can a straight-line classifier spot a fault in a three-phase transmission line?

An Applied Machine Learning experiment compares raw-feature logistic regression, squared-feature logistic regression, and linear/RBF support vector machines on simulated current and voltage samples.

The key result: accounting for magnitude matters. Squared features raised test accuracy from {test[0]['accuracy']:.2%} to {test[1]['accuracy']:.2%}. The selected RBF SVM reached {test[3]['accuracy']:.2%}, with {test[3]['missed_faults']} missed faults and {test[3]['false_alarms']} false alarms.

The project includes NumPy gradient descent, validation-only hyperparameter selection, confusion matrices, and a prediction interface. It also explains why a random split of neighbouring waveform samples can produce optimistic scores.

Read the blog: [insert the actual published Medium URL]
Explore the code: {REPO_URL}

#ElectricalEngineering #MachineLearning #PowerSystems #FaultDetection #NumPy
""")
    readme = f"""# PhaseGuard - Three-phase transmission-line fault detection

AI4003 Applied Machine Learning, Assignment 02, Fall 2026. A reproducible experiment on whether nonlinear features or a kernel are needed to classify instantaneous three-phase measurements.

![PhaseGuard prediction interface](results/figures/frontend_desktop.png)

## Measured results

{table}

Dataset: 12,001 rows. Split: 7,200 / 2,400 / 2,401 using exactly NumPy's seed-42 permutation. Test evaluation count: one. Missed faults are actual faults predicted as normal. Read [the full analysis](results/analysis.md) and [all five required tables](results/tables.md).

## Quick start

Python 3.12 was used. From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python app/serve.py
```

Open **http://127.0.0.1:8000**. Saved models and their browser export are already included. No training is needed to try the interface. You can also open `app/index.html` directly because the model export is loaded as a local script.

Enter Ia, Ib, Ic (A) and Va, Vb, Vc (per unit), then select **Analyse snapshot**. Two training-sample buttons provide labelled examples. The Set-Q probability uses threshold 0.5; the RBF output is a signed decision margin, not a probability. Computation stays in the browser and uses the actual saved-model parameters.

## Reproduce the experiment

```bash
python src/load_data.py
python run_all.py
python -m unittest discover -s tests -v
```

`run_all.py` preserves an existing experiment and verifies its test fingerprints. To train a completely fresh reproduction, copy or archive `results/`, remove the old directory, and run `python run_all.py`; this starts a new documented experiment, never an automatic second test evaluation. Avoid changing settings after examining the test results.

For a fresh experiment, the individual task commands are:

```bash
python -m src.prepare
python -m src.eda
python -m src.train_logistic
python -m src.train_svm
python -m src.test_models
python -m src.export_web
python -m src.report
```

Use these from the root and in this order. The C=100 linear SVM is the slowest grid fit. For small matrices, setting OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 avoids oversubscription; `run_all.py` sets those before importing NumPy. Hyperparameter ties choose smaller C, then smaller gamma. Population standard deviation uses ddof=0, and square features are created from raw values before scaling.

## Repository layout and task coverage

| Task | Implementation | Evidence |
| --- | --- | --- |
| 1 - Load, clean, split, scale | `src/load_data.py`, `src/prepare.py` | `results/raw_preview.txt`, `table_1_data.md`, `scaling_statistics.csv` |
| 2 - Exploratory plots | `src/eda.py` | `results/eda.md`, three raw-data figures |
| 3 - Logistic regression by hand | `src/logistic.py`, `src/train_logistic.py` | rate-trial CSVs, loss figures, theta, saved models |
| 4 - Linear/RBF SVM grid | `src/train_svm.py` | all 12 validation results, selected settings, saved SVCs |
| 5 - Test once, manual metrics | `src/metrics.py`, `src/test_models.py` | `test_results.json`, CSV, prediction arrays, Table 4 |
| 6 - Engineering analysis | `src/report.py` | `results/analysis.md`, all six answers |
| 7 - UI and communication | `app/`, `communication/` | screenshot, Medium draft, LinkedIn draft |

```text
data/detect_dataset.csv
src/load_data.py  prepare.py  eda.py  logistic.py  metrics.py
src/train_logistic.py  train_svm.py  test_models.py  export_web.py  report.py
results/figures/  tables.md  analysis.md  *.csv  *.json  *.joblib  *.npy  *.npz
app/index.html  style.css  ui.js  inference.js  models.js  examples.js  serve.py
communication/medium_blog.md  linkedin_post.md  submission_checklist.md
tests/test_core.py  check_web.cjs
README.md  requirements.txt  run_all.py  CONTRIBUTING.md
```

## Validation

Core tests verify the gradient numerically, sigmoid extremes, log(2) initial loss, strict cost decrease, exact split indices, train-only scaling, raw-before-square construction, metric orientation, saved artifacts and the permitted scikit-learn import. Browser parity checks compare exported predictions and margins with Python on 64 validation rows, including both classes. No browser verification evaluates the test models again. See `results/verification.json` for recorded browser checks and desktop/mobile screenshots in `results/figures/`.

The experiment uses only NumPy, pandas, matplotlib, joblib, the Python standard library, and `sklearn.svm.SVC` for ML. Browser automation is a development-time UI verification tool; it does not split, scale, train, or score the experiment.

## Optional GitHub Pages hosting

The frontend is static, including the trained RBF support-vector expansion, so GitHub Pages can host it without a Python backend. In repository **Settings → Pages**, select **GitHub Actions** as the source. Then run the **Deploy PhaseGuard interface** workflow or push an app change. Until a deployment succeeds, use the local run command above. The workflow publishes only `app/`.

## Communication and human submission steps

The [Medium draft](communication/medium_blog.md) has {words} words and includes three figures, the results table and all six analysis topics. The [LinkedIn draft](communication/linkedin_post.md) links the repository and has a clearly marked slot for your real Medium URL. Neither has been published. The assignment also requires two group members and genuine meaningful commits from both; those must be made by the actual members. See [CONTRIBUTING](CONTRIBUTING.md) and the [submission checklist](communication/submission_checklist.md).

**Authorship notice:** This repository was prepared with AI assistance at the user's request. The assignment manual prohibits directly submitting AI-generated work and awards marks through a viva. Review, understand and independently revise it according to your course rules; do not represent the generated implementation or drafts as unaided work. No teammate identity or contribution has been invented.

## Data attribution and limits

[E. Sathya Prakash, Electrical Fault Detection and Classification](https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification), detection CSV, retrieved 11 October 2026. Kaggle's dataset metadata lists CC0: Public Domain. The original CSV bytes and a SHA-256 checksum are included; `classData.csv` is intentionally not used.

These are consecutive simulated samples, randomly split as required. Near-neighbour dependence can inflate performance. No event-level generalisation, sensor robustness, worst-case latency or real-relay certification is established. The RBF SVM is a candidate for further evaluation, not a validated protection system.
"""
    (ROOT / "README.md").write_text(readme)
    (RESULTS / "tables.md").write_text("# Required results tables\n\n" + "\n\n".join(
        (RESULTS / f).read_text() for f in ["table_1_data.md", "table_2_logistic.md", "table_3_svm.md", "table_4_test.md", "table_5_theta_Q.md"]))
    (communication / "submission_checklist.md").write_text(f"""# Submission status

| Requirement | Status |
| --- | --- |
| Data preparation, plots and scratch logistic regression | Implemented and executed |
| All SVM validation combinations and four final test results | Implemented and executed |
| Saved models, learned theta and five result tables | Included |
| Six analysis answers | Included |
| Working six-input interface and README screenshot | Included; verification recorded in results/verification.json |
| 800-1500-word Medium blog with at least two figures | Draft ready ({words} words); review, rewrite and publish from your account |
| LinkedIn post linking blog and repository | Draft ready; insert published Medium URL, review and publish |
| Public repository | {REPO_URL} |
| Exactly two members; meaningful commits by both | Actual member identities/contributions still required |
| Repo/blog/post links submitted to instructor | Your submission portal action still required |
| Optional live website | Pages workflow supplied; GitHub Pages settings must be enabled |

Do not fabricate publication URLs, commit authors, teamwork, or unaided authorship. The original manual prohibits direct AI-generated submission. Use this material to understand and independently revise the experiment before the viva.
""")
    print(f"Analysis, tables, README and {words}-word blog draft generated.")


if __name__ == "__main__":
    main()
