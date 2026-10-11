"""Run from the repository root: python run_all.py."""
import os
# Small dense matrices are faster and reproducible without oversubscribed BLAS.
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"
import json
from src import prepare, eda, train_logistic, train_svm, test_models
from src.common import RESULTS


if __name__ == "__main__":
    if (RESULTS / "test_results.json").exists():
        # Preserve the original experiment and check that artifacts still match.
        test_models.main()
        print("Existing experiment retained. Delete results only to start a documented new experiment.")
    else:
        prepare.main()
        eda.main()
        train_logistic.main()
        train_svm.main()
        test_models.main()
    from src import export_web, report
    export_web.main()
    report.main()
