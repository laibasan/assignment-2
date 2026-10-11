/* Exact evaluation of the trained, exported Set-Q logistic and RBF SVM. */
(function (root) {
  'use strict';
  function sigmoid(z) {
    const e = Math.exp(-Math.abs(z));
    return z >= 0 ? 1 / (1 + e) : e / (1 + e);
  }
  function infer(values, models) {
    if (!Array.isArray(values) || values.length !== 6 || !values.every(Number.isFinite)) {
      throw new Error('Enter six finite measurements.');
    }
    const q = models.logistic, svm = models.rbf;
    const features = values.concat(values.map(x => x * x));
    if (!features.every(Number.isFinite)) throw new Error('Measurements are too large to square.');
    let z = q.theta[0];
    features.forEach((value, j) => { z += q.theta[j + 1] * ((value - q.mean[j]) / q.std[j]); });
    const probability = sigmoid(z);
    const scaled = values.map((value, j) => (value - svm.mean[j]) / svm.std[j]);
    let margin = svm.intercept;
    svm.support_vectors.forEach((sv, i) => {
      let distance2 = 0;
      sv.forEach((value, j) => { distance2 += (scaled[j] - value) ** 2; });
      margin += svm.dual_coef[i] * Math.exp(-svm.gamma * distance2);
    });
    return { logistic: {prediction: probability >= 0.5 ? 1 : 0, probability, logit: z},
             rbf: {prediction: margin > 0 ? svm.classes[1] : svm.classes[0], margin} };
  }
  root.FaultInference = {sigmoid, infer};
  if (typeof module !== 'undefined' && module.exports) module.exports = root.FaultInference;
})(typeof window !== 'undefined' ? window : globalThis);
