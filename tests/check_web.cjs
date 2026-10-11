/* Compare exported JS with saved Python predictions on validation samples only. */
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const root = path.resolve(__dirname, '..');
global.window = {};
require(path.join(root, 'app/models.js'));
const {infer} = require(path.join(root, 'app/inference.js'));
const expected = JSON.parse(fs.readFileSync(path.join(root, 'results/web_parity_validation.json')));
let maxProbabilityError = 0, maxMarginError = 0;
expected.raw.forEach((values, i) => {
  const actual = infer(values, window.FAULT_MODELS);
  const pError = Math.abs(actual.logistic.probability - expected.logistic_probability[i]);
  const marginError = Math.abs(actual.rbf.margin - expected.rbf_margin[i]);
  maxProbabilityError = Math.max(maxProbabilityError, pError);
  maxMarginError = Math.max(maxMarginError, marginError);
  assert.ok(pError < 1e-12, 'Logistic export parity');
  assert.ok(marginError < 1e-9, 'RBF margin parity');
  assert.equal(actual.rbf.prediction, expected.rbf_prediction[i]);
  assert.equal(actual.logistic.prediction, expected.logistic_probability[i] >= 0.5 ? 1 : 0);
});
assert.throws(() => infer([NaN, 0, 0, 0, 0, 0], window.FAULT_MODELS));
assert.throws(() => infer([1e308, 0, 0, 0, 0, 0], window.FAULT_MODELS));
assert.throws(() => infer([], window.FAULT_MODELS));
console.log(JSON.stringify({validation_rows: expected.raw.length, max_probability_error: maxProbabilityError,
  max_rbf_margin_error: maxMarginError, predictions_match: true, invalid_inputs_rejected: true}, null, 2));
