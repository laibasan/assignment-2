/* UI only. All model calculations are in the separately tested inference.js. */
'use strict';
const models = window.FAULT_MODELS;
const inputs = Array.from(document.querySelectorAll('input'));
const names = models.feature_order;
const el = id => document.getElementById(id);
function resetResults() {
  ['logistic', 'svm'].forEach(id => { el(id + '-status').textContent = 'Snapshot changed'; el(id + '-status').className = 'status'; });
  el('agreement').textContent = 'Run analysis'; el('agreement').className = 'agreement';
  el('probability').textContent = '—'; el('margin').textContent = '—'; el('probability-fill').style.width = '0%';
  el('sample-note').textContent = '';
}
function update() {
  el('error').textContent = '';
  try {
    const values = inputs.map(input => input.value.trim() === '' ? NaN : Number(input.value));
    const result = FaultInference.infer(values, models);
    for (const [id, out] of [['logistic', result.logistic], ['svm', result.rbf]]) {
      const status = el(id + '-status');
      status.textContent = out.prediction ? '● Fault detected' : '● No fault detected';
      status.className = 'status ' + (out.prediction ? 'fault' : 'normal');
    }
    el('probability').textContent = (100 * result.logistic.probability).toFixed(2) + '%';
    el('probability-fill').style.width = (100 * result.logistic.probability) + '%';
    el('margin').textContent = (result.rbf.margin >= 0 ? '+' : '') + result.rbf.margin.toFixed(3);
    const agrees = result.logistic.prediction === result.rbf.prediction;
    el('agreement').textContent = agrees ? 'Models agree' : 'Models disagree';
    el('agreement').className = 'agreement' + (agrees ? '' : ' disagree');
  } catch (error) { resetResults(); el('error').textContent = error.message; }
}
document.getElementById('prediction-form').addEventListener('submit', event => { event.preventDefault(); update(); });
inputs.forEach(input => input.addEventListener('input', resetResults));
function example(index) {
  const sample = window.FAULT_EXAMPLES[index];
  inputs.forEach((input, i) => { input.value = String(Number(sample.values[i].toFixed(i < 3 ? 2 : 4))); });
  update(); el('sample-note').textContent = 'Rounded training sample · actual label: ' + (sample.actual_class ? 'fault' : 'no fault');
}
el('example-normal').addEventListener('click', () => example(0));
el('example-fault').addEventListener('click', () => example(1));
el('logistic-accuracy').textContent = (100 * models.test_results[1].accuracy).toFixed(2) + '%';
el('svm-accuracy').textContent = (100 * models.test_results[3].accuracy).toFixed(2) + '%';
el('support-count').textContent = models.rbf.support_vectors.length.toLocaleString();
el('svm-setting').textContent = 'C = ' + models.rbf.C + ' · gamma = ' + models.rbf.gamma;
example(1);
