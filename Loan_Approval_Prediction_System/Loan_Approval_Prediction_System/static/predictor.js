/* Inference helpers matching the fitted scikit-learn preprocessing and classifiers. */
(function () {
  "use strict";

  function encodeApplicant(metadata, model, input) {
    const row = [];
    metadata.numeric_features.forEach((feature) => {
      const spec = model.numeric_specs[feature];
      let value = Number(input[feature]);
      if (!Number.isFinite(value)) value = spec.median;
      if (Object.hasOwn(spec, "scale")) value = (value - spec.mean) / spec.scale;
      row.push(value);
    });
    metadata.categorical_features.forEach((feature) => {
      const spec = model.categorical_specs[feature];
      const value = String(input[feature] || spec.most_frequent).trim();
      spec.categories.forEach((category) => row.push(value === String(category).trim() ? 1 : 0));
    });
    return row;
  }

  function sigmoid(value) {
    if (value > 35) return 1;
    if (value < -35) return 0;
    return 1 / (1 + Math.exp(-value));
  }

  function normalized(values) {
    const total = values.reduce((sum, value) => sum + value, 0);
    if (total === 0) return values.map(() => 1 / values.length);
    return values.map((value) => value / total);
  }

  function predictTree(tree, row) {
    let node = 0;
    while (tree.left[node] !== -1) {
      const feature = tree.feature[node];
      node = row[feature] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    }
    return normalized(tree.values[node]);
  }

  function predict(metadata, model, input) {
    const encoded = encodeApplicant(metadata, model, input);
    let probabilities;
    if (model.kind === "logistic_regression") {
      let decision = model.intercept;
      model.coefficients.forEach((weight, index) => { decision += weight * encoded[index]; });
      const positiveProbability = sigmoid(decision);
      probabilities = [1 - positiveProbability, positiveProbability];
    } else if (model.kind === "decision_tree") {
      probabilities = predictTree(model.tree, encoded);
    } else {
      const totals = model.classes.map(() => 0);
      model.trees.forEach((tree) => {
        const treeProbabilities = predictTree(tree, encoded);
        treeProbabilities.forEach((probability, index) => { totals[index] += probability; });
      });
      probabilities = totals.map((total) => total / model.trees.length);
    }

    let bestIndex = 0;
    for (let index = 1; index < probabilities.length; index += 1) {
      if (probabilities[index] > probabilities[bestIndex]) bestIndex = index;
    }
    return { outcome: model.classes[bestIndex], probabilities };
  }

  window.LoanPredictor = { predict };
}());
