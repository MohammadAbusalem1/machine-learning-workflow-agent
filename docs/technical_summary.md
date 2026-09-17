# Technical Summary

## Project Scope

This project is a refactored version of a university machine-learning agent assignment. The original notebook explored an LLM-controlled workflow with tools for dataset lookup, loading, preprocessing, model training, evaluation, and visualization.

The portfolio version focuses on making that idea runnable, modular, secure, and easier to test. It supports tabular classification workflows on the Iris and Palmer Penguins datasets with Logistic Regression, Decision Tree, and K-Nearest Neighbors classifiers.

## Architecture

The project separates the system into two layers:

1. **Deterministic ML tools** perform dataset operations, preprocessing, training, evaluation, and plotting.
2. **LLM orchestration** decides which tool should be called next based on a natural-language request and prior tool observations.

The LLM does not calculate metrics itself. Evaluation values are produced by scikit-learn after the model is trained.

## Tool State

`MLWorkflowTools` stores the currently loaded dataset, train/test split, scaler, selected model, and feature metadata. This allows multiple tools to participate in one workflow without passing large arrays through the language model.

## Dataset Handling

### Iris

The built-in scikit-learn Iris dataset is used as a three-class classification problem with four numeric features.

### Palmer Penguins

The Palmer Penguins dataset is treated as a species-classification problem. Four physical measurements are used as numeric predictors, incomplete rows are removed, and species is preserved as the classification target.

## Preprocessing

The dataset is split into training and test sets using a fixed random seed and stratification. `StandardScaler` is fitted only on the training features and then applied to the test set, avoiding test-set leakage.

## Supported Models

- Logistic Regression with `solver="lbfgs"`
- Decision Tree with `max_depth=3`
- K-Nearest Neighbors with `n_neighbors=5`

Model-name aliases are normalized so inputs such as `LogisticRegression`, `logistic_regression`, and `logistic regression` resolve to the same implementation.

## Evaluation

The evaluation tool reports:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1 score

The visualization tool can generate a confusion matrix and class-wise accuracy plot. Models with coefficients or feature importances also receive a feature-importance plot.

## Agent Control

The LLM is instructed to return one JSON tool action at a time. The Python controller executes the requested tool, returns the observation to the LLM, and repeats until the model returns a final completion message or the step limit is reached.

This structured interface is more robust than parsing free-form model responses for action strings.

## Security

API credentials are read from environment variables. `.env` is ignored by Git so secrets are not stored in the repository.
