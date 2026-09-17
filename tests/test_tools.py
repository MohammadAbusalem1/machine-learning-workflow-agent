from src.tools import MLWorkflowTools


def test_iris_logistic_regression_workflow(tmp_path):
    tools = MLWorkflowTools(output_dir=tmp_path)

    load_result = tools.dataset_loader("Iris dataset")
    assert "Loaded iris dataset" in load_result

    preprocess_result = tools.dataset_preprocessing("iris")
    assert "Preprocessed iris" in preprocess_result

    train_result = tools.train_model("LogisticRegression")
    assert "Trained logistic regression" in train_result

    evaluation = tools.evaluate_model()
    assert "accuracy" in evaluation


def test_model_aliases(tmp_path):
    tools = MLWorkflowTools(output_dir=tmp_path)
    tools.dataset_preprocessing("iris")

    for model_name in ["logistic_regression", "decision_tree", "KNN"]:
        result = tools.train_model(model_name)
        assert "Trained" in result
