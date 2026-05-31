from src.predict import PredictionRow
from src.utils import normalize_label, normalize_option, parse_model_output


def test_normalize_label():
    assert normalize_label("no_hallucination") == "no_hallucination"
    assert normalize_label("correct") == "no_hallucination"
    assert normalize_label("hallucination") == "hallucination"


def test_normalize_option_dynamic_keys():
    keys = ["A", "B", "C", "D", "E", "F"]
    assert normalize_option("A", keys) == "A"
    assert normalize_option("option b", keys) == "B"
    assert normalize_option("c", keys) == "C"
    assert normalize_option(None, keys) is None


def test_parse_model_output_json():
    text = '{"predicted_label": "hallucination", "selected_option": "C"}'
    result = parse_model_output(text, ["A", "B", "C"])
    assert result["predicted_label"] == "hallucination"
    assert result["selected_option"] == "C"


def test_parse_model_output_no_hallucination_keeps_option():
    text = '{"predicted_label": "no_hallucination", "selected_option": "B"}'
    result = parse_model_output(text, ["A", "B", "C"])
    assert result["predicted_label"] == "no_hallucination"
    assert result["selected_option"] == "B"


def test_parse_model_output_plain_no_hallucination_keeps_option():
    result = parse_model_output("no_hallucination answer B", ["A", "B", "C"])
    assert result["predicted_label"] == "no_hallucination"
    assert result["selected_option"] == "B"


def test_prediction_row_answer_is_kept_for_all_labels():
    clean = PredictionRow("1", "q", "model answer", "no_hallucination", "A", "raw")
    hallucinated = PredictionRow("2", "q", "model answer", "hallucination", "B", "raw")

    assert clean.to_dict()["answer"] == "A"
    assert hallucinated.to_dict()["answer"] == "B"
    assert clean.to_dict(codabench=True)["pred_option"] == "A"
