import pytest

from paligemma2_experiment.io import validate_full_experiment


def full_paintings() -> list[dict]:
    return [
        {
            "painting_id": f"p{index}",
            "title": "Title",
            "artist": "Artist",
            "year": "1500",
            "fame_group": "moderate",
            "image_path": "data/images/example.jpg",
            "source_url": "https://example.com",
        }
        for index in range(12)
    ]


def full_questions() -> list[dict]:
    types = [
        "distant_detail",
        "handheld_attribute",
        "inscription",
        "secondary_figure",
        "false_premise",
    ]
    return [
        {
            "painting_id": f"p{painting_index}",
            "question_id": f"p{painting_index}_{question_index}",
            "question_type": question_type,
            "prompt": "answer en Question?",
            "reference_answer": "Answer",
            "source_url": "https://example.com",
        }
        for painting_index in range(12)
        for question_index, question_type in enumerate(types)
    ]


def test_accepts_complete_full_experiment():
    validate_full_experiment(full_paintings(), full_questions())


def test_rejects_wrong_painting_count():
    with pytest.raises(ValueError, match="requires 12 paintings"):
        validate_full_experiment(full_paintings()[:-1], full_questions())
