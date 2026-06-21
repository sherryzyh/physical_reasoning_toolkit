from prkit.core.domain import PhysicalDataset, PhysicsDomain
from prkit.datasets.loaders.base_loader import (
    BaseDatasetLoader,
    is_mathematical_expression,
    is_pure_number,
)


class DummyLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def field_mapping(self):
        return {"qid": "problem_id", "prompt": "question", "gold": "answer"}

    @property
    def modalities(self):
        return ["text", "image"]

    def load(self, data_dir, **kwargs) -> PhysicalDataset:
        return PhysicalDataset(problems=[])

    def get_info(self):
        return {"variants": ["mini", "full"], "splits": ["train", "full"]}


def test_base_loader_numeric_and_math_detection():
    assert is_pure_number("1e-3") is True
    assert is_pure_number("3/4") is True
    assert is_pure_number("not-a-number") is False
    assert is_mathematical_expression("x + y") is True


def test_base_loader_defaults_validation_and_metadata(tmp_path, monkeypatch):
    loader = DummyLoader()

    assert loader.get_default_variant() == "full"
    assert loader.get_default_split() == "full"
    loader.validate_variant("mini")
    loader.validate_split("train")

    metadata = loader.initialize_metadata(
        {"qid": 7, "prompt": "Question?", "gold": "4", "language": "english"}
    )
    assert metadata["problem_id"] == "7"
    assert metadata["question"] == "Question?"
    assert metadata["answer"] == "4"
    assert metadata["language"] == "en"
    assert loader.validate_required_fields(metadata) == []
    assert loader.validate_required_fields({"problem_id": "x"}) == ["question"]

    monkeypatch.setenv("DATASET_CACHE_DIR", str(tmp_path))
    assert loader.resolve_data_dir(default_subdir="dummy") == tmp_path / "dummy"


def test_base_loader_creates_problem_and_loads_images(tmp_path):
    loader = DummyLoader()
    image_file = tmp_path / "sample.png"
    image_file.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    problem = loader.create_physics_problem(
        {
            "problem_id": "p1",
            "question": "What is 2 + 2?",
            "answer": "4",
            "domain": PhysicsDomain.CLASSICAL_MECHANICS,
            "image_paths": [image_file.name],
            "extra": "meta",
        },
        data_dir=tmp_path,
    )

    assert problem.problem_id == "p1"
    assert isinstance(problem.answer.value, str)
    assert problem.answer.value == "4"
    assert problem.answer.source_type is None
    assert problem.image_path == [str(image_file.resolve())]
    assert problem.additional_fields["extra"] == "meta"
    assert loader._determine_problem_type({"options": ["A", "B"]}) == "MC"
    assert loader._determine_problem_type({}) == "OE"

    images = loader.load_images_from_paths(problem.image_path, data_dir=tmp_path)
    assert len(images) == 1
    assert images[0].size == (1, 1)


def test_base_loader_handles_invalid_image_inputs_and_answer_types():
    loader = DummyLoader()

    assert loader.load_images_from_paths(None) == []
    assert loader.load_images_from_paths(123) == []

    # dict answer with unit → unit is preserved in Answer
    unit_answer = loader._create_answer_from_raw(
        {"answer": {"value": "5", "unit": "m"}}
    )
    assert unit_answer is not None
    assert unit_answer.unit == "m"
    assert unit_answer.value == "5"

    # plain expression answer → no kind needed; engine derives
    plain_answer = loader._create_answer_from_raw({"answer": "F = ma"})
    assert plain_answer is not None
    assert plain_answer.value == "F = ma"
    assert plain_answer.source_type is None


def test_base_loader_source_type_from_metadata():
    loader = DummyLoader()

    answer = loader._create_answer_from_raw({"answer": "42", "source_type": "Integer"})
    assert answer is not None
    assert answer.source_type == "Integer"


def test_base_loader_strips_latex_wrappers():
    loader = DummyLoader()

    boxed = loader._create_answer_from_raw({"answer": "\\boxed{9.81}"})
    assert boxed is not None
    assert boxed.value == "9.81"

    dollar = loader._create_answer_from_raw({"answer": "$x^2$"})
    assert dollar is not None
    assert dollar.value == "x^2"

    ddollar = loader._create_answer_from_raw({"answer": "$$42$$"})
    assert ddollar is not None
    assert ddollar.value == "42"
