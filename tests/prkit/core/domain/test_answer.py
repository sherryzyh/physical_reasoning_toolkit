"""Tests for the thin Answer observation record."""

from prkit.core.domain.answer import Answer


class TestAnswerCreation:
    def test_value_only(self):
        a = Answer(value="42")
        assert a.value == "42"
        assert a.unit is None
        assert a.source_type is None
        assert a.metadata == {}

    def test_with_unit(self):
        a = Answer(value="9.81", unit="m/s^2")
        assert a.value == "9.81"
        assert a.unit == "m/s^2"

    def test_with_source_type(self):
        a = Answer(value="A", source_type="MC")
        assert a.source_type == "MC"

    def test_with_all_fields(self):
        a = Answer(value="5", unit="N", source_type="NV", metadata={"raw": True})
        assert a.value == "5"
        assert a.unit == "N"
        assert a.source_type == "NV"
        assert a.metadata["raw"] is True

    def test_metadata_default_not_shared(self):
        a = Answer(value="x")
        b = Answer(value="y")
        a.metadata["k"] = 1
        assert "k" not in b.metadata

    def test_metadata_none_normalized(self):
        a = Answer(value="x", metadata=None)  # type: ignore[arg-type]
        assert a.metadata == {}


class TestAnswerAccessors:
    def test_get_value(self):
        a = Answer(value="hello")
        assert a.get_value() == "hello"

    def test_get_unit_present(self):
        a = Answer(value="3", unit="m")
        assert a.get_unit() == "m"

    def test_get_unit_absent(self):
        a = Answer(value="3")
        assert a.get_unit() is None

    def test_has_unit_true(self):
        a = Answer(value="3", unit="m")
        assert a.has_unit() is True

    def test_has_unit_false(self):
        a = Answer(value="3")
        assert a.has_unit() is False


class TestAnswerDunder:
    def test_str_with_unit(self):
        a = Answer(value="9.81", unit="m/s^2")
        assert str(a) == "9.81 m/s^2"

    def test_str_without_unit(self):
        a = Answer(value="42")
        assert str(a) == "42"

    def test_repr_contains_key_fields(self):
        a = Answer(value="5", unit="N", source_type="NV")
        r = repr(a)
        assert "Answer(" in r
        assert "'5'" in r
        assert "'N'" in r
        assert "'NV'" in r

    def test_repr_no_answer_kind(self):
        a = Answer(value="x")
        assert "answer_kind" not in repr(a)


class TestAnswerToDict:
    def test_value_only(self):
        d = Answer(value="42").to_dict()
        assert d == {"value": "42"}

    def test_with_unit(self):
        d = Answer(value="3", unit="m").to_dict()
        assert d == {"value": "3", "unit": "m"}

    def test_with_source_type(self):
        d = Answer(value="A", source_type="MCQ").to_dict()
        assert d == {"value": "A", "source_type": "MCQ"}

    def test_with_metadata(self):
        d = Answer(value="x", metadata={"raw": "1"}).to_dict()
        assert d["metadata"] == {"raw": "1"}

    def test_no_answer_kind_key(self):
        d = Answer(value="x").to_dict()
        assert "answer_kind" not in d
        assert "answer_category" not in d

    def test_empty_unit_omitted(self):
        d = Answer(value="x", unit=None).to_dict()
        assert "unit" not in d

    def test_none_source_type_omitted(self):
        d = Answer(value="x", source_type=None).to_dict()
        assert "source_type" not in d

    def test_empty_metadata_omitted(self):
        d = Answer(value="x", metadata={}).to_dict()
        assert "metadata" not in d
