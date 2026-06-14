"""Tests for prkit.core.exceptions hierarchy and factory wiring."""

from __future__ import annotations

import pytest

from prkit.core import (
    ModelClientError as CoreModelClientError,
)
from prkit.core import (
    PRKitError as CorePRKitError,
)
from prkit.core import (
    UnknownModelError as CoreUnknownModelError,
)
from prkit.core.exceptions import (
    ConfigError,
    DatasetError,
    ModelClientError,
    PRKitError,
    UnknownModelError,
)


class TestInheritanceContract:
    def test_unknown_model_error_is_prkit_error(self):
        exc = UnknownModelError("bad-model")
        assert isinstance(exc, PRKitError)

    def test_unknown_model_error_is_value_error(self):
        exc = UnknownModelError("bad-model")
        assert isinstance(exc, ValueError)

    def test_model_client_error_is_prkit_error(self):
        exc = ModelClientError("failed")
        assert isinstance(exc, PRKitError)

    def test_model_client_error_is_runtime_error(self):
        exc = ModelClientError("failed")
        assert isinstance(exc, RuntimeError)

    def test_config_error_is_prkit_error(self):
        assert isinstance(ConfigError("bad config"), PRKitError)

    def test_config_error_is_value_error(self):
        assert isinstance(ConfigError("bad config"), ValueError)

    def test_dataset_error_is_prkit_error(self):
        assert isinstance(DatasetError("load failed"), PRKitError)

    def test_dataset_error_is_runtime_error(self):
        assert isinstance(DatasetError("load failed"), RuntimeError)

    def test_all_subtypes_catchable_as_prkit_error(self):
        for exc_cls in (UnknownModelError, ModelClientError, ConfigError, DatasetError):
            with pytest.raises(PRKitError):
                raise exc_cls("test")


class TestCoreReExports:
    """Ensure the public surface in prkit.core re-exports the same objects."""

    def test_prkit_error_re_exported(self):
        assert CorePRKitError is PRKitError

    def test_unknown_model_error_re_exported(self):
        assert CoreUnknownModelError is UnknownModelError

    def test_model_client_error_re_exported(self):
        assert CoreModelClientError is ModelClientError


class TestFactoryRaisesUnknownModelError:
    """factory.create_model_client must raise UnknownModelError for bad model names."""

    def test_completely_unknown_model(self):
        from prkit.core.model_clients import create_model_client

        with pytest.raises(UnknownModelError, match="Unknown model"):
            create_model_client("no-such-provider-xyz")

    def test_unknown_model_also_caught_as_value_error(self):
        from prkit.core.model_clients import create_model_client

        with pytest.raises(ValueError):
            create_model_client("no-such-provider-xyz")

    def test_unsupported_gpt_variant(self):
        from prkit.core.model_clients import create_model_client

        with pytest.raises(UnknownModelError, match="Unsupported OpenAI model"):
            create_model_client("gpt-3-turbo")
