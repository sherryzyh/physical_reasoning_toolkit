"""Tests for the dataset-focused PRKit CLI."""

import subprocess
import sys
from unittest.mock import patch

from prkit.cli import main


def test_version_prints_package_version(capsys):
    """`prkit --version` should print the package version."""
    with patch("prkit.cli.__version__", "1.2.3"):
        exit_code = main(["--version"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "prkit 1.2.3"


def test_list_prints_available_datasets(capsys):
    """`prkit list` should print one dataset name per line."""
    with patch("prkit.cli.DatasetHub") as mock_hub:
        mock_hub.list_available.return_value = ["phyx", "ugphysics"]

        exit_code = main(["list"])

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == ["phyx", "ugphysics"]
    mock_hub.list_available.assert_called_once_with()


def test_info_prints_dataset_metadata_as_json(capsys):
    """`prkit info <dataset>` should print formatted JSON metadata."""
    with patch("prkit.cli.DatasetHub") as mock_hub:
        mock_hub.get_info.return_value = {"name": "phyx", "splits": ["test"]}

        exit_code = main(["info", "phyx"])

    assert exit_code == 0
    assert '"name": "phyx"' in capsys.readouterr().out
    mock_hub.get_info.assert_called_once_with("phyx")


def test_download_dispatches_to_registered_downloader(capsys):
    """`prkit download` should call the dataset downloader with CLI options."""
    with patch("prkit.cli.DatasetHub") as mock_hub:
        mock_hub.download.return_value = "/tmp/prkit/phyx"

        exit_code = main(
            [
                "download",
                "phyx",
                "--variant",
                "full",
                "--split",
                "test",
                "--force",
            ]
        )

    assert exit_code == 0
    assert "Downloaded phyx to /tmp/prkit/phyx" in capsys.readouterr().out
    mock_hub.download.assert_called_once_with(
        "phyx",
        data_dir=None,
        variant="full",
        split="test",
        force=True,
    )


def test_download_reports_missing_downloader(capsys):
    """`prkit download` should fail clearly when a dataset has no downloader."""
    with patch("prkit.cli.DatasetHub") as mock_hub:
        mock_hub.download.side_effect = ValueError(
            "No downloader available for unknown"
        )

        exit_code = main(["download", "unknown"])

    assert exit_code == 1
    assert "No downloader available for unknown" in capsys.readouterr().err


def test_cli_returns_nonzero_for_hub_errors(capsys):
    """CLI command failures should produce a concise stderr message."""
    with patch("prkit.cli.DatasetHub") as mock_hub:
        mock_hub.get_info.side_effect = ValueError("Unknown dataset: nope")

        exit_code = main(["info", "nope"])

    assert exit_code == 1
    assert "Unknown dataset: nope" in capsys.readouterr().err


def test_main_uses_sys_argv_when_argv_is_none(capsys, monkeypatch):
    """`main()` should work as a console-script entry point."""
    monkeypatch.setattr("sys.argv", ["prkit", "list"])
    with patch("prkit.cli.DatasetHub") as mock_hub:
        mock_hub.list_available.return_value = ["physics"]

        exit_code = main()

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "physics"


def test_module_entrypoint_supports_python_m_version():
    """`python -m prkit.cli --version` should behave like the console script."""
    result = subprocess.run(
        [sys.executable, "-m", "prkit.cli", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.startswith("prkit ")
