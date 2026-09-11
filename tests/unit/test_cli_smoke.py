from typer.testing import CliRunner

from dvi.cli import app

runner = CliRunner()


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_check_env_reports_python_ok():
    result = runner.invoke(app, ["check-env"])
    assert "Python 3.12" in result.stdout
