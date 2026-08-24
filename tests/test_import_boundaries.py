import subprocess
import sys


def _assert_clean_import(statement: str) -> None:
    subprocess.run(
        [sys.executable, "-c", statement],
        check=True,
    )


def test_awg_package_does_not_import_alazar_stack():
    _assert_clean_import(
        "import sys; import QAWG.awg5200; "
        "assert 'QAWG.alazar' not in sys.modules"
    )


def test_alazar_package_does_not_import_awg_driver():
    _assert_clean_import(
        "import sys; import QAWG.alazar; "
        "assert 'QAWG.awg5200.driver' not in sys.modules"
    )
