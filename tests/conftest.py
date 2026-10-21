import pytest


def pytest_addoption(parser):
    parser.addoption("--run-gui", action="store_true", help="Actually launch the Tk test window")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-gui"):
        for item in items:
            if "gui" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="GUI not requested; pass --run-gui on a desktop"))
