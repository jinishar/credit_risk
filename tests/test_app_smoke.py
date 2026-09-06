"""Smoke test: the UI section modules import cleanly and expose a render() callable.

The visual output is verified by running the app (locally and in Docker); this
just guards against import errors and signature drift in the view layer.
"""
import importlib

import pytest

SECTION_MODULES = ["eda", "scoring", "explainability", "rules", "chat"]


@pytest.mark.parametrize("name", SECTION_MODULES)
def test_section_module_exposes_a_render_callable(name):
    module = importlib.import_module(f"app.sections.{name}")
    assert callable(module.render)


def test_data_module_exposes_the_shared_loaders():
    from app import data

    for loader in ("dataset", "risk_model", "explainer", "rules", "metrics", "data_is_real"):
        assert callable(getattr(data, loader))


def test_streamlit_app_declares_one_page_per_section():
    # Import the module object without executing the page body.
    import ast

    source = (importlib.import_module("app").__path__[0] + "/streamlit_app.py")
    tree = ast.parse(open(source, encoding="utf-8").read())
    pages = next(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.Assign)
         and any(getattr(t, "id", "") == "PAGES" for t in node.targets)),
        None,
    )
    assert pages is not None
    assert len(pages.value.keys) == len(SECTION_MODULES)
