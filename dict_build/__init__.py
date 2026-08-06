"""dict_build - Statistical Chinese new word discovery from raw text."""

try:
    from importlib.metadata import version

    __version__ = version("dict-build")
except Exception:
    # Package not installed (running from source tree)
    __version__ = "1.4.1"

__author__ = "wainshine"
