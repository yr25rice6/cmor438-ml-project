import cmor438_ml


def test_package_imports():
    assert cmor438_ml is not None


def test_has_version():
    assert hasattr(cmor438_ml, "__version__")


def test_version_is_non_empty_string():
    assert isinstance(cmor438_ml.__version__, str)
    assert cmor438_ml.__version__ != ""
