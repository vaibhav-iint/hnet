def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (downloads, large models); deselect with -m 'not slow'")
