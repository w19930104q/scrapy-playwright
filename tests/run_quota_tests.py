"""Standalone deterministic test runner.

The project's tox setup drives these suites through pytest with
``--reactor=asyncio``; in this local environment the installed pytest 9 /
pytest-twisted / pytest-asyncio combination cannot resolve the unittest-style
fixtures, so the lifecycle regressions (plain ``IsolatedAsyncioTestCase``)
are run directly with the asyncio reactor installed beforehand. The test
module itself uses only the standard library and the project's helpers.
"""

import sys
import unittest

from twisted.internet.asyncioreactor import install

try:
    install()
except Exception:  # reactor already installed is fine
    pass


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(
        "tests.tests_asyncio.test_quota_lifecycle"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
