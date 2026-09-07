# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import logging
import unittest


class TestLoggingSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Ensure logging is configured for all test files."""
        logger = logging.getLogger()
        logger.info("Logging setup for tests.")


if __name__ == "__main__":
    unittest.main()
