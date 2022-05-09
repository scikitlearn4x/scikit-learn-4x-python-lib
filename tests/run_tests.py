import sys
import unittest
from unit_tests.test_binary_buffer import *

if __name__ == '__main__':
    tests = unittest.main(exit=False)

    # ------------------------------------------------------------------------------
    # Don't change these lines, they are used for the deployment script.
    # ------------------------------------------------------------------------------
    if tests.result.wasSuccessful():
        print('All Tests Passed!')
        print(f'Total tests: {tests.result.testsRun}')
    else:
        print('Unit Tests failed!')
        exit(1)
    # ------------------------------------------------------------------------------
