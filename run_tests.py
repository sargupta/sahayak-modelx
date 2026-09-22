import sys
import pytest

def run_all_tests():
    print("=" * 60)
    print("Running SyntheticTutor Test Suite")
    print("=" * 60)

    exit_code = pytest.main(["tests", "-v"])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_all_tests()


