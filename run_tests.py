#!/usr/bin/env python3
"""
Comprehensive test runner for SOWLv2 with different test categories and reporting.
"""
import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description or ' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def run_unit_tests():
    """Run unit tests."""
    cmd = [
        "pytest", 
        "tests/unit/", 
        "-v", 
        "--cov=sowlv2",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov/unit",
        "--tb=short"
    ]
    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    cmd = [
        "pytest",
        "tests/integration/",
        "-v",
        "--cov=sowlv2",
        "--cov-append",
        "--cov-report=term-missing", 
        "--cov-report=html:htmlcov/integration",
        "--tb=short"
    ]
    return run_command(cmd, "Integration Tests")


def run_output_structure_tests():
    """Run comprehensive output structure validation tests."""
    cmd = [
        "pytest",
        "tests/integration/test_output_structure.py",
        "-v",
        "--tb=short",
        "-k", "test_image_output_structure or test_video_output_structure"
    ]
    return run_command(cmd, "Output Structure Tests")


def run_flag_combination_tests():
    """Run all flag combination tests."""
    cmd = [
        "pytest",
        "tests/integration/test_output_structure.py::TestFlagCombinationMatrix",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Flag Combination Tests")


def run_cli_tests():
    """Run CLI functionality tests."""
    cmd = [
        "pytest",
        "tests/unit/test_cli.py",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "CLI Tests")


def run_edge_case_tests():
    """Run edge case and error handling tests."""
    cmd = [
        "pytest",
        "tests/integration/test_edge_cases.py",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Edge Case Tests")


def run_slow_tests():
    """Run slow/performance tests."""
    cmd = [
        "pytest",
        "tests/",
        "-v",
        "-m", "slow",
        "--tb=short"
    ]
    return run_command(cmd, "Slow/Performance Tests")


def run_lint():
    """Run linting checks."""
    cmd = ["pylint", "sowlv2/", "--exit-zero", "--output-format=text"]
    return run_command(cmd, "Linting (pylint)")


def run_all_tests():
    """Run all tests with comprehensive coverage."""
    cmd = [
        "pytest",
        "tests/",
        "-v",
        "--cov=sowlv2",
        "--cov-branch",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov/all",
        "--cov-report=xml:coverage.xml",
        "--tb=short",
        "-m", "not slow"  # Exclude slow tests by default
    ]
    return run_command(cmd, "All Tests (excluding slow)")


def generate_coverage_report():
    """Generate final coverage report."""
    print(f"\n{'='*60}")
    print("Coverage Report Summary")
    print(f"{'='*60}")
    
    # Try to read coverage data
    try:
        result = subprocess.run(
            ["coverage", "report", "--show-missing"],
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Coverage data not available. Run tests with coverage first.")
    except FileNotFoundError:
        print("Coverage tool not available. Install with: pip install coverage")


def validate_test_structure():
    """Validate that test structure is complete."""
    print(f"\n{'='*60}")
    print("Validating Test Structure")
    print(f"{'='*60}")
    
    required_files = [
        "tests/conftest.py",
        "tests/unit/test_cli.py",
        "tests/unit/utils/test_filesystem_utils.py",
        "tests/unit/utils/test_path_config.py",
        "tests/unit/models/test_owl.py",
        "tests/integration/test_output_structure.py",
        "tests/integration/test_edge_cases.py",
        "tests/test_requirements.txt",
        "pytest.ini"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing test files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("✅ All required test files are present")
        return True


def install_test_dependencies():
    """Install test dependencies."""
    cmd = ["pip", "install", "-r", "tests/test_requirements.txt"]
    return run_command(cmd, "Installing Test Dependencies")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="SOWLv2 Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--output-structure", action="store_true", help="Run output structure tests")
    parser.add_argument("--flags", action="store_true", help="Run flag combination tests")
    parser.add_argument("--cli", action="store_true", help="Run CLI tests")
    parser.add_argument("--edge-cases", action="store_true", help="Run edge case tests")
    parser.add_argument("--slow", action="store_true", help="Run slow/performance tests")
    parser.add_argument("--lint", action="store_true", help="Run linting only")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    parser.add_argument("--validate", action="store_true", help="Validate test structure")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Change to project root directory
    os.chdir(Path(__file__).parent)
    
    success = True
    
    if args.install_deps:
        success &= install_test_dependencies()
    
    if args.validate:
        success &= validate_test_structure()
    
    if args.lint:
        success &= run_lint()
    
    if args.unit:
        success &= run_unit_tests()
    
    if args.integration:
        success &= run_integration_tests()
    
    if args.output_structure:
        success &= run_output_structure_tests()
    
    if args.flags:
        success &= run_flag_combination_tests()
    
    if args.cli:
        success &= run_cli_tests()
    
    if args.edge_cases:
        success &= run_edge_case_tests()
    
    if args.slow:
        success &= run_slow_tests()
    
    if args.all or not any([args.unit, args.integration, args.output_structure, 
                           args.flags, args.cli, args.edge_cases, args.slow, 
                           args.lint, args.validate]):
        # Run all tests by default
        success &= validate_test_structure()
        success &= run_all_tests()
    
    if args.coverage:
        generate_coverage_report()
    
    # Final summary
    print(f"\n{'='*60}")
    print("Test Run Summary")
    print(f"{'='*60}")
    
    if success:
        print("✅ All tests completed successfully!")
        
        # Show coverage location
        if os.path.exists("htmlcov"):
            print(f"\n📊 Coverage reports available at:")
            print(f"  - HTML: file://{os.path.abspath('htmlcov')}/index.html")
        
        if os.path.exists("coverage.xml"):
            print(f"  - XML: {os.path.abspath('coverage.xml')}")
        
        return 0
    else:
        print("❌ Some tests failed or encountered errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())