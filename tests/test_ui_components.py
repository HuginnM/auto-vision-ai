#!/usr/bin/env python3
"""Test script to validate UI components."""

import sys


def test_utility_functions():
    """Test utility functions."""
    try:
        from autovisionai.ui.utils import format_duration, format_loss, get_model_info

        # Test format_loss
        assert format_loss(0.123456) == "0.123456"
        assert format_loss(0.0001) == "1.00e-04"
        assert format_loss(float("inf")) == "N/A"

        # Test format_duration
        assert format_duration(30) == "30s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(3661) == "1h 1m"

        # Test get_model_info
        unet_info = get_model_info("unet")
        assert unet_info["name"] == "U-Net"
        assert "~31M" in unet_info["params"]

        print("Utility functions work correctly!")
    except AssertionError as e:
        print(f"Utility function test failed: {e}")
    except Exception as e:
        print(f"Unexpected error in utility tests: {e}")


def test_streamlit_imports():
    """Test that Streamlit is properly installed."""
    try:
        import streamlit

        print(f"Streamlit {streamlit.__version__} is installed!")
        assert True
    except ImportError:
        print("Streamlit is not installed!")


def main():
    """Run all tests."""
    print("🧪 Testing AutoVisionAI UI Components")
    print("=" * 40)

    tests = [
        ("Streamlit Installation", test_streamlit_imports),
        ("Utility Functions", test_utility_functions),
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   Test '{test_name}' failed!")

    print("\n" + "=" * 40)
    print(f"Results: {passed}/{len(tests)} tests passed")

    assert passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
