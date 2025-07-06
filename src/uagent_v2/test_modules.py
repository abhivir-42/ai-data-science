#!/usr/bin/env python3
"""
Test script for the new uAgent v2 modules.

This script tests the basic functionality of the new modules to ensure
they work correctly before integrating them into the main uAgent.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_config_module():
    """Test the configuration module."""
    print("🧪 Testing configuration module...")
    
    try:
        from uagent_v2.config import UAgentConfig, default_config
        
        # Test default configuration
        config = UAgentConfig()
        print(f"✅ Default config created: port={config.port}")
        
        # Test environment-based configuration
        config_env = UAgentConfig.from_env()
        print(f"✅ Environment config created: port={config_env.port}")
        
        # Test validation
        assert config.port == 8102, "Default port should be 8102"
        assert config.max_file_size_mb == 50, "Default max file size should be 50MB"
        
        # Test utility methods
        file_size_bytes = config.get_file_size_bytes()
        assert file_size_bytes == 50 * 1024 * 1024, "File size calculation incorrect"
        
        print("✅ Configuration module test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration module test failed: {e}")
        return False


def test_exceptions_module():
    """Test the exceptions module."""
    print("🧪 Testing exceptions module...")
    
    try:
        from uagent_v2.exceptions import (
            DataAnalysisError, SecurityError, FileProcessingError,
            handle_analysis_error, create_error_response
        )
        
        # Test custom exception
        try:
            raise SecurityError("Test security error")
        except SecurityError as e:
            assert str(e) == "Test security error"
            print("✅ SecurityError works correctly")
        
        # Test error handling
        test_error = ValueError("Test value error")
        error_msg = handle_analysis_error(test_error, "test_context")
        assert "Invalid value" in error_msg
        print("✅ Error handling works correctly")
        
        # Test error response
        error_response = create_error_response(test_error, "test_context")
        assert error_response["success"] is False
        assert "error" in error_response
        print("✅ Error response generation works correctly")
        
        print("✅ Exceptions module test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Exceptions module test failed: {e}")
        return False


def test_utils_module():
    """Test the utilities module."""
    print("🧪 Testing utilities module...")
    
    try:
        print("  - Importing modules...")
        from uagent_v2.utils import (
            MemoryEfficientCSVProcessor, DataDeliveryOptimizer,
            format_file_size, sanitize_filename
        )
        from uagent_v2.config import UAgentConfig
        print("  - Imports successful")
        
        config = UAgentConfig()
        print("  - Config created")
        
        # Test file size formatting
        result1 = format_file_size(1024)
        print(f"  - format_file_size(1024) = '{result1}'")
        assert result1 == "1.0 KB", f"Expected '1.0 KB', got '{result1}'"
        
        result2 = format_file_size(1024 * 1024)
        print(f"  - format_file_size(1048576) = '{result2}'")
        assert result2 == "1.0 MB", f"Expected '1.0 MB', got '{result2}'"
        print("✅ File size formatting works correctly")
        
        # Test filename sanitization
        unsafe_name = "../../dangerous<file>.csv"
        safe_name = sanitize_filename(unsafe_name)
        print(f"  - sanitize_filename('{unsafe_name}') = '{safe_name}'")
        assert ".." not in safe_name, f"'..' still present in '{safe_name}'"
        assert "<" not in safe_name, f"'<' still present in '{safe_name}'"
        print("✅ Filename sanitization works correctly")
        
        # Test CSV processor
        print("  - Creating CSV processor...")
        processor = MemoryEfficientCSVProcessor(config)
        assert processor.config == config
        print("✅ CSV processor initialization works correctly")
        
        # Test delivery optimizer
        print("  - Creating delivery optimizer...")
        optimizer = DataDeliveryOptimizer(config)
        assert optimizer.config == config
        print("✅ Delivery optimizer initialization works correctly")
        
        print("✅ Utils module test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Utils module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_handlers_module():
    """Test the file handlers module."""
    print("🧪 Testing file handlers module...")
    
    try:
        from uagent_v2.file_handlers import (
            SecureFileUploader, SecureFileDownloader, FileContentHandler
        )
        from uagent_v2.config import UAgentConfig
        
        config = UAgentConfig()
        
        # Test secure uploader
        uploader = SecureFileUploader(config)
        assert uploader.config == config
        print("✅ Secure uploader initialization works correctly")
        
        # Test secure downloader
        downloader = SecureFileDownloader(config)
        assert downloader.config == config
        print("✅ Secure downloader initialization works correctly")
        
        # Test content handler
        handler = FileContentHandler(config)
        assert handler.config == config
        print("✅ Content handler initialization works correctly")
        
        print("✅ File handlers module test passed!")
        return True
        
    except Exception as e:
        print(f"❌ File handlers module test failed: {e}")
        return False


def create_test_csv():
    """Create a test CSV file for testing."""
    try:
        import pandas as pd
        
        # Create a simple test DataFrame
        data = {
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'London', 'Tokyo']
        }
        df = pd.DataFrame(data)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        print(f"✅ Test CSV created: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        print(f"❌ Could not create test CSV: {e}")
        return None


def test_integration():
    """Test integration between modules."""
    print("🧪 Testing module integration...")
    
    try:
        # Create test CSV
        test_csv = create_test_csv()
        if not test_csv:
            return False
        
        from uagent_v2.config import UAgentConfig
        from uagent_v2.utils import MemoryEfficientCSVProcessor
        from uagent_v2.file_handlers import FileContentHandler
        
        config = UAgentConfig()
        processor = MemoryEfficientCSVProcessor(config)
        handler = FileContentHandler(config)
        
        # Test file size calculation
        file_size = processor.get_file_size_safe(test_csv)
        assert file_size > 0, "File size should be greater than 0"
        print(f"✅ File size calculation: {file_size} bytes")
        
        # Test file display
        display_lines = handler.create_file_display(test_csv, "test_data")
        assert len(display_lines) > 0, "Display should have content"
        print("✅ File display generation works correctly")
        
        # Clean up
        os.unlink(test_csv)
        
        print("✅ Module integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Module integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Running uAgent v2 module tests...")
    print("=" * 50)
    
    tests = [
        test_config_module,
        test_exceptions_module,
        test_utils_module,
        test_file_handlers_module,
        test_integration
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Modules are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 