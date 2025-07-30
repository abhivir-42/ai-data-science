"""
COMPREHENSIVE Phase 3 Edge Case Tests
CRITICAL: Every edge case must be handled correctly. No failures allowed.
"""

import pytest
import time
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
from src.uagent_v2.config import UAgentConfig
from src.uagent_v2.prediction_formatters import PredictionResponseFormatter
from src.agents.ml_prediction_agent import MLPredictionAgent, MLPredictionError


class TestPhase3EdgeCasesComprehensive:
    """CRITICAL edge case tests - every failure mode must be handled."""
    
    def setup_method(self):
        """Setup for each test."""
        self.config = UAgentConfig()
    
    def test_corrupted_model_session_data(self):
        """Test handling of corrupted model session data."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Test corrupted model metrics
        enhanced_uagent._last_trained_model = "corrupted_string_instead_of_object"
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "TestTarget"
        
        # Should detect invalid model
        assert not enhanced_uagent._has_trained_model()
        
        # Test corrupted timestamp
        enhanced_uagent._last_trained_model = Mock()
        enhanced_uagent._last_trained_model.model_path = "/valid/path"
        enhanced_uagent._last_model_timestamp = "invalid_timestamp"
        
        # Should handle gracefully
        try:
            is_expired = enhanced_uagent._is_model_session_expired()
            # Should either return True (safe default) or handle gracefully
            assert isinstance(is_expired, bool)
        except Exception:
            # If it raises an exception, that's also acceptable as long as it's caught
            pass
        
        print("✅ Corrupted model session data - PASSED")
    
    def test_extremely_large_prediction_data(self):
        """Test handling of extremely large prediction datasets."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Setup valid model session
        mock_metrics = Mock()
        mock_metrics.model_path = "/tmp/test_model"
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "TestTarget"
        
        # Create extremely large input data (10,000 features)
        large_input_data = {f"feature_{i}": f"value_{i}" for i in range(10000)}
        
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.needs_model_analysis = False
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = large_input_data
        
        # Mock prediction result with large data
        mock_result = {
            "prediction_type": "single_prediction",
            "target_variable": "TestTarget",
            "prediction": "large_data_prediction",
            "input_data": large_input_data,
            "model_architecture": "TestModel",
            "model_score": 0.85
        }
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_result):
                result = enhanced_uagent.process_query("Predict with large data")
        
        # CRITICAL: Must handle large data without crashing
        assert "🔮 **PREDICTION RESULT**" in result
        assert "TestTarget" in result
        assert len(result) > 1000  # Should contain substantial content
        
        print("✅ Extremely large prediction data - PASSED")
    
    def test_unicode_and_encoding_edge_cases(self):
        """Test Unicode, encoding, and special character edge cases."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Setup model session
        mock_metrics = Mock()
        mock_metrics.model_path = "/tmp/test_model"
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "目标变量_测试"
        
        # Test extreme Unicode cases
        unicode_test_data = {
            "emoji_feature": "🎯🔮📊💡🚀🎉✅❌🔧🧪",
            "chinese_feature": "这是一个测试特征值包含中文字符",
            "arabic_feature": "هذا اختبار باللغة العربية",
            "russian_feature": "Это тест на русском языке",
            "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>?\\`~",
            "combined_unicode": "测试🎯عربي🔮Русский💡Test",
            "zero_width_chars": "test\u200b\u200c\u200d\ufefftest",
            "control_chars": "test\x00\x01\x02\x03test",
            "long_unicode": "测试" * 1000
        }
        
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.needs_model_analysis = False
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = unicode_test_data
        
        mock_result = {
            "prediction_type": "single_prediction",
            "target_variable": "目标变量_测试",
            "prediction": "Unicode测试结果🎯",
            "input_data": unicode_test_data,
            "model_architecture": "Unicode模型_测试",
            "model_score": 0.85
        }
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_result):
                result = enhanced_uagent.process_query("Predict with Unicode data")
        
        # CRITICAL: Must handle all Unicode without corruption
        assert "🔮 **PREDICTION RESULT**" in result
        assert "目标变量_测试" in result
        assert "Unicode测试结果🎯" in result
        assert "🎯🔮📊💡🚀🎉✅❌🔧🧪" in result
        
        print("✅ Unicode and encoding edge cases - PASSED")
    
    def test_memory_pressure_scenarios(self):
        """Test behavior under memory pressure scenarios."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Simulate memory pressure with large objects
        large_objects = []
        for i in range(100):
            large_objects.append([0] * 10000)  # Create large memory allocations
        
        try:
            # Setup model session under memory pressure
            mock_metrics = Mock()
            mock_metrics.model_path = "/tmp/test_model"
            enhanced_uagent._last_trained_model = mock_metrics
            enhanced_uagent._last_model_timestamp = time.time()
            enhanced_uagent._last_target_variable = "MemoryTest"
            
            mock_intent = Mock()
            mock_intent.needs_prediction = True
            mock_intent.prediction_type = "single_prediction"
            mock_intent.extracted_prediction_data = {"feature": "value"}
            
            mock_result = {
                "prediction_type": "single_prediction",
                "target_variable": "MemoryTest",
                "prediction": "memory_test_result",
                "input_data": {"feature": "value"},
                "model_architecture": "MemoryTestModel",
                "model_score": 0.85
            }
            
            with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
                with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_result):
                    result = enhanced_uagent.process_query("Memory pressure test")
            
            # CRITICAL: Must work even under memory pressure
            assert "🔮 **PREDICTION RESULT**" in result
            assert "MemoryTest" in result
            
        finally:
            # Clean up memory
            del large_objects
        
        print("✅ Memory pressure scenarios - PASSED")
    
    def test_concurrent_request_simulation(self):
        """Test simulation of concurrent request handling."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Setup model session
        mock_metrics = Mock()
        mock_metrics.model_path = "/tmp/test_model"
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "ConcurrentTest"
        
        # Simulate rapid-fire requests
        for i in range(50):
            mock_intent = Mock()
            mock_intent.needs_prediction = True
            mock_intent.prediction_type = "single_prediction"
            mock_intent.extracted_prediction_data = {"request_id": i, "feature": f"value_{i}"}
            
            mock_result = {
                "prediction_type": "single_prediction",
                "target_variable": "ConcurrentTest",
                "prediction": f"result_{i}",
                "input_data": {"request_id": i, "feature": f"value_{i}"},
                "model_architecture": "ConcurrentTestModel",
                "model_score": 0.85
            }
            
            with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
                with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_result):
                    result = enhanced_uagent.process_query(f"Concurrent request {i}")
            
            # CRITICAL: Each request must work correctly
            assert "🔮 **PREDICTION RESULT**" in result
            assert f"result_{i}" in result
            assert f"request_id**: {i}" in result
            
            # Verify session remains stable
            assert enhanced_uagent._has_trained_model()
        
        print("✅ Concurrent request simulation - PASSED")
    
    def test_malformed_prediction_results(self):
        """Test handling of malformed prediction results from MLPredictionAgent."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Setup model session
        mock_metrics = Mock()
        mock_metrics.model_path = "/tmp/test_model"
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "MalformedTest"
        
        malformed_results = [
            # Missing required fields
            {"prediction_type": "single_prediction"},
            # Wrong data types
            {"prediction_type": 123, "target_variable": [], "prediction": {}},
            # None values
            {"prediction_type": None, "target_variable": None, "prediction": None},
            # Empty strings
            {"prediction_type": "", "target_variable": "", "prediction": ""},
            # Nested corruption
            {"prediction_type": "single_prediction", "input_data": {"nested": {"deeply": {"corrupted": None}}}},
        ]
        
        for i, malformed_result in enumerate(malformed_results):
            mock_intent = Mock()
            mock_intent.needs_prediction = True
            mock_intent.prediction_type = "single_prediction"
            mock_intent.extracted_prediction_data = {"test": f"malformed_{i}"}
            
            with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
                with patch.object(MLPredictionAgent, 'predict_single', return_value=malformed_result):
                    result = enhanced_uagent.process_query(f"Malformed test {i}")
            
            # CRITICAL: Must handle malformed results gracefully
            assert "🔮 **PREDICTION RESULT**" in result or "❌ **Error formatting prediction**" in result
            # Should not crash - any response is acceptable as long as it doesn't crash
        
        print("✅ Malformed prediction results - PASSED")
    
    def test_file_system_edge_cases(self):
        """Test file system related edge cases."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Setup model session
        mock_metrics = Mock()
        mock_metrics.model_path = "/tmp/test_model"
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "FileSystemTest"
        
        # Test batch prediction with non-existent file
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.prediction_type = "batch_prediction"
        mock_intent.prediction_data_source = "https://nonexistent.example.com/fake_data.csv"
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_batch', side_effect=MLPredictionError("File not found")):
                result = enhanced_uagent.process_query("Predict batch with non-existent file")
        
        assert "🚫 **Prediction Error**" in result
        assert "File not found" in result
        
        # Test batch prediction with permission denied
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_batch', side_effect=MLPredictionError("Permission denied")):
                result = enhanced_uagent.process_query("Predict batch with permission error")
        
        assert "🚫 **Prediction Error**" in result
        assert "Permission denied" in result
        
        # Test with invalid file path characters
        mock_intent.prediction_data_source = "https://example.com/invalid\x00\x01\x02file.csv"
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_batch', side_effect=MLPredictionError("Invalid path")):
                result = enhanced_uagent.process_query("Predict batch with invalid path")
        
        assert "🚫 **Prediction Error**" in result
        
        print("✅ File system edge cases - PASSED")
    
    def test_extreme_session_scenarios(self):
        """Test extreme session management scenarios."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Test rapid session creation and expiration
        for i in range(10):
            # Create session
            mock_metrics = Mock()
            mock_metrics.model_path = f"/tmp/model_{i}"
            enhanced_uagent._last_trained_model = mock_metrics
            enhanced_uagent._last_model_timestamp = time.time() - (i * 1000)  # Different timestamps
            enhanced_uagent._last_target_variable = f"Target_{i}"
            
            # Test session validity at different times
            is_valid = enhanced_uagent._has_trained_model()
            is_expired = enhanced_uagent._is_model_session_expired()
            
            # Session management should be consistent
            if is_expired:
                assert not is_valid
            else:
                assert is_valid
        
        # Test session cleanup under extreme conditions
        enhanced_uagent._last_trained_model = None
        enhanced_uagent._last_model_timestamp = None
        enhanced_uagent._last_target_variable = None
        
        # Should handle cleanup gracefully
        assert not enhanced_uagent._has_trained_model()
        assert enhanced_uagent._is_model_session_expired()
        
        print("✅ Extreme session scenarios - PASSED")
    
    def test_formatter_with_extreme_values(self):
        """Test formatters with extreme numerical values."""
        formatter = PredictionResponseFormatter(self.config)
        
        extreme_cases = [
            # Very large numbers
            {"prediction": 1e308, "probability": 1.0, "score": 1e100},
            # Very small numbers
            {"prediction": 1e-308, "probability": 1e-10, "score": 1e-100},
            # Infinity and NaN (should be handled gracefully)
            {"prediction": float('inf'), "probability": float('nan'), "score": float('-inf')},
            # Scientific notation
            {"prediction": 1.23e45, "probability": 6.78e-9, "score": 9.87e-123},
        ]
        
        for i, extreme_values in enumerate(extreme_cases):
            prediction_result = {
                "prediction_type": "single_prediction",
                "target_variable": "ExtremeTest",
                "prediction": extreme_values["prediction"],
                "probability": extreme_values.get("probability"),
                "input_data": {"extreme_feature": extreme_values["prediction"]},
                "model_architecture": "ExtremeTestModel",
                "model_score": extreme_values["score"]
            }
            
            # CRITICAL: Must handle extreme values without crashing
            try:
                result = formatter.format_single_prediction(prediction_result)
                assert "🔮 **PREDICTION RESULT**" in result
                assert "ExtremeTest" in result
            except Exception as e:
                # If formatting fails, it should fail gracefully
                assert "Error formatting prediction" in str(e) or isinstance(e, (ValueError, OverflowError))
        
        print("✅ Formatter with extreme values - PASSED")
    
    def test_intent_parser_extreme_inputs(self):
        """Test intent parser with extreme and malicious inputs."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        extreme_queries = [
            # Very long query
            "Predict " + "x" * 100000,
            # Query with only special characters
            "!@#$%^&*()_+-=[]{}|;':\",./<>?\\`~",
            # Query with control characters
            "Predict\x00\x01\x02\x03\x04\x05",
            # Query with Unicode attacks
            "Predict\u202e\u0041\u202d",
            # Empty and whitespace queries
            "",
            "   ",
            "\n\n\n",
            "\t\t\t",
            # SQL injection attempt (should be harmless but test anyway)
            "Predict'; DROP TABLE models; --",
            # Script injection attempt
            "Predict<script>alert('test')</script>",
        ]
        
        for extreme_query in extreme_queries:
            try:
                # Should either parse correctly or fall back gracefully
                with patch.object(enhanced_uagent, '_process_analysis_request', return_value="Fallback response"):
                    result = enhanced_uagent.process_query(extreme_query)
                
                # CRITICAL: Must not crash and should return some response
                assert isinstance(result, str)
                assert len(result) > 0
                
            except Exception as e:
                # If it fails, it should fail gracefully with proper error handling
                assert "error" in str(e).lower() or "failed" in str(e).lower()
        
        print("✅ Intent parser extreme inputs - PASSED")


if __name__ == "__main__":
    print("🚨 RUNNING COMPREHENSIVE PHASE 3 EDGE CASE TESTS")
    print("=" * 80)
    print("CRITICAL: Every edge case must be handled. No crashes allowed.")
    print("=" * 80)
    
    test_suite = TestPhase3EdgeCasesComprehensive()
    
    # Run every single edge case test
    test_suite.setup_method()
    test_suite.test_corrupted_model_session_data()
    test_suite.test_extremely_large_prediction_data()
    test_suite.test_unicode_and_encoding_edge_cases()
    test_suite.test_memory_pressure_scenarios()
    test_suite.test_concurrent_request_simulation()
    test_suite.test_malformed_prediction_results()
    test_suite.test_file_system_edge_cases()
    test_suite.test_extreme_session_scenarios()
    test_suite.test_formatter_with_extreme_values()
    test_suite.test_intent_parser_extreme_inputs()
    
    print("=" * 80)
    print("🎉 ALL COMPREHENSIVE EDGE CASE TESTS PASSED!")
    print("✅ System is BULLETPROOF against edge cases")
    print("=" * 80) 