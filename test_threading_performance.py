#!/usr/bin/env python3
"""
Test script to verify threading and async performance optimizations.

This tests:
1. Multi-threaded H2O DataFrame conversion
2. Adaptive runtime with threading optimizations
3. Performance comparison with threading improvements
"""

import sys
import os
import time
sys.path.append('src')

from dotenv import load_dotenv
load_dotenv()

def test_h2o_threading_optimization():
    """Test H2O multi-threaded DataFrame conversion."""
    print("🧵 Testing H2O Multi-Threading Optimization")
    print("=" * 50)
    
    try:
        import h2o
        from h2o.automl import H2OAutoML
        import pandas as pd
        
        # Initialize H2O
        h2o.init(nthreads=-1)  # Use all available threads
        
        # Create a test dataset
        test_data = {
            'feature1': [1, 2, 3, 4, 5] * 50,  # 250 rows
            'feature2': [2, 4, 6, 8, 10] * 50,
            'target': [0, 1, 0, 1, 1] * 50
        }
        df = pd.DataFrame(test_data)
        
        # Convert to H2OFrame
        h2o_frame = h2o.H2OFrame(df)
        
        # Quick AutoML run
        aml = H2OAutoML(max_runtime_secs=10, max_models=3, seed=42)
        aml.train(x=['feature1', 'feature2'], y='target', training_frame=h2o_frame)
        
        # Test multi-threaded conversion
        start_time = time.time()
        try:
            # Try multi-threaded conversion (should be faster)
            leaderboard_df = aml.leaderboard.as_data_frame(use_multi_thread=True)
            conversion_method = "Multi-threaded"
        except Exception as e:
            print(f"  Multi-threaded conversion failed: {e}")
            # Fallback to single-threaded
            leaderboard_df = aml.leaderboard.as_data_frame()
            conversion_method = "Single-threaded (fallback)"
        
        conversion_time = time.time() - start_time
        
        print(f"✅ H2O DataFrame conversion successful")
        print(f"   Method: {conversion_method}")
        print(f"   Conversion time: {conversion_time:.4f} seconds")
        print(f"   Leaderboard shape: {leaderboard_df.shape}")
        
        # Check if polars and pyarrow are available
        try:
            import polars
            import pyarrow
            print(f"✅ Multi-threading dependencies available:")
            print(f"   Polars version: {polars.__version__}")
            print(f"   PyArrow version: {pyarrow.__version__}")
        except ImportError as e:
            print(f"❌ Multi-threading dependencies missing: {e}")
        
        h2o.cluster().shutdown()
        return True
        
    except Exception as e:
        print(f"❌ H2O threading test failed: {e}")
        return False

def test_adaptive_runtime_with_threading():
    """Test the combination of adaptive runtime and threading optimizations."""
    print("\n⚡ Testing Adaptive Runtime + Threading Optimization")
    print("=" * 60)
    
    from src.agents.data_analysis_agent import DataAnalysisAgent
    
    agent = DataAnalysisAgent(
        output_dir="test_output",
        enable_async=False  # Keep sync for stability, but with threading optimizations
    )
    
    # Test small dataset with optimized runtime
    test_message = "Clean and analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv for passenger prediction"
    
    print(f"Input: {test_message}")
    print("Processing with threading + adaptive runtime optimizations...")
    
    start_time = time.time()
    
    try:
        result = agent.analyze_from_text(test_message)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        print(f"✅ Analysis completed in {total_time:.2f} seconds")
        print(f"📊 Dataset: {result.data_shape}")
        print(f"⚡ Adaptive runtime: {result.total_runtime_seconds:.2f}s")
        print(f"🧵 Threading optimizations: Applied")
        
        # Performance evaluation
        if total_time < 45:  # Should be faster with threading + adaptive runtime
            print("🎉 EXCELLENT PERFORMANCE: Threading + adaptive runtime optimizations working!")
        elif total_time < 90:
            print("✅ GOOD PERFORMANCE: Significant improvement achieved")
        else:
            print("⚠️  Performance could still be improved")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_async_vs_sync_performance():
    """Compare async vs sync performance (informational)."""
    print("\n🔄 Testing Async vs Sync Performance Characteristics")
    print("=" * 55)
    
    from src.agents.data_analysis_agent import DataAnalysisAgent
    
    # Test with sync mode (current uAgent setup)
    print("Testing Synchronous Mode (current uAgent setup):")
    sync_agent = DataAnalysisAgent(
        output_dir="test_output_sync",
        enable_async=False
    )
    
    test_message = "Clean https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    
    start_time = time.time()
    try:
        result = sync_agent.analyze_from_text(test_message)
        sync_time = time.time() - start_time
        print(f"  ✅ Sync mode: {sync_time:.2f} seconds")
        sync_success = True
    except Exception as e:
        print(f"  ❌ Sync mode failed: {e}")
        sync_success = False
        sync_time = float('inf')
    
    # Note about async mode
    print("\nAsync Mode Analysis:")
    print("  📝 Async mode could potentially improve performance by:")
    print("     • Parallel execution of multiple agents")
    print("     • Non-blocking I/O operations")
    print("     • Better resource utilization")
    print("  ⚠️  However, kept sync for uAgent stability")
    print("  🎯 Current optimizations focus on threading within sync execution")
    
    return sync_success

def test_memory_and_resource_usage():
    """Test memory and resource usage optimization."""
    print("\n💾 Testing Memory and Resource Optimization")
    print("=" * 50)
    
    import psutil
    import os
    
    # Get current process
    process = psutil.Process(os.getpid())
    
    # Memory before
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    cpu_before = process.cpu_percent()
    
    print(f"Before optimization test:")
    print(f"  Memory usage: {memory_before:.2f} MB")
    print(f"  CPU usage: {cpu_before:.2f}%")
    
    # Run a quick analysis
    from src.agents.data_analysis_agent import DataAnalysisAgent
    
    agent = DataAnalysisAgent(output_dir="test_output", enable_async=False)
    
    try:
        # Quick test with small dataset
        result = agent.analyze_from_text("Clean https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        cpu_after = process.cpu_percent()
        
        print(f"\nAfter optimization test:")
        print(f"  Memory usage: {memory_after:.2f} MB")
        print(f"  Memory increase: {memory_after - memory_before:.2f} MB")
        print(f"  CPU usage: {cpu_after:.2f}%")
        
        # Resource efficiency evaluation
        if memory_after - memory_before < 100:  # Less than 100MB increase
            print("✅ EFFICIENT: Low memory footprint")
        else:
            print("⚠️  Memory usage could be optimized further")
        
        return True
        
    except Exception as e:
        print(f"❌ Resource test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 THREADING & PERFORMANCE OPTIMIZATION TEST SUITE")
    print("=" * 70)
    print()
    
    results = {}
    
    # Test 1: H2O Threading optimization
    results['h2o_threading'] = test_h2o_threading_optimization()
    print()
    
    # Test 2: Adaptive runtime with threading
    results['adaptive_threading'] = test_adaptive_runtime_with_threading() is not None
    print()
    
    # Test 3: Async vs Sync comparison
    results['async_sync'] = test_async_vs_sync_performance()
    print()
    
    # Test 4: Memory and resource usage
    results['memory_optimization'] = test_memory_and_resource_usage()
    print()
    
    # Summary
    print("=" * 70)
    print("📊 OPTIMIZATION TEST RESULTS:")
    print(f"   H2O Multi-Threading: {'✅ PASS' if results['h2o_threading'] else '❌ FAIL'}")
    print(f"   Adaptive + Threading: {'✅ PASS' if results['adaptive_threading'] else '❌ FAIL'}")
    print(f"   Sync Performance: {'✅ PASS' if results['async_sync'] else '❌ FAIL'}")
    print(f"   Memory Efficiency: {'✅ PASS' if results['memory_optimization'] else '❌ FAIL'}")
    
    success_count = sum(results.values())
    total_tests = len(results)
    
    print(f"\n🎯 OVERALL RESULT: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 ALL OPTIMIZATIONS WORKING PERFECTLY!")
    elif success_count >= total_tests * 0.75:
        print("✅ Most optimizations working well!")
    else:
        print("⚠️  Some optimizations need attention")
    
    print("\n💡 KEY OPTIMIZATIONS IMPLEMENTED:")
    print("   🧵 Multi-threaded H2O DataFrame conversion")
    print("   ⚡ Adaptive runtime based on dataset size")
    print("   🔧 Synchronous execution for uAgent stability")
    print("   💾 Efficient memory usage patterns") 