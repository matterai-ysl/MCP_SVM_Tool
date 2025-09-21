#!/usr/bin/env python3
"""
Quick test script for asynchronous training queue functionality.
"""

import asyncio
import sys
import os
sys.path.insert(0, 'src')

# Set matplotlib to non-interactive mode
import matplotlib
matplotlib.use('Agg')

from src.mcp_svm_tool.training_queue import get_queue_manager, initialize_queue_manager
import numpy as np
import pandas as pd
from pathlib import Path

async def create_simple_test_data():
    """Create very simple test dataset."""
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Simple 2D classification data
    np.random.seed(42)
    n_samples = 60  # Small dataset for fast training
    
    # Generate simple data
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear boundary
    y = np.where(y == 1, 'class_A', 'class_B')  # Convert to strings
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    df['target'] = y
    
    test_file = test_data_dir / "simple_test.csv"
    df.to_csv(test_file, index=False)
    
    print(f"✅ Simple test data created: {test_file}")
    return str(test_file)

async def quick_test():
    """Quick test of async queue functionality."""
    print("🧪 Quick Async Training Queue Test")
    print("=" * 40)
    
    try:
        # Initialize queue
        queue_manager = await initialize_queue_manager()
        
        # Create test data
        test_file = await create_simple_test_data()
        
        # Submit simple task
        print("📤 Submitting simple training task...")
        task_params = {
            'data_source': test_file,
            'target_column': 'target',
            'kernel': "linear",  # Simple kernel
            'optimize_hyperparameters': False,  # No optimization for speed
            'n_trials': 1,
            'cv_folds': 2,  # Minimal CV
            'validate_data': False  # Skip validation for speed
        }
        
        task_id = await queue_manager.submit_task(
            task_type="svm_classification",
            params=task_params,
            user_id="quick_test"
        )
        print(f"   Task submitted: {task_id}")
        
        # Monitor task
        print("👀 Monitoring task progress...")
        timeout = 60  # 1 minute timeout
        interval = 2   # Check every 2 seconds
        elapsed = 0
        
        while elapsed < timeout:
            status = await queue_manager.get_task_status(task_id)
            if status:
                print(f"   Status: {status['status']} ({status.get('progress', 0)}%)")
                
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    if status['status'] == 'completed':
                        print("   ✅ Task completed successfully!")
                        return True
                    else:
                        print(f"   ❌ Task failed: {status.get('error_message', 'Unknown error')}")
                        return False
            
            await asyncio.sleep(interval)
            elapsed += interval
        
        print("   ⏰ Test timed out")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Stop queue manager
        queue_manager = get_queue_manager()
        if queue_manager.is_running:
            await queue_manager.stop()
            print("🛑 Queue manager stopped")

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    if success:
        print("\n🎉 Quick test passed! Async training queue is working.")
    else:
        print("\n❌ Quick test failed.")