# 修复后的异步训练使用示例

## 问题修复

✅ **修复了异步队列逻辑顺序问题**：

**之前的问题**：
- 在函数开始就检查`use_async_queue`，跳过了重要的参数验证和数据加载
- 没有确定正确的目标列（target column）就提交任务
- 可能导致异步任务中参数错误或找不到目标列

**修复后的逻辑**：
1. 首先进行参数验证（kernel类型、评分指标等）
2. 加载数据并确定目标列
3. 然后再检查是否使用异步队列
4. 传递正确的`target_column`给异步任务

## 使用示例

### 1. 分类任务 - 异步训练

```python
# 现在会正确处理参数验证和目标列确定
result = await train_svm_classifier(
    data_source="data/iris.csv",
    kernel="rbf",
    optimize_hyperparameters=True,
    n_trials=20,
    cv_folds=5,
    scoring_metric="f1_weighted",
    use_async_queue=True,  # 启用异步队列
    user_id="user123"
)

print(f"任务ID: {result['task_id']}")
print(f"任务类型: {result['task_type']}")
```

### 2. 回归任务 - 异步训练

```python
# 现在会正确处理目标维度和目标列确定
result = await train_svm_regressor(
    data_source="data/house_prices.csv", 
    target_dimension=1,  # 单目标回归
    kernel="linear",
    optimize_hyperparameters=True,
    n_trials=30,
    cv_folds=5,
    scoring_metric="r2",
    use_async_queue=True,  # 启用异步队列
    user_id="user123"
)

print(f"任务ID: {result['task_id']}")
```

### 3. 任务监控

```python
import asyncio

async def monitor_training_task(task_id):
    """监控训练任务进度"""
    
    while True:
        # 获取任务状态
        status = await get_task_status(task_id)
        
        print(f"任务 {task_id[:8]}...")
        print(f"  状态: {status['status']}")
        print(f"  进度: {status.get('progress', 0)}%")
        
        if status.get('error_message'):
            print(f"  错误: {status['error_message']}")
        
        # 检查是否完成
        if status['status'] in ['completed', 'failed', 'cancelled']:
            if status['status'] == 'completed':
                print("✅ 训练完成!")
                if 'result' in status:
                    print(f"  模型ID: {status['result'].get('model_id')}")
                    print(f"  模型性能: {status['result'].get('best_score')}")
            else:
                print(f"❌ 训练{status['status']}")
            break
        
        await asyncio.sleep(5)  # 每5秒检查一次

# 使用示例
# await monitor_training_task("your-task-id-here")
```

## 修复的关键改进

### 1. 参数验证前置

```python
# 修复前：直接检查异步队列，跳过验证
if use_async_queue:
    return await submit_training_task(...)  # 参数可能无效

# 修复后：先验证参数
valid_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
if kernel not in valid_kernels and kernel is not None:
    raise ValueError(f"Invalid kernel: {kernel}")

# 然后才检查异步队列
if use_async_queue:
    return await submit_training_task(...)
```

### 2. 目标列确定

```python
# 修复前：传递target_dimension给异步任务
return await submit_training_task(
    task_type="svm_regression",
    data_source=data_source,
    target_dimension=target_dimension,  # 异步任务需要自己确定目标列
    ...
)

# 修复后：先确定目标列，再传递给异步任务
data_processor = DataProcessor()
df = data_processor.load_data(data_source)
target_columns = df.columns[-target_dimension:].tolist()
target_column = target_columns if target_dimension > 1 else target_columns[0]

return await submit_training_task(
    task_type="svm_regression", 
    data_source=data_source,
    target_column=target_column,  # 直接传递确定的目标列
    ...
)
```

### 3. 错误处理改进

现在如果参数无效（如不支持的kernel或scoring_metric），会在提交异步任务之前就抛出错误，避免了无效任务进入队列。

## 测试结果

✅ **快速测试通过**：
- 参数验证正确执行
- 目标列正确确定和传递
- 异步训练成功完成
- 任务状态正确更新

```bash
$ python quick_test_async.py
🧪 Quick Async Training Queue Test
========================================
✅ Simple test data created: test_data/simple_test.csv
📤 Submitting simple training task...
   Task submitted: cb548a8f-8555-4bb7-9328-05c008c9bfc4
👀 Monitoring task progress...
   Status: queued (0.0%)
   Status: completed (100.0%)
   ✅ Task completed successfully!
🛑 Queue manager stopped

🎉 Quick test passed! Async training queue is working.
```

## 总结

通过将参数验证和数据加载移到异步队列检查之前，我们确保了：

1. **更早的错误发现**：无效参数在提交任务前就被发现
2. **正确的目标列传递**：异步任务收到准确的目标列信息
3. **更好的用户体验**：减少了无效任务和错误任务的数量
4. **系统稳定性**：避免了队列中的任务因参数问题而失败

现在异步训练队列系统更加健壮和可靠！