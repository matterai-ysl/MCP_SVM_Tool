# 异步训练队列使用说明

## 概述

本项目已成功实现了异步训练队列功能，将原来的同步超参数优化过程改为异步执行。这避免了长时间的训练任务阻塞线程，并支持并发训练多个模型。

## 主要特性

### 1. 异步训练队列
- **并发控制**: 支持最多3个并发训练任务
- **任务管理**: 完整的任务生命周期管理（排队、运行、完成、失败、取消）
- **持久化**: 任务状态保存到磁盘，支持服务重启后恢复
- **进度监控**: 实时任务进度跟踪

### 2. MCP工具接口
- `submit_training_task` - 提交训练任务到队列
- `get_task_status` - 查询任务状态
- `list_training_tasks` - 列出所有训练任务（支持过滤）
- `cancel_training_task` - 取消训练任务
- `get_queue_status` - 获取队列状态信息

### 3. 兼容性
- 保持原有训练接口不变
- 添加`use_async_queue`参数选择是否使用异步队列
- 支持直接训练和队列训练两种模式

## 使用方法

### 方式1：通过MCP工具提交异步任务

```python
# 提交分类训练任务
result = await submit_training_task(
    task_type="svm_classification",
    data_source="path/to/data.csv",
    target_column="target_column_name",
    kernel="rbf",
    optimize_hyperparameters=True,
    n_trials=50,
    cv_folds=5,
    user_id="user123"
)

task_id = result['task_id']
print(f"任务已提交: {task_id}")

# 监控任务状态
while True:
    status = await get_task_status(task_id)
    print(f"任务状态: {status['status']} ({status['progress']}%)")
    
    if status['status'] in ['completed', 'failed', 'cancelled']:
        break
    
    await asyncio.sleep(5)
```

### 方式2：通过原有接口使用异步队列

```python
# 分类任务
result = await train_svm_classifier(
    data_source="path/to/data.csv",
    kernel="rbf",
    optimize_hyperparameters=True,
    n_trials=50,
    use_async_queue=True,  # 启用异步队列
    user_id="user123"
)

# 回归任务
result = await train_svm_regressor(
    data_source="path/to/data.csv",
    target_dimension=1,
    kernel="linear",
    optimize_hyperparameters=True,
    n_trials=30,
    use_async_queue=True,  # 启用异步队列
    user_id="user123"
)
```

### 方式3：直接使用队列管理器

```python
from src.mcp_svm_tool.training_queue import get_queue_manager, initialize_queue_manager

# 初始化队列管理器
queue_manager = await initialize_queue_manager()

# 提交任务
task_params = {
    'data_source': 'data.csv',
    'target_column': 'target',
    'kernel': 'rbf',
    'optimize_hyperparameters': True,
    'n_trials': 20
}

task_id = await queue_manager.submit_task(
    task_type="svm_classification",
    params=task_params,
    user_id="user123"
)

# 查询状态
status = await queue_manager.get_task_status(task_id)
print(status)
```

## 队列管理

### 查看队列状态
```python
status = await get_queue_status()
print(f"总任务数: {status['total_tasks']}")
print(f"运行中: {status['running_tasks']}")
print(f"排队中: {status['queued_tasks']}")
```

### 列出所有任务
```python
# 列出所有任务
tasks = await list_training_tasks()

# 按用户过滤
user_tasks = await list_training_tasks(user_id="user123")

# 按状态过滤
completed_tasks = await list_training_tasks(status="completed")
```

### 取消任务
```python
result = await cancel_training_task(task_id)
if result['success']:
    print(f"任务 {task_id} 已取消")
```

## 技术实现

### 核心组件

1. **TrainingQueueManager** (`training_queue.py`)
   - 异步队列管理
   - 并发控制（信号量）
   - 任务持久化

2. **MCP工具接口** (`mcp_server.py`)
   - 队列操作的MCP工具封装
   - 统一的API接口

3. **训练引擎集成** (`training.py`)
   - 修复变量作用域问题
   - 确保异步执行兼容性

### 任务状态流转

```
QUEUED → RUNNING → COMPLETED/FAILED/CANCELLED
```

### 并发控制

- 默认最多3个并发任务
- 使用asyncio.Semaphore控制并发
- 任务在线程池中执行以避免阻塞

## 测试验证

运行测试脚本验证功能：

```bash
# 快速测试
python quick_test_async.py

# 完整测试
python test_async_training.py
```

## 目录结构

```
queue/                          # 任务持久化目录
├── task_id_1.json             # 任务状态文件
├── task_id_2.json
└── ...

trained_models/                 # 训练结果目录
├── model_id_1/                # 模型目录
│   ├── model.pkl              # 训练好的模型
│   ├── metadata.json          # 模型元数据
│   └── reports/               # 训练报告
└── ...
```

## 注意事项

1. **资源管理**: 建议根据系统资源调整最大并发任务数
2. **磁盘空间**: 训练结果会保存到磁盘，注意空间管理
3. **错误处理**: 失败任务的错误信息会记录在任务状态中
4. **服务重启**: 队列管理器重启后会从磁盘恢复任务状态

## 性能优化建议

1. **减少trial数量**: 对于快速测试，可以设置较小的`n_trials`值
2. **关闭数据验证**: 设置`validate_data=False`可以加快训练速度
3. **选择合适的kernel**: `linear`核通常比`rbf`核训练更快
4. **调整CV fold**: 减少`cv_folds`可以加快交叉验证速度

---

通过以上实现，成功将SVM训练的超参数优化过程从同步改为异步，提供了完整的队列管理和任务监控功能。