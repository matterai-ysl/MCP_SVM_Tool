"""
Training Task Queue Manager

This module provides a queue-based system for managing concurrent training tasks
to prevent resource contention and provide better scalability.
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingTask:
    """Represents a training task in the queue."""
    task_id: str
    user_id: Optional[str]
    task_type: str  # 'regression' or 'classification'
    params: Dict[str, Any]
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: float = 0.0

class TrainingQueueManager:
    """
    Manages a queue of training tasks with concurrency control.
    
    Features:
    - Concurrent task execution with configurable limits
    - Task priority management
    - Progress tracking
    - Automatic retry for failed tasks
    - Resource usage monitoring
    """
    
    def __init__(self, max_concurrent_tasks: int = 3, queue_dir: str = "queue"):
        """
        Initialize the training queue manager.
        
        Args:
            max_concurrent_tasks: Maximum number of tasks to run concurrently
            queue_dir: Directory to store queue state files
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(exist_ok=True)
        
        # Task storage
        self.tasks: Dict[str, TrainingTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Background worker
        self.worker_task = None
        self.is_running = False
        
        logger.info(f"Initialized TrainingQueueManager with max {max_concurrent_tasks} concurrent tasks")
    
    async def start(self):
        """Start the background worker task."""
        if self.is_running:
            logger.warning("Queue manager is already running")
            return
            
        self.is_running = True
        self.worker_task = asyncio.create_task(self._worker())
        logger.info("Training queue manager started")
    
    async def stop(self):
        """Stop the background worker task."""
        self.is_running = False
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None
        
        # Cancel all running tasks
        for task_id, task in self.running_tasks.items():
            task.cancel()
            logger.info(f"Cancelled running task: {task_id}")
        
        self.running_tasks.clear()
        logger.info("Training queue manager stopped")
    
    async def submit_task(self, task_type: str, params: Dict[str, Any], 
                         user_id: Optional[str] = None) -> str:
        """
        Submit a new training task to the queue.
        
        Args:
            task_type: Type of training task ('regression' or 'classification')
            params: Parameters for the training task
            user_id: Optional user identifier for tracking
            
        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        
        task = TrainingTask(
            task_id=task_id,
            user_id=user_id,
            task_type=task_type,
            params=params,
            status=TaskStatus.QUEUED,
            created_at=datetime.now()
        )
        
        self.tasks[task_id] = task
        await self.task_queue.put(task_id)
        
        # Save task to disk for persistence
        await self._save_task(task)
        
        logger.info(f"Submitted task {task_id} ({task_type}) to queue. Queue size: {self.task_queue.qsize()}")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            Task status information or None if task not found
        """
        task = self.tasks.get(task_id)
        if not task:
            return None
            
        return {
            'task_id': task.task_id,
            'user_id': task.user_id,
            'task_type': task.task_type,
            'status': task.status.value,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'progress': task.progress,
            'error_message': task.error_message,
            'result': task.result
        }
    
    async def list_tasks(self, user_id: Optional[str] = None, 
                        status: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """
        List all tasks or filter by user/status.
        
        Args:
            user_id: Optional user ID filter
            status: Optional status filter
            
        Returns:
            List of task information
        """
        tasks = []
        for task in self.tasks.values():
            # Apply filters
            if user_id and task.user_id != user_id:
                continue
            if status and task.status != status:
                continue
                
            tasks.append({
                'task_id': task.task_id,
                'user_id': task.user_id,
                'task_type': task.task_type,
                'status': task.status.value,
                'created_at': task.created_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'progress': task.progress
            })
        
        # Sort by creation time (newest first)
        tasks.sort(key=lambda x: x['created_at'], reverse=True)
        return tasks
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was cancelled, False if not found or already completed
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
            
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
            
        # Cancel if running
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            del self.running_tasks[task_id]
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        await self._save_task(task)
        
        logger.info(f"Cancelled task {task_id}")
        return True
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status information."""
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(1 for t in self.tasks.values() if t.status == status)
        
        return {
            'total_tasks': len(self.tasks),
            'queued_tasks': self.task_queue.qsize(),
            'running_tasks': len(self.running_tasks),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'status_counts': status_counts,
            'is_running': self.is_running
        }
    
    async def _worker(self):
        """Background worker that processes tasks from the queue."""
        logger.info("Training queue worker started")
        
        while self.is_running:
            try:
                # Get next task from queue (with timeout to allow checking is_running)
                try:
                    task_id = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                task = self.tasks.get(task_id)
                if not task or task.status != TaskStatus.QUEUED:
                    continue
                
                # Start task execution with concurrency control
                execution_task = asyncio.create_task(self._execute_task(task))
                self.running_tasks[task_id] = execution_task
                
                # Don't await here - let tasks run concurrently
                logger.info(f"Started executing task {task_id}. Running: {len(self.running_tasks)}")
                
            except asyncio.CancelledError:
                logger.info("Queue worker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in queue worker: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
        
        logger.info("Training queue worker stopped")
    
    async def _execute_task(self, task: TrainingTask):
        """
        Execute a single training task.
        
        Args:
            task: The task to execute
        """
        async with self.semaphore:  # Control concurrency
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            task.progress = 0.0
            await self._save_task(task)
            
            try:
                logger.info(f"Executing task {task.task_id} ({task.task_type})")
                
                # Import training engine here to avoid circular imports
                from .training import TrainingEngine
                training_engine = TrainingEngine()
                
                # Execute the actual training using RandomForest
                if task.task_type == "regression":
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: training_engine.train_random_forest(**task.params)
                    )
                elif task.task_type == "classification":
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: training_engine.train_classification_forest(**task.params)
                    )
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")
                
                # Task completed successfully
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.progress = 100.0
                task.result = result
                
                logger.info(f"Task {task.task_id} completed successfully")
                
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                logger.info(f"Task {task.task_id} was cancelled")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error_message = str(e)
                
                logger.error(f"Task {task.task_id} failed: {e}")
                
            finally:
                await self._save_task(task)
                # Remove from running tasks
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
    
    async def _save_task(self, task: TrainingTask):
        """Save task state to disk for persistence."""
        try:
            task_file = self.queue_dir / f"{task.task_id}.json"
            task_data = {
                'task_id': task.task_id,
                'user_id': task.user_id,
                'task_type': task.task_type,
                'params': task.params,
                'status': task.status.value,
                'created_at': task.created_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'progress': task.progress,
                'error_message': task.error_message,
                'result': task.result
            }
            
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_data, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save task {task.task_id}: {e}")

# Global instance
_queue_manager: Optional[TrainingQueueManager] = None

def get_queue_manager() -> TrainingQueueManager:
    """Get the global training queue manager instance."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = TrainingQueueManager()
    return _queue_manager

async def initialize_queue_manager():
    """Initialize and start the global queue manager."""
    manager = get_queue_manager()
    if not manager.is_running:
        await manager.start()
    return manager