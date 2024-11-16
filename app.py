# Core System Components

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4
# import aiohttp
# import websockets
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
import time
import uvicorn


# --- Data Models ---

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"

class ResourceConstraints(BaseModel):
    cpu_percent: float = 0.75
    memory_mb: int = 2048
    disk_mb: int = 1024

class Task(BaseModel):
    id: UUID
    intent: str
    constraints: ResourceConstraints
    status: TaskStatus
    created_at: float
    updated_at: float

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"

class CreateTaskRequest(BaseModel):
    intent: str
    constraints: ResourceConstraints

# --- Base Agent Interface ---

class BaseAgent(ABC):
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.id = agent_id
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self._max_concurrent_tasks = 1

    @abstractmethod
    async def execute(self, task: Task) -> dict:
        pass

    @abstractmethod
    async def validate_preconditions(self, task: Task) -> bool:
        pass

# --- Agent Implementation ---

class TaskExecutorAgent(BaseAgent):
    def __init__(self, agent_id: str, capabilities: List[str]):
        super().__init__(agent_id, capabilities)
        self.current_tasks: Dict[UUID, Task] = {}

    async def execute(self, task: Task) -> dict:
        if len(self.current_tasks) >= self._max_concurrent_tasks:
            raise RuntimeError("Agent at maximum capacity")

        self.current_tasks[task.id] = task
        self.status = AgentStatus.BUSY

        try:
            # Simulate task execution
            await asyncio.sleep(1)
            result = {"status": "success", "data": f"Completed task {task.id}"}
            self.status = AgentStatus.IDLE
            return result
        except Exception as e:
            self.status = AgentStatus.ERROR
            raise e
        finally:
            del self.current_tasks[task.id]

    async def validate_preconditions(self, task: Task) -> bool:
        return all(cap in self.capabilities for cap in task.required_capabilities)

# --- Orchestrator ---

class AgentOrchestrator:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: Dict[UUID, Task] = {}
        self.event_bus = EventBus()

    async def register_agent(self, agent: BaseAgent):
        self.agents[agent.id] = agent
        await self.event_bus.publish("agent.registered", {"agent_id": agent.id})

    async def submit_task(self, intent: str, constraints: ResourceConstraints) -> Task:
        task = Task(
            id=uuid4(),
            intent=intent,
            constraints=constraints,
            status=TaskStatus.PENDING,
            created_at=time.time(),
            updated_at=time.time()
        )
        self.tasks[task.id] = task
        await self.event_bus.publish("task.submitted", {"task_id": str(task.id)})
        
        # Schedule task execution
        asyncio.create_task(self._execute_task(task))
        return task

    async def _execute_task(self, task: Task):
        try:
            # Find suitable agent
            agent = self._find_suitable_agent(task)
            if not agent:
                raise RuntimeError("No suitable agent found")

            # Update task status
            task.status = TaskStatus.RUNNING
            task.updated_at = time.time()
            await self.event_bus.publish("task.started", {"task_id": str(task.id)})

            # Execute task
            result = await agent.execute(task)

            # Update task status
            task.status = TaskStatus.COMPLETED
            task.updated_at = time.time()
            await self.event_bus.publish("task.completed", {
                "task_id": str(task.id),
                "result": result
            })

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.updated_at = time.time()
            await self.event_bus.publish("task.failed", {
                "task_id": str(task.id),
                "error": str(e)
            })

    def _find_suitable_agent(self, task: Task) -> Optional[BaseAgent]:
        for agent in self.agents.values():
            if (agent.status == AgentStatus.IDLE and 
                agent.validate_preconditions(task)):
                return agent
        return None

# --- Event System ---

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, Set[callable]] = {}

    async def publish(self, event: str, data: dict):
        if event in self._subscribers:
            for callback in self._subscribers[event]:
                await callback(data)

    def subscribe(self, event: str, callback: callable):
        if event not in self._subscribers:
            self._subscribers[event] = set()
        self._subscribers[event].add(callback)

    def unsubscribe(self, event: str, callback: callable):
        if event in self._subscribers:
            self._subscribers[event].remove(callback)

# --- Resource Management ---

class ResourceManager:
    def __init__(self):
        self.allocated_resources: Dict[UUID, ResourceConstraints] = {}

    async def allocate_resources(self, task_id: UUID, constraints: ResourceConstraints) -> bool:
        # Check if we have enough resources
        if not self._check_resource_availability(constraints):
            return False

        self.allocated_resources[task_id] = constraints
        return True

    async def release_resources(self, task_id: UUID):
        if task_id in self.allocated_resources:
            del self.allocated_resources[task_id]

    def _check_resource_availability(self, constraints: ResourceConstraints) -> bool:
        # Implement resource checking logic
        return True  # Simplified for example

# --- API Gateway ---

class APIGateway:
    def __init__(self):
        self.app = FastAPI()
        self.orchestrator = AgentOrchestrator()
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/tasks")
        async def create_task(request: CreateTaskRequest):
            task = await self.orchestrator.submit_task(request.intent, request.constraints)
            return {"task_id": str(task.id)}

        @self.app.get("/tasks/{task_id}")
        async def get_task(task_id: UUID):
            task = self.orchestrator.tasks.get(task_id)
            if task:
                return task
            return {"error": "Task not found"}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            
            async def send_update(data: dict):
                await websocket.send_json(data)
            
            # Subscribe to all task events
            self.orchestrator.event_bus.subscribe("task.*", send_update)
            
            try:
                while True:
                    await websocket.receive_text()
            except:
                self.orchestrator.event_bus.unsubscribe("task.*", send_update)

# --- Application Integration Example ---

class ExcelIntegration:
    def __init__(self):
        self.capabilities = ["excel.read", "excel.write", "excel.analyze"]

    async def register_with_system(self, gateway: APIGateway):
        agent = TaskExecutorAgent(
            agent_id="excel_agent",
            capabilities=self.capabilities
        )
        await gateway.orchestrator.register_agent(agent)

    async def handle_excel_operation(self, operation: str, data: dict) -> dict:
        # Implement Excel operations
        print('Excel operation:', operation)    
        
        return {"status": "success", "result": f"Executed {operation}"}

# --- System Bootstrap ---

class AISystem:
    def __init__(self):
        self.gateway = APIGateway()
        self.resource_manager = ResourceManager()
        self._integrations = []

    async def start(self):
        # Initialize system components
        logging.info("Starting AI System...")
        
        # Register integrations
        excel_integration = ExcelIntegration()
        await excel_integration.register_with_system(self.gateway)
        self._integrations.append(excel_integration)

        # Start API gateway
        uvicorn.run(self.gateway.app, host="0.0.0.0", port=8000)

# Initialize the FastAPI app
api_gateway = APIGateway()
app = api_gateway.app

# --- Usage Example ---

async def main():
    # Initialize and start the system
    system = AISystem()
    await system.start()

    # Submit a task
    task = await system.gateway.orchestrator.submit_task(
        intent="Analyze sales data in Excel and create summary",
        constraints=ResourceConstraints(
            cpu_percent=0.5,
            memory_mb=1024
        )
    )

    # Monitor task status
    while task.status != TaskStatus.COMPLETED:
        await asyncio.sleep(0.1)

    print(f"Task completed: {task.id}")

if __name__ == "__main__":
    asyncio.run(main())