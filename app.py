# Core System Components

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any  # Add Any to imports
from uuid import UUID, uuid4
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

# Add TaskResult model
class TaskResult(BaseModel):
    content: Dict[str, Any]  # Change from any to Any
    format: str = "json"  # could be "text", "json", "binary", etc.
    metadata: Dict[str, str] = {}

    class Config:
        arbitrary_types_allowed = True

class Task(BaseModel):
    id: UUID
    intent: str
    constraints: ResourceConstraints
    status: TaskStatus
    created_at: float
    updated_at: float
    required_capabilities: List[str] = []
    dependencies: List[UUID] = []  # Add dependencies field
    result: Optional[TaskResult] = None  # Replace result_data with structured result

    class Config:
        arbitrary_types_allowed = True

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"

class CreateTaskRequest(BaseModel):
    intent: str
    constraints: ResourceConstraints

class UserIntentRequest(BaseModel):
    intent: str
    metadata: Dict[str, str] = {}  

    class Config:
        arbitrary_types_allowed = True  # If need to support arbitrary types

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
        self.orchestrator = None  # Will be set when registered with system

    async def validate_preconditions(self, task: Task) -> bool:
        # Check if agent has all required capabilities
        if not all(cap in self.capabilities for cap in task.required_capabilities):
            return False
            
        # Check if agent has capacity
        if len(self.current_tasks) >= self._max_concurrent_tasks:
            return False
            
        return True

    async def execute(self, task: Task) -> dict:
        if len(self.current_tasks) >= self._max_concurrent_tasks:
            raise RuntimeError("Agent at maximum capacity")

        self.current_tasks[task.id] = task
        self.status = AgentStatus.BUSY

        try:
            # Get results from dependencies if they exist
            dependency_results = {}
            for dep_id in task.dependencies:
                dep_task = self.orchestrator.tasks.get(dep_id)
                if dep_task and dep_task.result:
                    dependency_results[str(dep_id)] = dep_task.result

            # Execute task based on capability
            if "excel.analyze" in task.required_capabilities:
                result = await self._simulate_excel_analysis(dependency_results)
            elif "email.send" in task.required_capabilities:
                result = await self._execute_email_task(dependency_results)
            elif "slack.post" in task.required_capabilities:
                result = await self._execute_slack_task(dependency_results)

            self.status = AgentStatus.IDLE
            return result
        except Exception as e:
            self.status = AgentStatus.ERROR
            raise e
        finally:
            del self.current_tasks[task.id]

    async def _simulate_excel_analysis(self, dependency_results: Dict[str, TaskResult] = None) -> TaskResult:
        """
        Simulate Excel analysis with LLM-like functionality.
        In reality, this would:
        1. Use actual Excel files
        2. Call LLM to understand what analysis to perform
        3. Execute the analysis
        4. Have LLM summarize the results
        """
        # Simulate LLM analyzing Excel data
        simulated_analysis = await self._simulate_llm_excel_analysis()
        
        return TaskResult(
            content=simulated_analysis,
            format="json",
            metadata={"type": "excel_analysis"}
        )

    async def _simulate_llm_excel_analysis(self) -> dict:
        """
        Simulate LLM analyzing Excel data and generating insights.
        In reality, this would call an LLM API with a prompt like:
        "Analyze this Excel data and provide key insights: {excel_data}"
        """
        # Simulate finding patterns and insights in the data
        analysis_result = {
            "summary": "Q4 sales exceeded expectations with significant growth",
            "details": {
                "total_sales": 1234567,
                "growth_rate": 25.0,
                "top_product": "Widget X",
                "key_insights": [
                    "Sales grew 25% year-over-year",
                    "Widget X dominated with 45% market share",
                    "New customer acquisition up 30%"
                ],
                "recommendations": [
                    "Increase Widget X production",
                    "Expand marketing in high-growth regions"
                ]
            }
        }
        return analysis_result

    async def _execute_email_task(self, dependency_results: Dict[str, TaskResult]) -> TaskResult:
        # Get raw results from dependencies
        raw_results = [result.content for result in dependency_results.values()]
        
        # Simulate LLM call to analyze and format results for email
        # In reality, this would be an actual LLM API call
        formatted_content = await self._simulate_llm_email_formatting(raw_results)
        
        return TaskResult(
            content=formatted_content,
            format="json",
            metadata={"type": "email_content"}
        )

    async def _execute_slack_task(self, dependency_results: Dict[str, TaskResult]) -> TaskResult:
        # Get raw results from dependencies
        raw_results = [result.content for result in dependency_results.values()]
        
        # Simulate LLM call to analyze and format results for Slack
        # In reality, this would be an actual LLM API call
        formatted_content = await self._simulate_llm_slack_formatting(raw_results)
        
        return TaskResult(
            content=formatted_content,
            format="json",
            metadata={"type": "slack_message"}
        )

    async def _simulate_llm_email_formatting(self, raw_results: List[dict]) -> dict:
        """
        Simulate LLM analyzing raw results and formatting them for email.
        In reality, this would call an LLM API with a prompt like:
        "Format the following analysis results into a professional email: {raw_results}"
        """
        # This is just a simulation of what the LLM would do
        combined_data = raw_results[0]  # In reality, LLM would analyze all results
        
        email_content = {
            "subject": "Analysis Results",
            "body": (
                "Dear team,\n\n"
                "Based on the analysis results, here are the key findings:\n\n"
                f"{combined_data.get('summary', 'No summary available')}\n\n"
                "Key Details:\n"
            )
        }
        
        # Simulate LLM extracting and formatting relevant details
        if 'details' in combined_data:
            for key, value in combined_data['details'].items():
                email_content['body'] += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        email_content['body'] += "\nBest regards,\nAI Assistant"
        
        return email_content

    async def _simulate_llm_slack_formatting(self, raw_results: List[dict]) -> dict:
        """
        Simulate LLM analyzing raw results and formatting them for Slack.
        In reality, this would call an LLM API with a prompt like:
        "Format the following analysis results into a concise Slack message with appropriate formatting: {raw_results}"
        """
        # This is just a simulation of what the LLM would do
        combined_data = raw_results[0]  # In reality, LLM would analyze all results
        
        # Simulate LLM generating Slack blocks with appropriate formatting
        slack_message = {
            "channel": "#updates",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "📊 Analysis Results"
                    }
                }
            ]
        }
        
        # Simulate LLM deciding what's important and how to format it
        if 'summary' in combined_data:
            slack_message['blocks'].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Key Finding:*\n{combined_data['summary']}"
                }
            })
        
        # Simulate LLM organizing details into a clean format
        if 'details' in combined_data:
            details_text = "*Details:*\n"
            for key, value in combined_data['details'].items():
                details_text += f"• {key.replace('_', ' ').title()}: {value}\n"
            
            slack_message['blocks'].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": details_text
                }
            })
        
        return slack_message

# --- Orchestrator ---
class AgentOrchestrator:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: Dict[UUID, Task] = {}
        self.event_bus = EventBus()
        self.pending_tasks: Dict[UUID, Task] = {}  # Tasks waiting for dependencies

    async def register_agent(self, agent: BaseAgent):
        self.agents[agent.id] = agent
        await self.event_bus.publish("agent.registered", {"agent_id": agent.id})

    async def submit_task(self, task: Task) -> Task:
        """
        Submit an existing task to the orchestrator.
        """
        self.tasks[task.id] = task
        await self.event_bus.publish("task.submitted", {"task_id": str(task.id)})
        
        # Schedule task execution
        asyncio.create_task(self._execute_task(task))
        return task

    async def _execute_task(self, task: Task):
        logging.info(f"Executing task {task.id} with capabilities {task.required_capabilities}")
        logging.info(f"Task dependencies: {task.dependencies}")
        
        # Check if dependencies are met
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                logging.info(f"Task {task.id} waiting for dependency {dep_id}")
                self.pending_tasks[task.id] = task
                return

        try:
            # Find suitable agent
            agent = await self._find_suitable_agent(task)
            if not agent:
                logging.error(f"No suitable agent found for task {task.id}")
                raise RuntimeError(f"No suitable agent found for capabilities: {task.required_capabilities}")

            # Update task status
            task.status = TaskStatus.RUNNING
            task.updated_at = time.time()
            await self.event_bus.publish("task.started", {"task_id": str(task.id)})
            
            logging.info(f"Found agent {agent.id} for task {task.id}")

            # Execute task
            result = await agent.execute(task)

            # Store result in task
            task.result = result if isinstance(result, TaskResult) else TaskResult(
                content=result,
                format="json"
            )

            # Update task status
            task.status = TaskStatus.COMPLETED
            task.updated_at = time.time()
            logging.info(f"Task {task.id} completed")
            
            await self.event_bus.publish("task.completed", {
                "task_id": str(task.id),
                "result": task.result.dict()
            })

            # After task completes, check for dependent tasks
            pending_tasks_to_execute = []
            for pending_task in list(self.pending_tasks.values()):
                if all(self.tasks.get(dep_id).status == TaskStatus.COMPLETED 
                      for dep_id in pending_task.dependencies):
                    del self.pending_tasks[pending_task.id]
                    pending_tasks_to_execute.append(pending_task)

            # Execute any ready tasks
            for pending_task in pending_tasks_to_execute:
                asyncio.create_task(self._execute_task(pending_task))

        except Exception as e:
            logging.error(f"Task {task.id} failed: {str(e)}")
            task.status = TaskStatus.FAILED
            task.updated_at = time.time()
            await self.event_bus.publish("task.failed", {
                "task_id": str(task.id),
                "error": str(e)
            })

    async def _find_suitable_agent(self, task: Task) -> Optional[BaseAgent]:
        logging.info(f"Looking for agent with capabilities: {task.required_capabilities}")
        for agent in self.agents.values():
            if (agent.status == AgentStatus.IDLE and 
                await agent.validate_preconditions(task)):
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
        self.llm_coordinator = LLMCoordinator(self.orchestrator)
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

        @self.app.post("/process-intent")
        async def process_intent(request: UserIntentRequest):
            if not request.intent or len(request.intent.strip()) == 0:
                return {"error": "Intent cannot be empty"}
            return await self.llm_coordinator.process_intent(request.intent)

# --- Application Integration Example ---
class ExcelIntegration:
    def __init__(self):
        self.capabilities = ["excel.read", "excel.write", "excel.analyze"]

    async def register_with_system(self, gateway: APIGateway):
        agent = TaskExecutorAgent(
            agent_id="excel_agent",
            capabilities=self.capabilities
        )
        agent.orchestrator = gateway.orchestrator  # Set orchestrator reference
        await gateway.orchestrator.register_agent(agent)

    async def handle_excel_operation(self, operation: str, data: dict) -> dict:
        # Implement Excel operations
        print('Excel operation:', operation)    
        
        return {"status": "success", "result": f"Executed {operation}"}

class SlackIntegration:
    def __init__(self):
        self.capabilities = ["slack.post", "slack.read", "slack.reply"]

    async def register_with_system(self, gateway: APIGateway):
        agent = TaskExecutorAgent(
            agent_id="slack_agent",
            capabilities=self.capabilities
        )
        agent.orchestrator = gateway.orchestrator  # Add this line
        await gateway.orchestrator.register_agent(agent)

    async def handle_slack_operation(self, operation: str, data: dict) -> dict:
        # Dummy implementation
        print('Slack operation:', operation)
        return {"status": "success", "result": f"Executed {operation}"}

class EmailIntegration:
    def __init__(self):
        self.capabilities = ["email.send", "email.read", "email.attach"]

    async def register_with_system(self, gateway: APIGateway):
        agent = TaskExecutorAgent(
            agent_id="email_agent",
            capabilities=self.capabilities
        )
        agent.orchestrator = gateway.orchestrator  # Add this line
        await gateway.orchestrator.register_agent(agent)

    async def handle_email_operation(self, operation: str, data: dict) -> dict:
        # Dummy implementation
        print('Email operation:', operation)
        return {"status": "success", "result": f"Executed {operation}"}

class LLMCoordinator:
    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
        self.available_integrations = {
            "excel": ["excel.read", "excel.write", "excel.analyze"],
            "slack": ["slack.post", "slack.read", "slack.reply"],
            "email": ["email.send", "email.read", "email.attach"]
        }

    async def analyze_intent(self, user_intent: str) -> List[Task]:
        tasks = []
        user_intent = user_intent.lower()
        excel_task_id = None
        
        # Create Excel analysis task if needed
        if "excel" in user_intent:
            excel_task = Task(
                id=uuid4(),
                intent="Process Excel data",
                constraints=ResourceConstraints(),
                status=TaskStatus.PENDING,
                created_at=time.time(),
                updated_at=time.time(),
                required_capabilities=["excel.analyze"]
            )
            tasks.append(excel_task)
            excel_task_id = excel_task.id  # Store the ID separately
            logging.info(f"Created Excel task with ID: {excel_task_id}")

            # Create dependent tasks only if we have an Excel task ID
            if "email" in user_intent:
                email_task = Task(
                    id=uuid4(),
                    intent="Send email with analysis results",
                    constraints=ResourceConstraints(),
                    status=TaskStatus.PENDING,
                    created_at=time.time(),
                    updated_at=time.time(),
                    required_capabilities=["email.send"],
                    dependencies=[excel_task_id]
                )
                tasks.append(email_task)
                logging.info(f"Created Email task depending on Excel task: {excel_task_id}")

            if "slack" in user_intent:
                slack_task = Task(
                    id=uuid4(),
                    intent="Post analysis results to Slack",
                    constraints=ResourceConstraints(),
                    status=TaskStatus.PENDING,
                    created_at=time.time(),
                    updated_at=time.time(),
                    required_capabilities=["slack.post"],
                    dependencies=[excel_task_id]
                )
                tasks.append(slack_task)
                logging.info(f"Created Slack task depending on Excel task: {excel_task_id}")

            # Validate dependencies
            for task in tasks:
                if task.dependencies:
                    logging.info(f"Task {task.id} depends on: {task.dependencies}")

        return tasks

    async def process_intent(self, user_intent: str) -> dict:
        """
        Process a natural language intent from the user
        Example: "Please analyze the Q4 sales data in Excel and email a summary to the team"
        """
        tasks = await self.analyze_intent(user_intent)
        results = []
        
        for task in tasks:
            # Submit the existing task without creating a new one
            submitted_task = await self.orchestrator.submit_task(task)
            results.append(str(submitted_task.id))
            
        return {"message": "Intent processing started", "task_ids": results}

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

# Register an Excel agent for testing
@app.on_event("startup")
async def startup_event():
    # Register all integrations
    excel_integration = ExcelIntegration()
    slack_integration = SlackIntegration()
    email_integration = EmailIntegration()
    
    await excel_integration.register_with_system(api_gateway)
    await slack_integration.register_with_system(api_gateway)
    await email_integration.register_with_system(api_gateway)
    
    logging.basicConfig(level=logging.INFO)
    logging.info("System initialized with all integrations")

# Add a test endpoint
@app.get("/test")
async def test_system():
    return await api_gateway.llm_coordinator.process_intent(
        # "Please analyze excel data and email the results"        
        "Please analyze excel data, email the results and post to slack"
    )