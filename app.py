# Core System Components

import asyncio
import json
import logging
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any  # Add Any to imports
from uuid import UUID, uuid4
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
import time
import uvicorn

load_dotenv()

groq_key = os.getenv("GROQ_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")


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
    data: Optional[Dict[str, Any]] = None  # Add data field

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

            # Execute task through integration manager
            integration_manager = self.orchestrator.integration_manager
            if "excel.analyze" in task.required_capabilities:
                result = await integration_manager.handle_integration_operation("excel", "analyze", task.data)
            elif "email.send" in task.required_capabilities:
                result = await integration_manager.handle_integration_operation("email", "send", dependency_results)
            elif "slack.post" in task.required_capabilities:
                result = await integration_manager.handle_integration_operation("slack", "post", dependency_results)
            elif "reminder.set" in task.required_capabilities:
                result = await integration_manager.handle_integration_operation("reminder", "set", dependency_results)
            elif any(cap.startswith("postgres.") for cap in task.required_capabilities):
                operation = next(cap.split(".")[1] for cap in task.required_capabilities if cap.startswith("postgres."))
                result = await integration_manager.handle_integration_operation("postgres", operation, task.data)
            else:
                raise ValueError("No matching capabilities found")

            self.status = AgentStatus.IDLE
            return result
        except Exception as e:
            self.status = AgentStatus.ERROR
            raise e
        finally:
            del self.current_tasks[task.id]


# --- Orchestrator ---
class AgentOrchestrator:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: Dict[UUID, Task] = {}
        self.event_bus = EventBus()
        self.pending_tasks: Dict[UUID, Task] = {}  # Tasks waiting for dependencies
        self.integration_manager = IntegrationManager()  

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
        self.app = FastAPI(
            title="AI Native System",
            description="An AI-native system for intelligent task orchestration",
            version="0.1.0"
        )
        self.orchestrator = AgentOrchestrator()
        self.llm_coordinator = LLMCoordinator(self.orchestrator)
        self._setup_routes()
        self._setup_events()  

    async def _register_integrations(self):
        # Register default integrations
        integrations = [
            ExcelIntegration(),
            EmailIntegration(),
            SlackIntegration(),
            ReminderIntegration(),
            PostgresIntegration()  
        ]
        for integration in integrations:
            self.orchestrator.integration_manager.register_integration(integration)
            await integration.register_with_system(self)

    def _setup_events(self):
        @self.app.on_event("startup")
        async def startup():
            await self._register_integrations()
            logging.basicConfig(level=logging.INFO)
            logging.info("System initialized with all integrations")

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

        '''
        Enables building responsive applications that can react to task status changes 
        in real-time, without polling the server for updates.
        '''
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

    def _setup_events(self):
        @self.app.on_event("startup")
        async def startup():
            # Register integrations
            await self._register_integrations()
            logging.basicConfig(level=logging.INFO)
            logging.info("System initialized with all integrations")

# --- Application Integration Examples ---
class Integration(ABC):
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities

    @abstractmethod
    async def register_with_system(self, gateway: 'APIGateway'):
        pass

    @abstractmethod
    async def handle_operation(self, operation: str, data: dict) -> dict:
        pass

class ExcelIntegration(Integration):
    def __init__(self):
        super().__init__("Excel", ["excel.read", "excel.write", "excel.analyze"])

    async def register_with_system(self, gateway: APIGateway):
        agent = TaskExecutorAgent(
            agent_id=f"{self.name.lower()}_agent",
            capabilities=self.capabilities
        )
        agent.orchestrator = gateway.orchestrator
        await gateway.orchestrator.register_agent(agent)

    async def _simulate_llm_excel_analysis(self) -> dict:
        """
        Simulate LLM analyzing Excel data and generating insights.
        """
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
        

    async def handle_operation(self, operation: str, data: dict) -> dict:
        """
        Handle specific Excel operations.
        :param operation: The type of operation to perform (e.g., "read", "write", "analyze").
        :param data: The data required for the operation.
        :return: A dictionary with the operation results.
        """
        if operation == "analyze":
            # Use the rich simulation logic
            analysis_result = await self._simulate_llm_excel_analysis()
            return TaskResult(
                content=analysis_result,
                format="json",
                metadata={"type": "excel_analysis"},
                data=data
            )
        elif operation == "read":
            return TaskResult(
                content={"status": "success", "data": "Excel data read successfully"},
                format="json",
                metadata={"type": "excel_read"}
            )
        elif operation == "write":
            return TaskResult(
                content={"status": "success", "data": "Excel data written successfully"},
                format="json",
                metadata={"type": "excel_write"}
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
class SlackIntegration(Integration):
    def __init__(self):
        super().__init__("Slack", ["slack.post", "slack.read", "slack.reply"])

    async def register_with_system(self, gateway: APIGateway):
        agent = TaskExecutorAgent(
            agent_id=f"{self.name.lower()}_agent",
            capabilities=self.capabilities
        )
        agent.orchestrator = gateway.orchestrator
        await gateway.orchestrator.register_agent(agent)
        
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
        
    async def handle_operation(self, operation: str, data: dict) -> dict:
        # Dummy implementation
        
        raw_results = [result.content for result in data.values()]
                
        operation_result = await self._simulate_llm_slack_formatting(raw_results)
        return TaskResult(
                content=operation_result,
                format="json",
                metadata={"type": "email_content"},
                data=data
            )        
        
class EmailIntegration(Integration):
    def __init__(self):
        super().__init__("Email", ["email.send", "email.read", "email.attach"])

    async def register_with_system(self, gateway: APIGateway):
        agent = TaskExecutorAgent(
            agent_id=f"{self.name.lower()}_agent",
            capabilities=self.capabilities
        )
        agent.orchestrator = gateway.orchestrator
        await gateway.orchestrator.register_agent(agent)
                
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
        

    async def handle_operation(self, operation: str, data: dict) -> dict:
        # Dummy implementation
        
        raw_results = [result.content for result in data.values()]
                
        operation_result = await self._simulate_llm_email_formatting(raw_results)
        return TaskResult(
                content=operation_result,
                format="json",
                metadata={"type": "email_content"},
                data=data
            )        

class ReminderIntegration(Integration):
    def __init__(self):
        super().__init__("Reminder", ["reminder.set", "reminder.get", "reminder.list", "reminder.delete"])

    async def register_with_system(self, gateway: APIGateway):
        agent = TaskExecutorAgent(
            agent_id=f"{self.name.lower()}_agent",
            capabilities=self.capabilities
        )
        agent.orchestrator = gateway.orchestrator
        await gateway.orchestrator.register_agent(agent)
        
    async def _simulate_llm_reminder_formatting(self, raw_results: List[dict]) -> dict:
        """
        Simulate LLM analyzing raw results and formatting them for email.
        In reality, this would call an LLM API with a prompt like:
        "Format the following analysis results into a professional email: {raw_results}"
        """
        # This is just a simulation of what the LLM would do
        combined_data = raw_results[0]  # In reality, LLM would analyze all results
        
        reminder_content = {
            f"{combined_data.get('summary', 'No summary available')}\n\n"
        }
        
        return reminder_content

    async def handle_operation(self, operation: str, data: dict) -> dict:
        """
        Handle specific reminder operations.
        :param operation: The type of operation to perform (e.g., "set", "get").
        :param data: The data required for the operation.
        :return: A TaskResult with the operation results.
        """
        
        raw_results = [result.content for result in data.values()]
        operation_result = await self._simulate_llm_reminder_formatting(raw_results)
        
        if operation == "set":
            reminder_data = {
                "status": "success",
                'content': operation_result,
                "reminder_id": str(uuid4()),
                "message": "Reminder set successfully",
                "time": time.time() + 3600  # Set for 1 hour from now
            }
            return TaskResult(
                content=reminder_data,
                format="json",
                metadata={"type": "reminder_set"}
            )
        elif operation == "get":
            reminder_data = {
                "status": "success",
                "reminders": [
                    {"id": str(uuid4()), "message": "Sample reminder", "time": time.time() + 3600}
                ]
            }
            return TaskResult(
                content=reminder_data,
                format="json",
                metadata={"type": "reminder_get"}
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")

class PostgresIntegration(Integration):
    def __init__(self):
        super().__init__("Postgres", [
            "postgres.query",
            "postgres.insert",
            "postgres.update",
            "postgres.delete"
        ])

    async def register_with_system(self, gateway: APIGateway):
        agent = TaskExecutorAgent(
            agent_id=f"{self.name.lower()}_agent",
            capabilities=self.capabilities
        )
        agent.orchestrator = gateway.orchestrator
        await gateway.orchestrator.register_agent(agent)

    async def handle_operation(self, operation: str, data: dict) -> dict:
        """
        Handle PostgreSQL database operations.
        """
        if operation == "query":
            print('Executing PostgreSQL query:', data)
            result = {"status": "success", "data": "Query executed successfully"}
        elif operation == "insert":
            print('Inserting data:', data)
            result = {"status": "success", "data": "Data inserted successfully"}
        elif operation == "update":
            print('Updating data:', data)
            result = {"status": "success", "data": "Data updated successfully"}
        elif operation == "delete":
            print('Deleting data:', data)
            result = {"status": "success", "data": "Data deleted successfully"}
        else:
            result = {"status": "error", "message": f"Unknown operation: {operation}"}
        
        return result

class LLMCoordinator:
    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator

    async def analyze_intent(self, user_intent: str) -> List[Task]:
        """
        Use the LLM to analyze the user's intent and generate tasks with dependencies.
        """
        # Call LLM (simulated) to generate a plan based on the user intent
        plan = await self._call_llm_to_generate_plan(user_intent)

        tasks = []
        task_name_to_id = {}

        # Create Task objects based on the plan
        for task_info in plan['tasks']:
            task = Task(
                id=uuid4(),
                intent=task_info['intent'],
                constraints=ResourceConstraints(),
                status=TaskStatus.PENDING,
                created_at=time.time(),
                updated_at=time.time(),
                required_capabilities=task_info['capabilities'],
                dependencies=[]  # Dependencies will be set after all tasks are created
            )
            tasks.append(task)
            task_name_to_id[task_info['name']] = task.id

        # Set dependencies using task IDs
        for task, task_info in zip(tasks, plan['tasks']):
            if 'dependencies' in task_info:
                task.dependencies = [task_name_to_id[dep_name] for dep_name in task_info['dependencies']]

        return tasks

    async def _call_llm_to_generate_plan(self, user_intent: str) -> Dict[str, Any]:
        """
        Simulate an LLM call to generate a plan of tasks with their capabilities and dependencies.
        In reality, this would:
        1. Get available capabilities from integration manager
        2. Ask LLM to analyze intent and match with available capabilities
        3. LLM would determine optimal task sequence and dependencies
        """
        # Get all available capabilities from registered integrations
        available_integrations = self.orchestrator.integration_manager.get_all_capabilities()
        logging.info(f"Available integrations and capabilities: {available_integrations}")

        # Here we would send to LLM:
        # 1. The user's intent
        # 2. Available integrations and their capabilities
        # 3. Ask it to create a plan using only available capabilities
        
        # Example prompt template:
        prompt = f"""
        User Intent: {user_intent}
        Available Integrations: {json.dumps(available_integrations, indent=2)}
        
        Create a plan using only the available capabilities.
        Each task must use capabilities that exist in the system.
        Determine which tasks depend on other tasks.
        """
        
        logging.info(prompt)
        
        # For now, we'll simulate the LLM's response
        plan = {"tasks": []}
        intent_lower = user_intent.lower()

        # Check if requested operations match available capabilities
        if "excel" in intent_lower and "analyze" in intent_lower:
            if "excel.analyze" in available_integrations.get("excel", []):
                plan['tasks'].append({
                    "name": "excel_analysis",
                    "intent": "Process Excel data",
                    "capabilities": ["excel.analyze"]
                })

        # Add reminder task if capability exists
        if "reminder" in intent_lower and "reminder.set" in available_integrations.get("reminder", []):
            dependencies = ["excel_analysis"] if any(task['name'] == "excel_analysis" for task in plan['tasks']) else []
            plan['tasks'].append({
                "name": "set_reminder",
                "intent": "Set a reminder for follow-up",
                "capabilities": ["reminder.set"],
                "dependencies": dependencies
            })

        # Only add email task if capability exists
        if "email" in intent_lower and "email.send" in available_integrations.get("email", []):
            dependencies = ["excel_analysis"] if any(task['name'] == "excel_analysis" for task in plan['tasks']) else []
            plan['tasks'].append({
                "name": "send_email",
                "intent": "Send an email with the results",
                "capabilities": ["email.send"],
                "dependencies": dependencies
            })

        # Only add slack task if capability exists
        if "slack" in intent_lower and "slack.post" in available_integrations.get("slack", []):
            dependencies = ["excel_analysis"] if any(task['name'] == "excel_analysis" for task in plan['tasks']) else []
            plan['tasks'].append({
                "name": "post_slack",
                "intent": "Post the results to Slack",
                "capabilities": ["slack.post"],
                "dependencies": dependencies
            })

        # Add PostgreSQL task if capability exists
        if any(word in intent_lower for word in ["database", "query", "sql", "postgres"]):
            operation = "query"  # Default to query, but could be determined by intent analysis
            if "postgres." + operation in available_integrations.get("postgres", []):
                plan['tasks'].append({
                    "name": "database_operation",
                    "intent": f"Execute PostgreSQL {operation}",
                    "capabilities": [f"postgres.{operation}"],
                    "dependencies": []
                })

        if len(plan['tasks']) == 0:
            logging.warning(f"No matching capabilities found for intent: {user_intent}")
            raise ValueError("No available integrations can handle this request")

        logging.info(f"Generated plan: {json.dumps(plan, indent=2)}")
        return plan

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

class IntegrationManager:
    def __init__(self):
        self.integrations: Dict[str, Integration] = {}

    def register_integration(self, integration: Integration):
        self.integrations[integration.name.lower()] = integration

    def get_integration(self, name: str) -> Optional[Integration]:
        return self.integrations.get(name.lower())

    def get_all_capabilities(self) -> Dict[str, List[str]]:
        return {name: integration.capabilities for name, integration in self.integrations.items()}
    
    async def handle_integration_operation(self, name: str, operation: str, data: dict) -> dict:
        integration = self.get_integration(name)
        if integration:
            return await integration.handle_operation(operation, data)
        else:
            raise ValueError(f"Integration {name} not found")

# --- System Bootstrap ---
class AISystem:
    def __init__(self):
        self.gateway = APIGateway()
        self.resource_manager = ResourceManager()

    async def initialize(self):
        """Initialize all system components"""
        self.gateway._setup_events()  
        logging.basicConfig(level=logging.INFO)
        logging.info("System initialized")

    def start(self):
        """Start the system"""
        logging.info("Starting AI System...")
        uvicorn.run(self.gateway.app, host="0.0.0.0", port=8000)

# Create single system instance
ai_system = AISystem()
app = ai_system.gateway.app

@app.on_event("startup")
async def startup_event():
    await ai_system.initialize()

@app.get("/test")
async def test_system():
    return await ai_system.gateway.llm_coordinator.process_intent(
        "Please analyze excel data, email the results and post to slack and set a reminder"
        # "Please read from the database 100 random records from the locations table"
    )

if __name__ == "__main__":
    ai_system.start()
