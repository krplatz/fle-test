# In agents/alphagenius/alphagenius_agent.py

import io
import contextlib
import traceback
import asyncio
import builtins # For mock execution fallback

"""AlphaGenius agent implementation."""

# The agent can operate with any environment that exposes an ``eval`` method
# returning ``(score, goal, result)``. We keep the optional imports for the
# Factorio Learning Environment for backwards compatibility but do not rely on
# them at run time.
try:
    from env.src.instance import FactorioInstance as BaseGameInstance
    from env.src.gym.factorio_environment import FactorioEnv as BaseGameEnv
    EnvironmentType = (BaseGameInstance, BaseGameEnv)
except ImportError:  # Generic fallback when FLE is unavailable
    BaseGameInstance = None
    BaseGameEnv = None
    EnvironmentType = (type(None),)

from .llm_interface.llm_client import ConcreteLLMClient
from .world_model import WorldModel
from .skill_library import greet
try:
    from agents.utils.llm_factory import LLMFactory
except ImportError:
    print("Warning: Real LLMFactory not found. Using placeholder in AlphaGeniusAgent.")
    class LLMFactory: # Placeholder if the real one isn't found
        def __init__(self, model_name: str = "mock_model", beam: int = 1): 
            self.model = model_name; self.beam = beam
            print(f"AlphaGeniusAgent: Initialized PLACEHOLDER LLMFactory with model: {model_name}")
        
        async def acall(self, messages, model, max_tokens, temperature): # Mock
            print("AlphaGeniusAgent: Using PLACEHOLDER LLMFactory.acall()")
            
            system_prompt_content = ""
            user_prompt_content = ""
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        if msg.get('role') == 'system':
                            system_prompt_content = msg.get('content', "")
                        elif msg.get('role') == 'user':
                            user_prompt_content = msg.get('content', "")
            
            response_content_str = ""
            is_planning_call = "planning assistant" in system_prompt_content.lower()
            is_reflection_call = "debugging assistant" in system_prompt_content.lower()

            if is_planning_call:
                print("Placeholder LLMFactory: Detected call for PLANNING.")
                if "greeting" in user_prompt_content.lower():
                    response_content_str = "1. Print a greeting.\n2. Confirm completion."
                else:
                    response_content_str = "1. Inspect environment.\n2. Perform a simple action."
            elif is_reflection_call:
                print("Placeholder LLMFactory: Detected call for REFLECTION.")
                if "undefined_variable" in user_prompt_content:
                    response_content_str = (
                        "Explanation: The script used an undefined variable.\n"
                        "```python\nprint('Corrected execution')\nvalue = 42\nprint(value)\n```"
                    )
                else:
                    response_content_str = "Suggestion: Review the error message and adjust the script."
            else:  # Call is for CODE GENERATION
                print("Placeholder LLMFactory: Detected call for CODE GENERATION.")
                if "print a greeting" in user_prompt_content.lower():
                    response_content_str = "print('Hello from AlphaGenius!')"
                elif "cause an intentional error" in user_prompt_content.lower():
                    response_content_str = (
                        "print('Executing script designed to fail...')\n"
                        "undefined_variable = 1 / 0"
                    )
                else:
                    response_content_str = (
                        f"print('Mock Python script executed for sub-goal: {user_prompt_content[:50]}...')"
                    )
            
            class MockMessage:
                def __init__(self, content): self.content = content
            class MockChoice:
                def __init__(self, content): self.message = MockMessage(content)
            class MockResponse:
                def __init__(self, content): self.choices = [MockChoice(content)]
            
            print(f"Placeholder LLMFactory: Returning mock response content:\n{response_content_str}")
            return MockResponse(response_content_str)

from .prompts.initial_prompts import SIMPLE_TASK_PROMPTS
from .intrinsic_motivation.discovery_tracker import DiscoveryTracker
from .planning.planner import Planner
from .metacognition.meta_cognition import MetaCognition
from .memory.memory_module import MemoryModule

PLANNING_SYSTEM_PROMPT = (
    "You are an expert planning assistant for a learning agent operating in an arbitrary environment. "
    "Given a high level goal and the latest observations, break the goal into a short list of actionable sub-goals. "
    "Keep the sub-goals concise and ordered."
)

REFLECTION_SYSTEM_PROMPT = (
    "You are an expert debugging assistant for an autonomous agent. "
    "Analyse the provided error and suggest a concise correction or alternative approach. "
    "Structure your response with a brief explanation followed by a corrected script (if any) or a suggestion."
)

class AlphaGeniusAgent:
    def __init__(self, config=None, environment=None, model_name: str = "gpt-4o-mini", agent_idx: int = 0):
        # ... (self.config, self.environment, self.agent_idx setup)
        self.config = config; self.environment = environment; self.agent_idx = agent_idx
        
        self.discovery_tracker = DiscoveryTracker()
        self.memory_module = MemoryModule(filepath="alphagenius_memory.jsonl")
        self.world_model = WorldModel()

        # ... (system_prompt setup)
        alpha_genius_meta_prompt = (
            "You are AlphaGenius, an autonomous agent capable of learning in many environments. "
            "You have been given a specific sub-task by your planning module. "
            "Generate a concise Python script using the tools provided by the current environment to achieve this sub-task. "
            "Consider the previous output (stdout/stderr) when forming your script. "
            "If the prior attempt produced an error, adapt your script to overcome the issue or try a different approach. "
            "Strive for a solution that is efficient, correct and directly contributes to the sub-task.\n"
        )
        if self.environment and hasattr(self.environment, 'get_system_prompt'):
            fle_system_prompt = self.environment.get_system_prompt(self.agent_idx)
            self.system_prompt = alpha_genius_meta_prompt + "\nFLE Tool Documentation..." + fle_system_prompt
        else:
            self.system_prompt = alpha_genius_meta_prompt + "\nAvailable tools (mock environment)..."
        
        print("AlphaGeniusAgent initialized.")
        greet()  # demonstrate a basic skill call
        # ... (LLMFactory, ConcreteLLMClient, Planner, MetaCognition instantiation as before)
        try: actual_llm_factory = LLMFactory(model=model_name)
        except TypeError: actual_llm_factory = LLMFactory()
        self.llm_client = ConcreteLLMClient(llm_factory=actual_llm_factory, model_name=model_name, default_system_prompt=self.system_prompt) # Pass default_system_prompt
        self.planner = Planner(llm_client=self.llm_client, system_prompt_for_planning=PLANNING_SYSTEM_PROMPT)
        self.metacognition = MetaCognition(llm_client=self.llm_client) # system_prompt_for_reflection is passed at call time

        # ... (rest of __init__ as before)
        print(f"Using system prompt for code generation (first 100 chars): {self.system_prompt[:100]}...")
        print(f"LLM Client: {type(self.llm_client).__name__} initialized...")
        print("Hint: To use a local LLM (e.g., via LM Studio)...")
        print("Planner module initialized.")
        print("MetaCognition module initialized.")
        print(f"Memory module initialized, will save to '{self.memory_module.filepath}'.")
        if self.environment:
            print(f"Agent connected to environment: {type(self.environment).__name__}.")
        else:
            print("Agent running without external environment (using mock execution).")


    # Execute a generated Python policy script.
    # Returns stdout, stderr and a numeric FLE score if available.
    def execute_policy_script(self, python_script: str) -> tuple:
        # Execute the script either via the environment's ``eval`` method or a
        # local sandbox if no environment is attached.
        print(f"AlphaGeniusAgent: execute_policy_script() called with Python script:\n---\n{python_script}\n---")
        stdout = ""; stderr = ""; fle_score = 0.0
        if self.environment and hasattr(self.environment, 'eval') and callable(self.environment.eval):
            print("Using environment.eval() for script execution.")
            try:
                agent_idx_to_use = getattr(self, 'agent_idx', 0)
                score, goal_str, result_str = self.environment.eval(python_script, agent_idx=agent_idx_to_use)
                print(f"environment.eval result: score={score}, goal='{goal_str}', result_str_length={len(result_str)}")
                fle_score = score
                if score == -1 or "Error:" in result_str or "Traceback (most recent call last):" in result_str:
                    stderr = result_str
                else:
                    stdout = result_str
                if not stdout and not stderr: stdout = "Script executed via environment.eval() with no output."
            except Exception as e:
                stderr = f"Python exception calling environment.eval(): {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        else:  # Mock execution
            print("No environment 'eval' method found. Using local mock exec() for script.")
            sandbox_globals = {"print": print}; import builtins 
            def mock_tool_factory(tool_name_display):
                def mock_tool_impl(*args, **kwargs):
                    print_func = sandbox_globals.get('print', builtins.print)
                    print_func(
                        f"MockTool '{tool_name_display}' called with args: {args}, kwargs: {kwargs}"
                    )
                    return f"Mock result from {tool_name_display}"
                return mock_tool_impl
            essential_tools = ['tool_a', 'tool_b', 'tool_c']
            for tool_name in essential_tools:
                sandbox_globals[tool_name] = mock_tool_factory(tool_name)
            stdout_capture = io.StringIO(); stderr_capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                    exec(python_script, sandbox_globals)
                stdout = stdout_capture.getvalue(); stderr = stderr_capture.getvalue()
                if not stderr and stdout.strip() == "": stdout = "Python script (mock exec) executed successfully with no output.\n"
            except Exception as e:
                stdout = stdout_capture.getvalue(); stderr = stderr_capture.getvalue()
                stderr += f"\nPython exception during mock script execution: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return stdout, stderr, fle_score


    # Generate a Python script for the given sub-task, execute it and
    # compute intrinsic and extrinsic rewards.
    async def policy_step(self, last_stdout: str, last_stderr: str, current_task_prompt: str) -> tuple[str, str, str, float, float]:
        print(f"\nAlphaGeniusAgent: policy_step() processing task: '{current_task_prompt}'")
        if last_stdout: print(f"Observation (last_stdout):\n{last_stdout}")
        if last_stderr: print(f"Observation (last_stderr):\n{last_stderr}")
        if not hasattr(self, 'system_prompt') or not self.system_prompt: self.system_prompt = "Default AlphaGenius System Prompt (from policy_step)"
        user_content_for_llm = (f"Previous action's output (Observation):\nSTDOUT:\n{last_stdout if last_stdout else 'None'}\nSTDERR:\n{last_stderr if last_stderr else 'None'}\n\nYour current specific sub-task to accomplish: {current_task_prompt}")
        print(f"Sending to LLM client for code generation:")
        print(f"  System Prompt Override (first 100 chars): '{self.system_prompt[:100]}...'")
        print(f"  User Content (first 200 chars): '{user_content_for_llm[:200]}...'")
        python_script_to_execute = await self.llm_client.generate_script_async(
            user_prompt_content=user_content_for_llm,
            system_prompt_override=self.system_prompt,
        )

        stdout, stderr, fle_score = self.execute_policy_script(python_script_to_execute)
        print(f"execute_policy_script stdout (new output):\n{stdout}")
        novelty_score = self._compute_novelty_score(python_script_to_execute, current_task_prompt, stdout, stderr)
        if stderr:
            print(f"MetaCognition Hint: Error or stderr output during Python script execution!\nexecute_policy_script stderr (new output):\n{stderr}")
        else:
            print("Python script execution reported no errors in stderr.")

        return stdout, stderr, python_script_to_execute, fle_score, novelty_score

    def _compute_novelty_score(self, script: str, task_prompt: str, stdout: str, stderr: str) -> float:
        """Return 1.0 if the script contains previously unseen tokens."""
        if stderr:
            return 0.0

        tokens = set(
            token.strip()
            for token in script.replace("(", " ").replace(")", " ").replace(",", " ").split()
            if token.strip()
        )
        is_new = False
        for t in tokens:
            if self.discovery_tracker.add_event(t):
                is_new = True
        return 1.0 if is_new else 0.0


    async def run(self):
        print("\nAlphaGeniusAgent async run method started (MemoryModule integration).")
        current_stdout = "Agent initialized."
        current_stderr = ""

        high_level_task = "Print a friendly greeting message."
        print(f"High-level task for AlphaGenius: {high_level_task}")

        observation_for_planner = f"STDOUT:\n{current_stdout}\nSTDERR:\n{current_stderr}"
        sub_goals = await self.planner.decompose_task(high_level_task, observation_for_planner)

        if not sub_goals or (len(sub_goals) == 1 and ("Error in planning" in sub_goals[0] or "Exception during planning" in sub_goals[0])):
            print(f"Planner failed or returned no valid sub-goals: {sub_goals[0] if sub_goals else 'No sub-goals'}. Stopping run.")
            # Log planning failure to memory
            self.memory_module.add_experience(
                sub_goal=high_level_task, script="N/A (Planning Failed)", 
                stdout=observation_for_planner, stderr=sub_goals[0] if sub_goals else "Planner returned empty list",
                fle_score=None, novelty_score=None, success=False
            )
            return

        print(f"\nPlanner decomposed '{high_level_task}' into {len(sub_goals)} sub-goals: {sub_goals}")
        print("\nExecuting sub-goals...")

        for i, sub_goal in enumerate(sub_goals):
            print(f"\n--- Executing Sub-goal {i+1}/{len(sub_goals)}: {sub_goal} ---")
            
            # These are observations *before* this sub_goal attempt
            observation_stdout_for_this_attempt = current_stdout 
            observation_stderr_for_this_attempt = current_stderr

            # Initial attempt for the sub-goal
            
            stdout_attempt1, stderr_attempt1, script_attempted1, fle_score_attempt1, novelty_score_attempt1 = await self.policy_step(
                observation_stdout_for_this_attempt,
                observation_stderr_for_this_attempt,
                sub_goal,
            )

            success_attempt1 = not (stderr_attempt1 and ("error" in stderr_attempt1.lower() or "exception" in stderr_attempt1.lower() or "traceback" in stderr_attempt1.lower()))

            self.memory_module.add_experience(
                sub_goal=sub_goal,
                script=script_attempted1,
                stdout=stdout_attempt1,
                stderr=stderr_attempt1,
                fle_score=fle_score_attempt1,
                novelty_score=novelty_score_attempt1,
                success=success_attempt1,
            )

            current_stdout, current_stderr = stdout_attempt1, stderr_attempt1

            if not success_attempt1:
                print(f"Error detected for sub-goal '{sub_goal}'. Attempting reflection...")
                
                llm_suggestion = await self.metacognition.reflect_on_error(
                    original_sub_goal=sub_goal,
                    faulty_python_script=script_attempted1,
                    error_message=stderr_attempt1,
                    observation_summary=stdout_attempt1, # Output from the failed script
                    system_prompt_for_reflection=REFLECTION_SYSTEM_PROMPT
                )
                
                # Attempt to extract a corrected script from the suggestion
                corrected_script = None
                if "```python" in llm_suggestion:
                    try:
                        code_block = llm_suggestion.split("```python")[1].split("```")[0].strip()
                        if code_block: corrected_script = code_block
                    except IndexError: print("MetaCognition: Could not parse corrected script (```python block incomplete).")
                
                if corrected_script:
                    print(f"MetaCognition proposed a corrected script:\n{corrected_script}")
                    print("Attempting corrected script...")
                    
                    # Execute the corrected script
                    stdout_corrected, stderr_corrected, fle_score_corrected = self.execute_policy_script(corrected_script)
                    novelty_score_corrected = self._compute_novelty_score(corrected_script, sub_goal, stdout_corrected, stderr_corrected)

                    success_corrected = not (stderr_corrected and ("error" in stderr_corrected.lower() or "exception" in stderr_corrected.lower() or "traceback" in stderr_corrected.lower()))

                    self.memory_module.add_experience(
                        sub_goal=f"{sub_goal} (Corrected Attempt)",
                        script=corrected_script,
                        stdout=stdout_corrected,
                        stderr=stderr_corrected,
                        fle_score=fle_score_corrected,
                        novelty_score=novelty_score_corrected,
                        success=success_corrected,
                    )
                    current_stdout, current_stderr = stdout_corrected, stderr_corrected
                else:
                    print(f"MetaCognition did not provide a script, suggestion was: {llm_suggestion}")
            
        print("\nAlphaGeniusAgent async run method (with MemoryModule) finished.")
        if self.memory_module.filepath:
            self.memory_module.save_experiences()

# ... (__main__ block as before)
if __name__ == '__main__':
    agent = AlphaGeniusAgent(model_name="gpt-4o-mini")
    try:
        asyncio.run(agent.run())
    except Exception as e:
        print(f"Error running AlphaGeniusAgent: {e}")
        traceback.print_exc()
