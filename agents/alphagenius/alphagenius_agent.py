# In agents/alphagenius/alphagenius_agent.py

import io
import contextlib
import traceback
import asyncio
import builtins # For mock execution fallback

try:
    from env.src.instance import FactorioInstance
    from env.src.gym.factorio_environment import FactorioEnv
    FactorioEnvironmentType = (FactorioInstance, FactorioEnv)
except ImportError:
    FactorioInstance = None
    FactorioEnv = None
    FactorioEnvironmentType = (type(None),)

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
            is_planning_call = "expert planning assistant" in system_prompt_content.lower() 
            is_reflection_call = "expert debugging and ai programming assistant" in system_prompt_content.lower()

            if is_planning_call:
                print("Placeholder LLMFactory: Detected call for PLANNING.")
                if "automate the production of iron plates" in user_prompt_content.lower():
                    response_content_str = (
                        "1. Mine 15 iron ore.\n"
                        "2. Cause an intentional error for testing reflection.\n" 
                        "3. Build a stone furnace after reflection."
                    )
                else: 
                    response_content_str = ("1. Survey area.\n2. Gather initial resources.")
            elif is_reflection_call:
                print("Placeholder LLMFactory: Detected call for REFLECTION.")
                if "faulty_python_script" in user_prompt_content and "undefined_variable_to_cause_error" in user_prompt_content:
                    response_content_str = (
                        "Explanation: The script tried to use 'undefined_variable_to_cause_error'.\n"
                        "```python\n"
                        "print('This is the CORRECTED mock script after reflection.')\n"
                        "corrected_result = 'reflection_success'\n"
                        "print(f'Corrected script result: {{corrected_result}}')\n"
                        "```"
                    )
                else:
                    response_content_str = "Suggestion: Unable to determine correction from this error context."
            else: # Call is for CODE GENERATION
                print("Placeholder LLMFactory: Detected call for CODE GENERATION.")
                if "mine 15 iron ore" in user_prompt_content.lower():
                    response_content_str = "print('Mock script: Mining 15 iron ore...')\nresult = mine_resource('iron-ore', 15)\nprint(f'Mining result: {{result}}')"
                elif "cause an intentional error" in user_prompt_content.lower():
                    response_content_str = "print('Executing script designed to fail...')\nundefined_variable_to_cause_error = 1 / 0" 
                elif "build a stone furnace after reflection" in user_prompt_content.lower():
                    response_content_str = "print('Mock script: Building a stone furnace post-reflection...')\nresult = place_entity(entity_name='stone-furnace', position={'x':5, 'y':5}, direction=Direction.NORTH)\nprint(f'Placement result: {{result}}')"
                else: 
                    response_content_str = "print(f'Mock Python script executed for sub-goal: {user_prompt_content[:50]}...')"
            
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
from .memory.memory_module import MemoryModule # <--- NEW IMPORT

PLANNING_SYSTEM_PROMPT = (
    "You are an expert planning assistant for an AI agent playing the game Factorio. "
    # ... (rest of prompt as before, shortened for brevity in this thought block)
    "1. Setup iron gear wheel production. "
    "2. Setup copper plate production. "
    "3. Setup assembly lines for red science packs using iron gear wheels and copper plates."
)

REFLECTION_SYSTEM_PROMPT = (
    "You are an expert debugging and AI programming assistant for an agent playing Factorio. "
    # ... (rest of prompt as before, shortened for brevity in this thought block)
    "Structure your response clearly, with the explanation first, then the corrected script (if any) or the suggestion."
)

class AlphaGeniusAgent:
    def __init__(self, config=None, environment=None, model_name: str = "gpt-4o-mini", agent_idx: int = 0):
        # ... (self.config, self.environment, self.agent_idx setup)
        self.config = config; self.environment = environment; self.agent_idx = agent_idx
        
        self.discovery_tracker = DiscoveryTracker()
        self.memory_module = MemoryModule(filepath="alphagenius_memory.jsonl") # <--- NEW
        self.world_model = WorldModel()  # Simple key-value world model
        self.world_model.set_fact("initialized", True)

        # ... (system_prompt setup)
        alpha_genius_meta_prompt = ("You are AlphaGenius, an advanced AI agent playing Factorio. " # Shortened for brevity
            "You have been given a specific sub-task by your planning module. " 
            "Your current objective is to generate a concise Python script using the available Factorio Learning Environment (FLE) tools to achieve this sub-task. " 
            "Carefully consider the output from the previous action (observation: stdout/stderr) to inform your script. "
            "If the previous step resulted in an error, try to understand it and adapt your current script to overcome the issue or achieve the sub-task in a different way. "
            "Strive for a script that is efficient, correct, and directly contributes to the sub-task. "
            "Ensure your script is directly executable and uses the FLE tool API as documented.\n")
        if self.environment and hasattr(self.environment, 'get_system_prompt'):
            fle_system_prompt = self.environment.get_system_prompt(self.agent_idx)
            self.system_prompt = alpha_genius_meta_prompt + "\nFLE Tool Documentation..." + fle_system_prompt
        else:
            self.system_prompt = alpha_genius_meta_prompt + "\nAvailable tools (mock environment)..."
        
        print("AlphaGeniusAgent initialized.")
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
        print(f"WorldModel initialized with facts: {self.world_model.all_facts()}")
        if self.environment and FactorioInstance: print(f"Agent has access to Factorio Environment: {type(self.environment).__name__}.")
        elif not self.environment: print("Agent running without recognized FactorioEnvironment access (using mocks for execution).")


    # execute_policy_script method remains the same for now
    # It will be updated in Step 3 of this plan to return fle_score
    def execute_policy_script(self, python_script: str) -> tuple: # Current: (stdout, stderr)
        # ... (implementation as before) ...
        # For now, let's assume it returns a mock score as the third element if it's not doing so already
        # This will be properly done in Step 3. For this step, we'll mock it in run.
        # stdout_val, stderr_val = "mock_stdout_from_exec", "" # Placeholder
        # This is just to make the call from run work, will be replaced by actual return in step 3
        # In reality, the existing exec_policy_script is fine, run will just pass a mock score for now.
        # ---- COPIED FROM PREVIOUS WORKING VERSION ----
        print(f"AlphaGeniusAgent: execute_policy_script() called with Python script:\n---\n{python_script}\n---")
        stdout = ""; stderr = ""
        if self.environment and hasattr(self.environment, 'eval') and callable(self.environment.eval):
            print("Using self.environment.eval() for script execution.")
            try:
                agent_idx_to_use = getattr(self, 'agent_idx', 0)
                score, goal_str, result_str = self.environment.eval(python_script, agent_idx=agent_idx_to_use)
                print(f"environment.eval result: score={score}, goal='{goal_str}', result_str_length={len(result_str)}")
                if score == -1 or "Error:" in result_str or "Traceback (most recent call last):" in result_str: stderr = result_str
                else: stdout = result_str
                if not stdout and not stderr: stdout = "Script executed via environment.eval() with no output."
            except Exception as e:
                stderr = f"Python exception calling environment.eval(): {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        else: # Mock execution
            print("No Factorio environment with 'eval'. Using local mock exec() for script.")
            sandbox_globals = {"print": print}; import builtins 
            def mock_tool_factory(tool_name_display):
                def mock_tool_impl(*args, **kwargs):
                    print_func = sandbox_globals.get('print', builtins.print)
                    print_func(f"MockTool '{tool_name_display}' called with args: {args}, kwargs: {kwargs}")
                    if tool_name_display == "get_player_inventory": return {"mock_item": 5}
                    return f"Mock result from {tool_name_display}"
                return mock_tool_impl
            essential_tools = ['mine_resource', 'craft_item', 'set_research', 'place_entity', 'get_player_inventory']
            for tool_name in essential_tools: sandbox_globals[tool_name] = mock_tool_factory(tool_name)
            class MockDirection: NORTH="NORTH"; SOUTH="SOUTH"; EAST="EAST"; WEST="WEST"
            class MockPrototype: IronOre="iron-ore"; StoneFurnace="stone-furnace"; MiningDrill="burner-mining-drill"
            class MockEntityStatus: WORKING="WORKING"; NO_INGREDIENTS="NO_INGREDIENTS"
            essential_enums = {'Direction': MockDirection, 'Prototype': MockPrototype, 'EntityStatus': MockEntityStatus, 
                               'Item': type('MockItem', (), {}), 'Recipe': type('MockRecipe', (), {}), 
                               'Technology': type('MockTechnology', (), {}), 'Resource': type('MockResource', (), {})}
            for name, mock_class in essential_enums.items(): sandbox_globals[name] = mock_class
            class MockDefines: pass; MockDefines.direction = MockDirection; class MockGame: defines = MockDefines(); sandbox_globals['game'] = MockGame(); sandbox_globals['defines'] = MockDefines
            stdout_capture = io.StringIO(); stderr_capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                    exec(python_script, sandbox_globals)
                stdout = stdout_capture.getvalue(); stderr = stderr_capture.getvalue()
                if not stderr and stdout.strip() == "": stdout = "Python script (mock exec) executed successfully with no output.\n"
            except Exception as e:
                stdout = stdout_capture.getvalue(); stderr = stderr_capture.getvalue()
                stderr += f"\nPython exception during mock script execution: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return stdout, stderr
        # ---- END COPIED SECTION ----


    # policy_step will be updated in Step 2 of this plan to return novelty_score
    # and in Step 3 to also return fle_score. For now, it returns (stdout, stderr, script)
    async def policy_step(self, last_stdout: str, last_stderr: str, current_task_prompt: str) -> tuple[str, str, str]:
        # ... (implementation as before, returning stdout, stderr, python_script_to_execute) ...
        # ---- COPIED FROM PREVIOUS WORKING VERSION ----
        print(f"\nAlphaGeniusAgent: policy_step() processing task: '{current_task_prompt}'")
        if last_stdout: print(f"Observation (last_stdout):\n{last_stdout}")
        if last_stderr: print(f"Observation (last_stderr):\n{last_stderr}")
        if not hasattr(self, 'system_prompt') or not self.system_prompt: self.system_prompt = "Default AlphaGenius System Prompt (from policy_step)" 
        user_content_for_llm = (f"Previous action's output (Observation):\nSTDOUT:\n{last_stdout if last_stdout else 'None'}\nSTDERR:\n{last_stderr if last_stderr else 'None'}\n\nYour current specific sub-task to accomplish: {current_task_prompt}")
        print(f"Sending to LLM client for code generation:")
        print(f"  System Prompt Override (first 100 chars): '{self.system_prompt[:100]}...'")
        print(f"  User Content (first 200 chars): '{user_content_for_llm[:200]}...'")
        python_script_to_execute = await self.llm_client.generate_script_async(user_prompt_content=user_content_for_llm, system_prompt_override=self.system_prompt)
        stdout, stderr = self.execute_policy_script(python_script_to_execute) # Original return
        print(f"execute_policy_script stdout (new output):\n{stdout}")
        if stderr: print(f"MetaCognition Hint: Error or stderr output during Python script execution!\nexecute_policy_script stderr (new output):\n{stderr}")
        else:
            print("Python script execution reported no errors in stderr.")
            # Intrinsic motivation / discovery tracker calls will be updated in Step 2 of this plan
            if "craft" in current_task_prompt.lower() or ("craft_item" in python_script_to_execute):
                mock_item_name = "unknown_item_crafted"
                if "furnace" in current_task_prompt.lower() or "stone-furnace" in python_script_to_execute: mock_item_name = "stone-furnace"
                elif "iron plate" in current_task_prompt.lower() or "iron-plate" in python_script_to_execute: mock_item_name = "iron-plate"
                # novelty_score = self.discovery_tracker.add_item_crafted(mock_item_name) # Will be done in step 2
                self.discovery_tracker.add_item_crafted(mock_item_name) # Call as before for now
            if "research" in current_task_prompt.lower() and ("automation" in current_task_prompt.lower() or "set_research('automation')" in python_script_to_execute):
                 self.discovery_tracker.add_technology_researched("automation")
        return stdout, stderr, python_script_to_execute 
        # ---- END COPIED SECTION ----


    async def run(self):
        print("\nAlphaGeniusAgent async run method started (MemoryModule integration).") # MODIFIED PRINT
        current_stdout = "Game started. No previous actions." 
        current_stderr = ""
        
        high_level_task = "Automate the production of iron plates starting from scratch."
        print(f"High-level task for AlphaGenius: {high_level_task}")

        observation_for_planner = f"STDOUT:\n{current_stdout}\nSTDERR:\n{current_stderr}"
        sub_goals = await self.planner.decompose_task(high_level_task, observation_for_planner)

        if not sub_goals or (len(sub_goals) == 1 and ("Error in planning" in sub_goals[0] or "Exception during planning" in sub_goals[0])):
            print(f"Planner failed or returned no valid sub-goals: {sub_goals[0] if sub_goals else 'No sub-goals'}. Stopping run.")
            # Log planning failure to memory
            self.memory_module.add_experience( # <--- NEW
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
            # policy_step will return (stdout, stderr, script_executed, fle_score, novelty_score) after later steps.
            # For now, it returns (stdout, stderr, script_executed).
            # We'll pass placeholder fle_score (0.0) and novelty_score (0.0) to memory.
            
            # Step 3 will make execute_policy_script return score.
            # Step 2 will make policy_step return novelty_score.
            # For THIS step (Step 1), we just pass placeholders to add_experience.
            
            stdout_attempt1, stderr_attempt1, script_attempted1 = await self.policy_step(
                observation_stdout_for_this_attempt, observation_stderr_for_this_attempt, sub_goal
            )
            
            # MOCK SCORES FOR NOW - these will be properly plumbed in steps 2 & 3
            mock_fle_score_attempt1 = 0.0 if not stderr_attempt1 else -1.0
            mock_novelty_score_attempt1 = 0.0 # Will get from discovery_tracker in step 2

            success_attempt1 = not (stderr_attempt1 and ("Error" in stderr_attempt1 or "Exception" in stderr_attempt1 or "Traceback" in stderr_attempt1))
            
            self.memory_module.add_experience( # <--- NEW
                sub_goal=sub_goal, script=script_attempted1, stdout=stdout_attempt1, stderr=stderr_attempt1,
                fle_score=mock_fle_score_attempt1, novelty_score=mock_novelty_score_attempt1, success=success_attempt1
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
                
                # Corrected script parsing (will be refined in Step 4 of this plan)
                corrected_script = None
                if "```python" in llm_suggestion:
                    try:
                        code_block = llm_suggestion.split("```python")[1].split("```")[0].strip()
                        if code_block: corrected_script = code_block
                    except IndexError: print("MetaCognition: Could not parse corrected script (```python block incomplete).")
                
                if corrected_script:
                    print(f"MetaCognition proposed a corrected script:\n{corrected_script}")
                    print("Attempting corrected script...")
                    
                    # stdout_corrected, stderr_corrected = self.execute_policy_script(corrected_script)
                    # This also needs to be updated to get fle_score in Step 3
                    # For now, using placeholders for score for memory logging.
                    stdout_corrected, stderr_corrected = self.execute_policy_script(corrected_script)
                    mock_fle_score_corrected = 0.0 if not stderr_corrected else -1.0
                    mock_novelty_score_corrected = 0.0 # Correction attempts might not generate novelty

                    success_corrected = not (stderr_corrected and ("Error" in stderr_corrected or "Exception" in stderr_corrected or "Traceback" in stderr_corrected))
                    
                    self.memory_module.add_experience( # <--- NEW
                        sub_goal=f"{sub_goal} (Corrected Attempt)", script=corrected_script, 
                        stdout=stdout_corrected, stderr=stderr_corrected,
                        fle_score=mock_fle_score_corrected, novelty_score=mock_novelty_score_corrected, success=success_corrected
                    )
                    current_stdout, current_stderr = stdout_corrected, stderr_corrected # Update observation
                else:
                    print(f"MetaCognition did not provide a script, suggestion was: {llm_suggestion}")
            
        print("\nAlphaGeniusAgent async run method (with MemoryModule) finished.")

# ... (__main__ block as before)
if __name__ == '__main__':
    greet()  # Demonstration of a simple skill
    agent = AlphaGeniusAgent(model_name="gpt-4o-mini")
    # Example: store and retrieve a fact from the world model
    agent.world_model.set_fact("example", 42)
    print("Example fact from world model:", agent.world_model.get_fact("example"))
    try:
        asyncio.run(agent.run())
    except Exception as e:
        print(f"Error running AlphaGeniusAgent: {e}")
        traceback.print_exc()
```
