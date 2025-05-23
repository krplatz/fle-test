# agents/alphagenius/metacognition/meta_cognition.py
from typing import Optional

# Attempt to import BaseLLMClient for type hinting
try:
    from ..llm_interface.llm_client import BaseLLMClient
except ImportError:
    # Define a placeholder if the import fails
    class BaseLLMClient:
        async def generate_script_async(self, user_prompt_content: str, system_prompt_override: Optional[str] = None) -> str:
            raise NotImplementedError("This is a placeholder LLMClient for MetaCognition.")

class MetaCognition:
    """
    Handles meta-cognitive processes for AlphaGenius, starting with error reflection.
    """
    def __init__(self, llm_client: BaseLLMClient):
        """
        Initializes the MetaCognition module.
        Args:
            llm_client: An LLM client instance to be used for reflective LLM calls.
        """
        self.llm_client = llm_client
        print("MetaCognition module initialized.")

    async def reflect_on_error(self, 
                               original_sub_goal: str, 
                               faulty_python_script: str, 
                               error_message: str, # This is likely stderr from script execution
                               observation_summary: str, # This is stdout from script execution (or prior observation)
                               system_prompt_for_reflection: str) -> str:
        """
        Prompts the LLM to reflect on an error and suggest a correction or analysis.

        Args:
            original_sub_goal: The sub-goal the agent was trying to achieve.
            faulty_python_script: The Python script that produced the error.
            error_message: The stderr output or exception message.
            observation_summary: The stdout output from the failed script, or general observation.
            system_prompt_for_reflection: The system prompt to guide the LLM's reflection.

        Returns:
            A string from the LLM, hopefully a corrected Python script or an analysis.
        """
        print("\n--- MetaCognition: Reflecting on Error ---")
        print(f"Original Sub-goal: {original_sub_goal}")
        print(f"Faulty Script:\n{faulty_python_script}")
        print(f"Error Message (stderr):\n{error_message}")
        print(f"Observation (stdout/context):\n{observation_summary}")

        user_reflection_prompt = (
            f"The following sub-goal was being attempted: '{original_sub_goal}'\n\n"
            f"The Python script executed was:\n```python\n{faulty_python_script}\n```\n\n"
            f"Execution resulted in the following error (stderr):\n{error_message}\n\n"
            f"The output from the script before the error, or general observation context (stdout), was:\n{observation_summary}\n\n"
            f"Please analyze this failure. Based on your analysis, provide:"
            f"1. A brief explanation of the likely cause of the error. (max 2 sentences)"
            f"2. A corrected Python script to achieve the original sub-goal: '{original_sub_goal}'. "
            f"   Enclose the corrected script in ```python ... ``` tags. "
            f"   If you believe the original sub-goal is unachievable with a simple correction or requires a different approach, "
            f"   instead of a script, provide a brief suggestion starting with 'Suggestion: ' (e.g., 'Suggestion: Re-evaluate resource availability before attempting this again.')."
        )
        
        print(f"MetaCognition: Sending to LLM for error reflection. System Prompt (first 100): '{system_prompt_for_reflection[:100]}...' User Prompt (first 100): '{user_reflection_prompt[:100]}...'")

        try:
            llm_suggestion_or_corrected_script = await self.llm_client.generate_script_async(
                user_prompt_content=user_reflection_prompt,
                system_prompt_override=system_prompt_for_reflection
            )
            print(f"MetaCognition: LLM reflection response:\n{llm_suggestion_or_corrected_script}")
            return llm_suggestion_or_corrected_script
        except Exception as e:
            print(f"MetaCognition: Exception during LLM call for error reflection: {e}")
            return f"print('Error: Exception during meta-cognitive reflection LLM call: {e!r}')"

if __name__ == '__main__':
    # Conceptual test for MetaCognition (requires mock LLM and asyncio)
    class MockLLMForReflection(BaseLLMClient):
        async def generate_script_async(self, user_prompt_content: str, system_prompt_override: Optional[str] = None) -> str:
            print(f"MockLLMForReflection received user prompt (first 100): {user_prompt_content[:100]}...")
            print(f"MockLLMForReflection received system override (first 100): {system_prompt_override[:100]}...")
            if "NameError" in user_prompt_content:
                return ("Explanation: The variable 'Directon' was misspelled. It should be 'Direction'.\n"
                        "```python\n"
                        "print('Corrected script: Using Direction.NORTH')\n"
                        "place_entity(entity_name='stone-furnace', position={'x':1, 'y':1}, direction=Direction.NORTH)\n"
                        "```")
            return "Suggestion: The error is complex. Try a simpler sub-goal first."

    async def test_reflection():
        mock_llm = MockLLMForReflection()
        # system_prompt_for_reflection would be defined in AlphaGeniusAgent or a prompts file
        test_reflection_prompt = "You are an expert debugger and Factorio AI assistant. Analyze the provided error and script, then offer a correction or suggestion."
        
        metacog = MetaCognition(llm_client=mock_llm)
        
        result = await metacog.reflect_on_error(
            original_sub_goal="Place a furnace facing North.",
            faulty_python_script="place_entity(entity_name='stone-furnace', position={'x':1, 'y':1}, direction=Directon.NORTH)", # Misspelled Direction
            error_message="NameError: name 'Directon' is not defined",
            observation_summary="Player inventory has stone-furnace: 1. No entities nearby.",
            system_prompt_for_reflection=test_reflection_prompt
        )
        print(f"\nReflection Test Result:\n{result}")

    import asyncio # Ensure asyncio is imported for the test
    # To run the test:
    # asyncio.run(test_reflection())
    print("MetaCognition class defined. To test, uncomment `asyncio.run(test_reflection())` in the main block.")

```
