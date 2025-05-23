import re # For parsing sub-goals later

import traceback # Added for more detailed error logging

# Attempt to import the LLM client base class for type hinting
# This assumes llm_client passed will be compatible with BaseLLMClient's interface (generate_script_async)
try:
    from ..llm_interface.llm_client import BaseLLMClient 
except ImportError:
    # Define a placeholder if the import fails (e.g. if file is run standalone or structure changes)
    class BaseLLMClient:
        async def generate_script_async(self, user_prompt_content: str, system_prompt_override: Optional[str] = None) -> str: # Adjusted placeholder signature
            raise NotImplementedError("This is a placeholder LLMClient.")

class Planner:
    """
    Handles decomposing high-level goals into a sequence of smaller, actionable sub-goals
    by prompting an LLM.
    """
    def __init__(self, llm_client: BaseLLMClient, system_prompt_for_planning: str):
        """
        Initializes the Planner.
        Args:
            llm_client: An instance of an LLM client (e.g., ConcreteLLMClient)
                         that has a generate_script_async(prompt) method.
            system_prompt_for_planning: A system prompt string that guides the LLM
                                        on how to decompose tasks.
        """
        self.llm_client = llm_client
        self.system_prompt_for_planning = system_prompt_for_planning
        print("Planner initialized.")

    def _parse_sub_goals(self, llm_response_text: str) -> list[str]:
        """
        Parses the LLM's text response to extract a list of sub-goals.
        This will be more robustly implemented in a later step.
        For now, assumes sub-goals are numbered lines.
        """
        sub_goals = []
        # Simple parsing for numbered lists (e.g., "1. Do X", "2. Do Y")
        # or lines starting with "- " or "* "
        lines = llm_response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Try to strip common list markers like "1. ", "- ", "* "
            match = re.match(r"^\s*(?:\d+\.\s*|-\s*|\*\s*)?(.*)", line)
            if match:
                goal_text = match.group(1).strip()
                if goal_text: # Avoid empty strings
                    sub_goals.append(goal_text)
        
        if not sub_goals and llm_response_text: # Fallback if no list markers found, take non-empty lines
            print("Warning: Sub-goal parsing found no list markers, using all non-empty lines as sub-goals.")
            for line in lines:
                if line: sub_goals.append(line)

        if not sub_goals:
             print(f"Warning: Could not parse any sub-goals from LLM response: '{llm_response_text}'")
             # Return a single fallback sub-goal to attempt the original goal directly
             return ["Attempt the original goal directly as a single step."]
        
        print(f"Parsed sub-goals: {sub_goals}")
        return sub_goals

    async def decompose_task(self, high_level_goal: str, observation_summary: str) -> list[str]:
        """
        Decomposes a high-level goal into a list of sub-goal strings using the LLM.
        Args:
            high_level_goal: The overall task to decompose.
            observation_summary: A summary of the current game state/observation.
        Returns:
            A list of strings, where each string is a sub-goal.
        """
        print(f"Planner: Decomposing task: '{high_level_goal}'")
        print(f"Based on observation summary: '{observation_summary[:200]}...'")

        # This is the content that forms the "user" part of the prompt to the LLM
        user_content_for_planner_llm = (
            f"High-Level Goal: {high_level_goal}\n\n"
            f"Current Observation/State Summary:\n{observation_summary}\n\n"
            f"Please decompose the high-level goal into a sequence of smaller, actionable sub-goals. "
            f"Return these as a list (e.g., numbered lines or bullet points)."
        )
        
        # The system_prompt_for_planning (e.g., "You are an expert planner...") is passed separately.
        # The llm_client (ConcreteLLMClient) will use this as the system message for the API call.
        
        print(f"Planner: Sending to LLM for decomposition. User content (first 200 chars):\n{user_content_for_planner_llm[:200]}...")
        print(f"Planner: Using system prompt override (first 100 chars):\n{self.system_prompt_for_planning[:100]}...")

        try:
            # Call ConcreteLLMClient's generate_script_async with separate user content and system prompt
            llm_response_text = await self.llm_client.generate_script_async(
                user_prompt_content=user_content_for_planner_llm,
                system_prompt_override=self.system_prompt_for_planning 
            )
            
            if llm_response_text.startswith("print('Error:"): 
                print(f"Planner: Error from LLM client: {llm_response_text}")
                return [f"Error in planning: Could not get sub-goals from LLM. Detail: {llm_response_text}"]

        except Exception as e:
            print(f"Planner: Exception during LLM call for task decomposition: {e}")
            # import traceback # For printing stack trace # Moved to top of file
            traceback.print_exc()
            return [f"Exception during planning: {e}"]

        sub_goals = self._parse_sub_goals(llm_response_text)
        return sub_goals

if __name__ == '__main__':
    # Example Usage (conceptual, requires a mock LLM client and asyncio setup)
    class MockLLMForPlanner(BaseLLMClient): # Mock to test Planner
        async def generate_script_async(self, user_prompt_content: str, system_prompt_override: Optional[str] = None) -> str:
            print(f"MockLLMForPlanner received user_prompt_content: {user_prompt_content[:100]}...")
            print(f"MockLLMForPlanner received system_prompt_override: {system_prompt_override[:100] if system_prompt_override else 'None'}...")
            # Simulate LLM decomposing a task based on user_prompt_content
            if "automate iron plates" in prompt.lower():
                return """
                1. Ensure enough iron ore is being mined.
                2. Set up a smelting column for iron ore.
                3. Route iron plates to a storage area.
                """
            else:
                return "1. Perform a generic first step for the goal."

    async def test_planner():
        print("Testing Planner...")
        mock_llm = MockLLMForPlanner()
        planning_prompt = "You are a planning assistant for a Factorio AI agent. Your task is to break down high-level goals into smaller, ordered steps. Focus on creating a logical sequence of sub-goals that can be translated into agent actions. Output a numbered list of these sub-goals."
        
        planner = Planner(llm_client=mock_llm, system_prompt_for_planning=planning_prompt)
        
        high_goal = "Automate iron plate production."
        observation = "Game just started. Inventory is empty. No structures placed."
        
        sub_goals_list = await planner.decompose_task(high_goal, observation)
        
        print(f"\nHigh-level goal: {high_goal}")
        print("Decomposed sub-goals:")
        if sub_goals_list:
            for i, sg in enumerate(sub_goals_list):
                print(f"{i+1}. {sg}")
        else:
            print("No sub-goals were generated.")

    import asyncio
    # To run the test:
    # asyncio.run(test_planner()) 
    # For now, just printing that it's testable:
    print("Planner class defined. To test, uncomment and run `asyncio.run(test_planner())` in the main block.")
