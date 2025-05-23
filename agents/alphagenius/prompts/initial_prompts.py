# initial_prompts.py

# This file contains initial prompts that can be used by the AlphaGenius agent,
# especially during early development and testing phases, as outlined in
# Phase 1 of the research proposal ("Initial Prompting").

# List of simple task prompts
SIMPLE_TASK_PROMPTS = [
    "Mine 10 iron ore.",
    "Craft a furnace.",
    "Place a furnace near your current location.",
    "Mine 5 coal.",
    "Smelt 5 iron plates using a furnace and coal for fuel."
]

# Prompts could also be structured, e.g., as a dictionary with more context
STRUCTURED_PROMPTS = {
    "task_001": {
        "goal": "Mine 10 iron ore.",
        "description": "Locate an iron ore patch and use the mining drill or basic mining tool to collect 10 units of iron ore. Ensure they are in the player's inventory.",
        "expected_outcome_keywords": ["iron_ore: 10", "inventory"]
    },
    "task_002": {
        "goal": "Craft a stone furnace.",
        "description": "Gather the necessary resources (e.g., 5 stone) and craft a stone furnace. Check your crafting menu for the recipe.",
        "expected_outcome_keywords": ["stone-furnace: 1", "crafted"]
    }
}

# You can add more prompts or more complex structures as needed.

if __name__ == '__main__':
    print("Available Simple Task Prompts:")
    for i, p in enumerate(SIMPLE_TASK_PROMPTS):
        print(f"{i+1}. {p}")

    print("\nAvailable Structured Prompts:")
    for key, val in STRUCTURED_PROMPTS.items():
        print(f"{key}: {val['goal']}")
