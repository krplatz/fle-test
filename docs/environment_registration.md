# Registering a Custom Environment for AlphaGenius

`AlphaGeniusAgent` interacts with its world though objects that implement the
`EnvironmentBase` interface. This allows the agent to be reused in different
games or simulations.

1. **Implement the interface**

   Create a subclass of `EnvironmentBase` and provide implementations for
   `eval()` and `get_system_prompt()`. Optionally override `reset()`.

   ```python
   from agents.alphagenius.environment.base import EnvironmentBase

   class ChessEnv(EnvironmentBase):
       def eval(self, code: str, agent_idx: int = 0):
           # Execute `code` inside the chess simulator
           ...
           return score, "move piece", output

       def get_system_prompt(self, agent_idx: int = 0) -> str:
           return "Documentation for the chess API"
   ```

2. **Create the agent with your environment**

   ```python
   from agents.alphagenius.alphagenius_agent import AlphaGeniusAgent
   env = ChessEnv()
   agent = AlphaGeniusAgent(environment=env)
   ```

The agent will automatically use the system prompt and `eval` method of your
environment when generating and executing code.
