import asyncio
import os

import anthropic
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, wait_exponential

from agents.utils.llm_utils import remove_whitespace_blocks, merge_contiguous_messages, format_messages_for_openai, \
    format_messages_for_anthropic, has_image_content


class LLMFactory:
    # Models that support image input
    MODELS_WITH_IMAGE_SUPPORT = [
        # Claude models with vision
        "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
        "claude-3-5-sonnet", "claude-3-7-sonnet", "claude-3.7-sonnet",
        # OpenAI models with vision
        "gpt-4-vision", "gpt-4-turbo", "gpt-4o", "gpt-4-1106-vision-preview"
    ]

    def __init__(self, model: str, beam: int = 1):
        self.model = model
        self.beam = beam

    def _is_model_image_compatible(self, model: str) -> bool:
        """
        Check if the model supports image inputs, accounting for model version suffixes.

        Examples:
            'claude-3.5-sonnet-20241022' -> matches 'claude-3.5-sonnet'
            'gpt-4o-2024-05-13' -> matches 'gpt-4o'
        """
        # Normalize the model name to lowercase
        model_lower = model.lower()

        # First check for exact matches
        if model_lower in self.MODELS_WITH_IMAGE_SUPPORT:
            return True

        # Check for models with version number suffixes
        for supported_model in self.MODELS_WITH_IMAGE_SUPPORT:
            if supported_model in model:
                return True

        # Special handling for custom adaptations
        if "vision" in model_lower and any(gpt in model_lower for gpt in ["gpt-4", "gpt4"]):
            return True

        return False

    @retry(wait=wait_exponential(multiplier=2, min=2, max=15))
    async def acall(self, *args, **kwargs):
        max_tokens = kwargs.get('max_tokens', 2000)
        model_to_use = kwargs.get('model', self.model)
        messages = kwargs.get('messages', [])

        # Check for image content
        has_images = has_image_content(messages)

        # Validate image capability if images are present
        if has_images and not self._is_model_image_compatible(model_to_use):
            raise ValueError(f"Model {model_to_use} does not support image inputs, but images were provided.")

        if 'open-router' in model_to_use:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv('OPEN_ROUTER_API_KEY'),
            )
            response = await client.chat.completions.create(
                model=model_to_use.replace('open-router', '').strip('-'),
                max_tokens=kwargs.get('max_tokens', 256),
                temperature=kwargs.get('temperature', 0.3),
                messages=kwargs.get('messages', None),
                logit_bias=kwargs.get('logit_bias', None),
                n=kwargs.get('n_samples', None),
                stop=kwargs.get('stop_sequences', None),
                stream=False,
                presence_penalty=kwargs.get('presence_penalty', None),
                frequency_penalty=kwargs.get('frequency_penalty', None),
            )
            return response

        if "claude" in model_to_use:
            # Set up and call the Anthropic API
            api_key = os.getenv('ANTHROPIC_API_KEY')

            # Process system message
            system_message = ""
            if messages and messages[0]['role'] == "system":
                system_message = messages[0]['content']
                if isinstance(system_message, list):
                    # Extract just the text parts for system message
                    system_text_parts = []
                    for part in system_message:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            system_text_parts.append(part.get('text', ''))
                        elif isinstance(part, str):
                            system_text_parts.append(part)
                    system_message = "\n".join(system_text_parts)
                system_message = system_message.strip()

            # If the most recent message is from the assistant and ends with whitespace, clean it
            if messages and messages[-1]['role'] == "assistant":
                if isinstance(messages[-1]['content'], str):
                    messages[-1]['content'] = messages[-1]['content'].strip()

            # If the most recent message is from the assistant, add a user message to prompt the assistant
            if messages and messages[-1]['role'] == "assistant":
                messages.append({
                    "role": "user",
                    "content": "Success."
                })

            if not has_images:
                # For text-only messages, use the standard processing
                messages = remove_whitespace_blocks(messages)
                messages = merge_contiguous_messages(messages)

                # Format for Claude API
                anthropic_messages = []
                for msg in messages:
                    if msg['role'] != 'system':  # System message handled separately
                        anthropic_messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
            else:
                # For messages with images, use the special formatter
                anthropic_messages = format_messages_for_anthropic(messages, system_message)

            if not system_message:
                raise RuntimeError("No system message!!")

            try:
                client = anthropic.Anthropic()
                # Use asyncio.to_thread for CPU-bound operations
                response = await asyncio.to_thread(
                    client.messages.create,
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=max_tokens,
                    model=model_to_use,
                    messages=anthropic_messages,
                    system=system_message,
                    stop_sequences=kwargs.get('stop_sequences', ["```END"]),
                )
            except Exception as e:
                print(e)
                raise

            return response

        elif "deepseek" in model_to_use:
            if has_images:
                raise ValueError(f"Deepseek models do not support image inputs, but images were provided.")

            client = AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
            try:
                response = await client.chat.completions.create(
                    model=model_to_use,
                    max_tokens=kwargs.get('max_tokens', 256),
                    temperature=kwargs.get('temperature', 0.3),
                    messages=kwargs.get('messages', None),
                    logit_bias=kwargs.get('logit_bias', None),
                    n=kwargs.get('n_samples', None),
                    stop=kwargs.get('stop_sequences', None),
                    stream=False,
                    presence_penalty=kwargs.get('presence_penalty', None),
                    frequency_penalty=kwargs.get('frequency_penalty', None),
                )
                return response
            except Exception as e:
                print(e)
                raise

        elif "gemini" in model_to_use:
            if has_images:
                raise ValueError(f"Gemini integration doesn't support image inputs through this interface.")

            client = AsyncOpenAI(api_key=os.getenv("GEMINI_API_KEY"),
                                 base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            response = await client.chat.completions.create(
                model=model_to_use,
                max_tokens=kwargs.get('max_tokens', 256),
                temperature=kwargs.get('temperature', 0.3),
                messages=kwargs.get('messages', None),
                # logit_bias=kwargs.get('logit_bias', None),
                n=kwargs.get('n_samples', None),
                # stop=kwargs.get('stop_sequences', None),
                stream=False
                # presence_penalty=kwargs.get('presence_penalty', None),
                # frequency_penalty=kwargs.get('frequency_penalty', None),
            )
            return response

        elif any(model in model_to_use for model in ["llama", "Qwen"]):
            if has_images:
                raise ValueError(f"Llama and Qwen models do not support image inputs through this interface.")

            client = AsyncOpenAI(api_key=os.getenv("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1")
            return await client.chat.completions.create(
                model=model_to_use,
                max_tokens=kwargs.get('max_tokens', 256),
                temperature=kwargs.get('temperature', 0.3),
                messages=kwargs.get('messages', None),
                logit_bias=kwargs.get('logit_bias', None),
                n=kwargs.get('n_samples', None),
                stop=kwargs.get('stop_sequences', None),
                stream=False
            )

        elif "o1-mini" in model_to_use or 'o3-mini' in model_to_use:
            if has_images:
                raise ValueError(f"Claude o1-mini and o3-mini models do not support image inputs.")

            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # replace `max_tokens` with `max_completion_tokens` for OpenAI API
            if "max_tokens" in kwargs:
                kwargs.pop("max_tokens")
            messages = kwargs.get('messages')
            messages[0]['role'] = 'developer'
            try:
                reasoning_length = "low"
                if "med" in model_to_use:
                    reasoning_length = "medium"
                elif "high" in model_to_use:
                    reasoning_length = "high"
                model = kwargs.get('model', 'o3-mini')
                if 'o3-mini' in model:
                    model = 'o3-mini'
                elif 'o1-mini' in model:
                    model = 'o1-mini'

                response = await client.chat.completions.create(
                    *args,
                    n=self.beam,
                    model=model,
                    messages=messages,
                    stream=False,
                    response_format={
                        "type": "text"
                    },
                    reasoning_effort=reasoning_length
                )
                return response
            except Exception as e:
                print(e)
        else: # This is the default OpenAI / OpenAI-compatible path
            try:
                custom_base_url = os.getenv("LMSTUDIO_OPENAI_BASE_URL")
                # Check for explicit custom API key, or an empty string for it.
                # os.getenv returns None if var is not found.
                # If LMSTUDIO_OPENAI_API_KEY is set to an empty string, custom_api_key will be "".
                custom_api_key = os.getenv("LMSTUDIO_OPENAI_API_KEY") 

                api_key_to_use = None 
                base_url_to_use = None

                if custom_base_url:
                    base_url_to_use = custom_base_url
                    if custom_api_key is not None: # Explicitly set, even if empty string
                        api_key_to_use = custom_api_key
                        print(f"LLMFactory: Using custom OpenAI-compatible endpoint. Base URL: {base_url_to_use}")
                        print(f"LLMFactory: Using API key from LMSTUDIO_OPENAI_API_KEY ('{api_key_to_use if api_key_to_use else 'empty string'}').")
                    else: # LMSTUDIO_OPENAI_API_KEY is not set at all
                        api_key_to_use = None # Pass None to AsyncOpenAI
                        print(f"LLMFactory: Using custom OpenAI-compatible endpoint. Base URL: {base_url_to_use}")
                        print("LLMFactory: LMSTUDIO_OPENAI_API_KEY not set, using api_key=None.")
                else:
                    # Default to OpenAI official
                    api_key_to_use = os.getenv("OPENAI_API_KEY")
                    print("LLMFactory: Using default OpenAI endpoint.")
                    if not api_key_to_use:
                        print("LLMFactory: Warning - OPENAI_API_KEY not set for default OpenAI endpoint.")


                client = AsyncOpenAI(api_key=api_key_to_use, base_url=base_url_to_use)
                
                assert "messages" in kwargs, "You must provide a list of messages to the model."

                # Message formatting logic (assuming has_images and format_messages_for_openai are defined elsewhere in the class or file)
                current_messages = kwargs.get('messages', [])
                # has_images is defined at the start of acall method.
                # model_to_use is also defined at the start of acall method.

                if has_images and not self._is_model_image_compatible(model_to_use): # Ensure self._is_model_image_compatible is available
                    raise ValueError(f"Model {model_to_use} does not support image inputs, but images were provided.")

                if has_images:
                    formatted_messages = format_messages_for_openai(current_messages) # Ensure this helper is available
                else:
                    formatted_messages = current_messages
                
                # Make the call
                response = await client.chat.completions.create(
                    model=model_to_use, # model_to_use is from kwargs.get('model', self.model)
                    max_tokens=kwargs.get('max_tokens', 256),
                    temperature=kwargs.get('temperature', 0.3),
                    messages=formatted_messages,
                    logit_bias=kwargs.get('logit_bias', None),
                    n=kwargs.get('n_samples', None),
                    stop=kwargs.get('stop_sequences', None),
                    stream=False,
                    presence_penalty=kwargs.get('presence_penalty', None),
                    frequency_penalty=kwargs.get('frequency_penalty', None),
                )
                return response
            except Exception as e:
                print(f"LLMFactory: Error in OpenAI compatible API call: {e}")
                # Fallback attempt with truncated messages (as was in original code)
                try:
                    print("LLMFactory: Retrying OpenAI call with truncated message history as fallback.")
                    # (Ensure client is initialized as above for this retry block as well)
                    # Re-apply custom URL/key logic for retry client
                    custom_base_url_retry = os.getenv("LMSTUDIO_OPENAI_BASE_URL")
                    custom_api_key_retry = os.getenv("LMSTUDIO_OPENAI_API_KEY")
                    api_key_retry = None
                    base_url_retry = None

                    if custom_base_url_retry:
                        base_url_retry = custom_base_url_retry
                        if custom_api_key_retry is not None:
                            api_key_retry = custom_api_key_retry
                        else: # Not set, pass None
                            api_key_retry = None 
                    else: # Default OpenAI for retry
                        api_key_retry = os.getenv("OPENAI_API_KEY")
                    
                    client_retry = AsyncOpenAI(api_key=api_key_retry, base_url=base_url_retry)

                    sys_msg = kwargs.get('messages', [])[0] # System message
                    # Truncate to system message + last 8 user/assistant messages
                    truncated_messages_raw = [sys_msg] + kwargs.get('messages', [])[-8:] 
                    
                    has_images_retry = has_image_content(truncated_messages_raw)
                    if has_images_retry and not self._is_model_image_compatible(model_to_use):
                         raise ValueError(f"Fallback: Model {model_to_use} does not support image inputs.")

                    if has_images_retry:
                        formatted_messages_retry = format_messages_for_openai(truncated_messages_raw)
                    else:
                        formatted_messages_retry = truncated_messages_raw

                    return await client_retry.chat.completions.create(
                        model=model_to_use,
                        max_tokens=kwargs.get('max_tokens', 256),
                        temperature=kwargs.get('temperature', 0.3),
                        messages=formatted_messages_retry, # Use formatted truncated messages
                        logit_bias=kwargs.get('logit_bias', None),
                        n=kwargs.get('n_samples', None),
                        stop=kwargs.get('stop_sequences', None),
                        stream=False,
                        presence_penalty=kwargs.get('presence_penalty', None),
                        frequency_penalty=kwargs.get('frequency_penalty', None),
                    )
                except Exception as final_e:
                    print(f"LLMFactory: Error in OpenAI fallback API call: {final_e}")
                    raise final_e # Re-raise the final exception

    def call(self, *args, **kwargs):
        # For the synchronous version, we should also implement image support,
        # but I'll leave this method unchanged as the focus is on the async version.
        # The same pattern would be applied here as in acall.
        max_tokens = kwargs.get('max_tokens', 1500)
        model_to_use = kwargs.get('model', self.model)

        messages = kwargs.get('messages', [])
        has_images = self._has_image_content(messages)

        # Validate image capability if images are present
        if has_images and not self._is_model_image_compatible(model_to_use):
            raise ValueError(f"Model {model_to_use} does not support image inputs, but images were provided.")

        if "claude" in model_to_use:
            # Set up and call the Anthropic API
            api_key = os.getenv('ANTHROPIC_API_KEY')

            # Process system message
            system_message = ""
            if messages and messages[0]['role'] == "system":
                system_message = messages[0]['content']
                if isinstance(system_message, list):
                    # Extract just the text parts for system message
                    system_text_parts = []
                    for part in system_message:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            system_text_parts.append(part.get('text', ''))
                        elif isinstance(part, str):
                            system_text_parts.append(part)
                    system_message = "\n".join(system_text_parts)
                system_message = system_message.strip()

            # Remove final assistant content that ends with trailing whitespace
            if messages[-1]['role'] == "assistant":
                if isinstance(messages[-1]['content'], str):
                    messages[-1]['content'] = messages[-1]['content'].strip()

            # If the most recent message is from the assistant, add a user message to prompt the assistant
            if messages[-1]['role'] == "assistant":
                messages.append({
                    "role": "user",
                    "content": "Success."
                })

            if not has_images:
                # Standard text processing
                messages = self.remove_whitespace_blocks(messages)
                messages = self.merge_contiguous_messages(messages)

                # Format for Claude API
                anthropic_messages = []
                for msg in messages:
                    if msg['role'] != 'system':  # System message handled separately
                        anthropic_messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
            else:
                # Format with image support
                anthropic_messages = self._format_messages_for_anthropic(messages, system_message)

            try:
                client = anthropic.Anthropic()
                response = client.messages.create(
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=max_tokens,
                    model=model_to_use,
                    messages=anthropic_messages,
                    system=system_message,
                    stop_sequences=kwargs.get('stop_sequences', None),
                )
            except Exception as e:
                print(e)
                raise

            return response

        elif "deepseek" in model_to_use:
            if has_images:
                raise ValueError(f"Deepseek models do not support image inputs, but images were provided.")

            client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
            response = client.chat.completions.create(*args,
                                                      **kwargs,
                                                      model=model_to_use,
                                                      presence_penalty=kwargs.get('presence_penalty', None),
                                                      frequency_penalty=kwargs.get('frequency_penalty', None),
                                                      logit_bias=kwargs.get('logit_bias', None),
                                                      n=kwargs.get('n_samples', None),
                                                      stop=kwargs.get('stop_sequences', None),
                                                      stream=False)
            return response

        elif "o1-mini" in model_to_use:
            if has_images:
                raise ValueError(f"Claude o1-mini model does not support image inputs.")

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # replace `max_tokens` with `max_completion_tokens` for OpenAI API
            if "max_tokens" in kwargs:
                kwargs.pop("max_tokens")

            return client.chat.completions.create(*args, n=self.beam,
                                                  **kwargs,
                                                  stream=False)
        else: # This is the default OpenAI / OpenAI-compatible path for the synchronous 'call'
            custom_base_url = os.getenv("LMSTUDIO_OPENAI_BASE_URL")
            custom_api_key = os.getenv("LMSTUDIO_OPENAI_API_KEY") # Will be None if not set

            api_key_to_use = None
            base_url_to_use = None 

            if custom_base_url:
                base_url_to_use = custom_base_url
                if custom_api_key is not None: # Explicitly set (could be empty string)
                    api_key_to_use = custom_api_key
                    print(f"LLMFactory (sync): Using custom OpenAI-compatible endpoint. Base URL: {base_url_to_use}")
                    print(f"LLMFactory (sync): Using API key from LMSTUDIO_OPENAI_API_KEY ('{api_key_to_use if api_key_to_use else 'empty string'}').")
                else: # LMSTUDIO_OPENAI_API_KEY is not set at all
                    api_key_to_use = None # Pass None to OpenAI client
                    print(f"LLMFactory (sync): Using custom OpenAI-compatible endpoint. Base URL: {base_url_to_use}")
                    print("LLMFactory (sync): LMSTUDIO_OPENAI_API_KEY not set, using api_key=None.")
            else:
                # Default to OpenAI official
                api_key_to_use = os.getenv("OPENAI_API_KEY")
                print("LLMFactory (sync): Using default OpenAI endpoint.")
                if not api_key_to_use:
                     print("LLMFactory (sync): Warning - OPENAI_API_KEY not set for default OpenAI endpoint.")
            
            client = OpenAI(api_key=api_key_to_use, base_url=base_url_to_use) 
            
            assert "messages" in kwargs, "You must provide a list of messages to the model."
            
            # Assuming formatted_messages are directly the messages from kwargs for sync path
            # (as per original structure of 'call' method which had less image handling than 'acall')
            # And as per the prompt's provided snippet for this change.
            formatted_messages = kwargs.get('messages', [])

            return client.chat.completions.create(
                model=kwargs.get('model', self.model), # model_to_use from kwargs
                max_tokens=kwargs.get('max_tokens', 256),
                temperature=kwargs.get('temperature', 0.3),
                messages=formatted_messages,
                logit_bias=kwargs.get('logit_bias', None),
                n=kwargs.get('n_samples', None),
                stop=kwargs.get('stop_sequences', None),
                stream=False
            )
