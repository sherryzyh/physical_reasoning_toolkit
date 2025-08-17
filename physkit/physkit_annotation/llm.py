import os
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
import logging

class LLMClient:
    def __init__(self, model: str, logger=None):
        load_dotenv()
        self.model = model
        self.client = None
        self.provider = None
        self.logger = logger
        
    def chat(self, messages):
        raise NotImplementedError("Subclasses must implement .chat()")

    def chat_structured(self, system_prompt, prompt, response_format=None):
        raise NotImplementedError("Subclasses must implement .chat_structured()")

    @staticmethod
    def from_model(model: str, logger=None):
        model_lower = model.lower()
        if "deepseek" in model_lower:
            return DeepseekModel(model, logger)
        elif model_lower.startswith("o"):
            return OAIReasonModel(model, logger)
        elif model_lower.startswith("gpt"):
            return OAIChatModel(model, logger)
        elif model_lower.startswith("gemini"):
            return GeminiModel(model, logger)
        else:
            raise ValueError(f"Unknown model: {model}")

class DeepseekModel(LLMClient):
    def __init__(self, model: str, logger = None):
        super().__init__(model, logger)
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.provider = "deepseek"

    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

class OAIReasonModel(LLMClient):
    def __init__(self, model: str, logger=None):
        super().__init__(model, logger)
        self.client = OpenAI()
        self.provider = "openai"

    def chat(self, messages):
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "medium"},
            input=messages
        )
        return response.output_text

    def chat_structured(self, system_prompt, prompt, response_format=None):
        response = self.client.responses.parse(
            model=self.model,
            input=prompt,
            text_format=response_format
        )
        return response.output_parsed

class OAIChatModel(LLMClient):
    def __init__(self, model: str, logger=None):
        super().__init__(model, logger)
        self.client = OpenAI()
        self.provider = "openai"

    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content 
    
    def chat_structured(self, system_prompt, prompt, response_format=None):
        if self.model != "gpt-4o":
            raise NotImplementedError("Structured Output is not supported for {self.model}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.responses.parse(
            model=self.model,
            input=messages,
            text_format=response_format
        )
        return response.output_parsed


class GeminiModel(LLMClient):
    def __init__(self, model: str, logger=None):
        super().__init__(model, logger)
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        self.provider = "google"
        # The client is just a reference to the `genai` module or can be kept as None
        # since we will create the model instance within the chat method.
        # This makes the class more flexible and stateless per chat call.
        self.client = None

    def _convert_openai_messages_to_gemini_contents(self, messages):
        """
        Converts OpenAI-style messages to Gemini-style contents and extracts
        the system instruction.
        Returns a tuple: (system_instruction, contents)
        """
        system_instruction = None
        contents = []

        if messages and messages[0]['role'] == 'system':
            system_instruction = messages[0]['content']
            conversation_messages = messages[1:]
        else:
            conversation_messages = messages
        
        for msg in conversation_messages:
            role = "user" if msg['role'] == 'user' else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg['content']}]
            })
            
        return system_instruction, contents

    def chat(self, messages, *args, **kwargs):
        """
        Sends a list of messages to the Gemini API and returns the response text.
        """
        system_instruction, contents = self._convert_openai_messages_to_gemini_contents(messages)
        
        # Instantiate the model with the system instruction for this specific chat call.
        # This is the crucial fix. The GenerativeModel instance is created for each call
        # if a system instruction is present.
        if system_instruction:
            model_instance = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_instruction
            )
        else:
            model_instance = genai.GenerativeModel(model_name=self.model)

        response = model_instance.generate_content(
            contents=contents,
            *args,
            **kwargs,
        )
        
        return response.text

    def chat_structured(self, messages, response_format=None):
        raise NotImplementedError(
            f"Structured chat for Gemini requires function calling, which is not implemented for {self.model}"
        )