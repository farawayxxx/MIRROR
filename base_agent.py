import time
from functools import wraps
from openai import OpenAI, AzureOpenAI
from tenacity import retry

def retry(max_attempts=3, delay=5):
    """
    A decorator for retrying a function call with a specified delay in case of exception.

    :param max_attempts: The maximum number of attempts. Default is 3.
    :param delay: The delay (in seconds) between attempts. Default is 1.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts}/{max_attempts} failed: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class BaseAgent:
    def __init__(self, model, azure_endpoint, api_key, api_version):
        self.model = model
        if "gpt" in model:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
                )
        else:
            print("MODEL_NAME_ERROR!!!")
        


    @retry(max_attempts=3, delay=5)
    def query_openai(self, system_prompt="", user_prompt="", json_mode=False, functions=None, function_call=None, temp=0.7, top_p=0.9):
        if functions == None:
            completion = self.client.chat.completions.create(
                model=self.model, # gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-4-0613, gpt-4-1106-preview, gpt-3.5-turbo-1106, gpt-4-turbo-2024-04-09, gpt-4o-2024-05-13
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                top_p=0.9,
                functions=functions,
                function_call=function_call
                )
        else:
            completion = self.client.chat.completions.create(
                model=self.model, # gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-4-0613, gpt-4-1106-preview, gpt-3.5-turbo-1106, gpt-4-turbo-2024-04-09, gpt-4o-2024-05-13
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                top_p=0.9,
                functions=functions,
                function_call="auto"
                )
        self.log_usage(completion)
        
        if functions == None:
            openai_response = completion.choices[0].message.content
        else:
            openai_response = completion.choices[0].message
        return openai_response