import openai
import os
import logging
import backoff
from pydantic import BaseModel
from openai import OpenAI

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

class ModelOutput(BaseModel):
    answer: bool
    explanation: str

class AgentDecision(BaseModel):
    exact_match: bool
    explanation: str

class NewQuery(BaseModel):
    new_query: str

class Tasks():
    def __init__(self):
        self.chat_completion_model = "gpt-4o"  

    def generate_message(self, task, query, title, description, label, agent_decision=None):

        task1 = [
        {
            "role": "system",
            "content": "You are verifying whether a product truly matches a user’s query according to the KDD Cup 2022 definition of an Exact match, labeled as an “Exact match” (label “E”) "
        },
        {
            "role": "user",
            "content": f"""
            According to the KDD Cup 2022 definition, a product is labeled “E”, that definition states:
            An item is relevant for the query and satisfies all the query specifications (e.g., brand, size, color).

            # Task 1
            *-extract query specifications: features, size, color, quantity, brand, etc
            *-standardize measurements for query and product title and description, example 8.5 inches is the same as 8-1/2.

            * Important
            Query specifications only relevant if the user query explicity requests it.
            For example query does not contain brand; therefore product brand may be irrelevant.
            Reasoning: Brand and Product name can be different since Brand may have many different product lines.

            Different representation of the same size does not impact classification of exact match. 
            Example:
            User's query size specified is '8.5 x 11', while the product lists the size as '8-1/2 x 11', these are an exact match on size.

            ## However, please also follow these additional labeling guidelines:

            *If the product information does not mention a specification in the query, it is acceptable to keep the label as “E”.

            *If the product includes additional features or bundled items beyond what was requested in the query, you may still consider the label to be “E”.
            
            Query: {query}
            
            Product Details:
            
            Title: {title}
            
            Description: {description}

            label: {label}
            
            Task:
            
            Does this product satisfy all the specifications of the user’s query, Answer True or False.
            
            If you answer False, please explain which query requirement(s) the product fails to meet.
            """
        }
        ]

        task2 = [
        {
            "role": "system",
            "content": "You are verifying whether an agent has made the correct decision."
        },
        {
            "role": "user",
            "content": f"""
            An agent was tasked with verifying whether a product truly matches a user’s query according to the KDD Cup 2022 definition of an Exact match, labeled as an “Exact match” (label “E”)
            According to the KDD Cup 2022 definition, a product is labeled “E”, that definition states:
            An item is relevant for the query and satisfies all the query specifications (e.g., brand, size, color).

            ## The agent also followed these additional labeling guidelines:

            *If the product information does not mention a specification in the query, it is acceptable to keep the label as “E”.

            *If the product includes additional features or bundled items beyond what was requested in the query, you may still consider the label to be “E”.
            
            Query: {query}
            
            Product Details:
            
            Title: {title}
            
            Description: {description}

            label: {label}

            agent decision: {agent_decision}
            
            Task:
            
            Was it an exact_match, answer True or False
            Give an explanation
            
            """
        }
        ]

        task3 = [
        {
            "role": "system",
            "content": "You are a product search assistant that helps rewrite customer queries so they exactly match a specific product."
        },
        {
            "role": "user",
            "content": f"""
            Step 3: Query Reformulation Prompt (Standalone or Follow-up)
            You are tasked with reformulating a user query so that it would result in a valid Exact match ("E") for a given product, according to the following standard:
            
            A query and product are considered an “Exact match” if the product is relevant and satisfies all the query specifications.
            
            You will be given:
            
            The original user query
            
            The product title
            
            The product description
            
            Please generate a concise, natural-language search query that accurately describes the product and would qualify for an “E” label under the KDD Cup 2022 definition.
            
            Do not reuse incorrect elements from the original query. Instead, base the reformulated query entirely on what is actually present in the product title and description.
            
            Original Query:
            {query}
            
            Product Title:
            {title}
            
            Product Description:
            {description}
            
            Task: Write a new query that would result in this product being labeled an “Exact match”.
            """
        }
        ]
        
        lookup = {'task1': task1,
                  'task2': task2,
                  'task3': task3}
        
        return lookup[task]

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def chat_completion(self, task, query, title, description, label, agent_decision):
        if task == 'task1':
            response_format = ModelOutput
        elif task == 'task2':
            response_format = AgentDecision
        else:
            response_format = NewQuery
        messages = self.generate_message(task, query, title, description, label, agent_decision)
        response = client.beta.chat.completions.parse( 
            model=self.chat_completion_model,
            messages=messages,
            response_format=response_format,
            
        )
        return response.choices[0].message.parsed

class ClassifierOutput(BaseModel):
    label: str
    explanation: str

class Classifier():
    def __init__(self):
        self.chat_completion_model = "gpt-4o" 

    def generate_message(self, task, query, title, description):

        task1 = [
        {
            "role": "system",
            "content": "You are and expert in classfying whether a product truly matches a user’s query according to the KDD Cup 2022 definition of an Exact match, labeled as an “Exact match” (label “E”) "
        },
        {
            "role": "user",
            "content": f"""
            
            According to the KDD Cup 2022 definition, a product is labeled “E”, that definition states:
            An item is relevant for the query and satisfies all the query specifications (e.g., brand, size, color).

            Example:
            
            Exact (E): the item is relevant for the query, and satisfies all the query specifications (e.g., water bottle matching all attributes of a query “plastic water bottle 24oz”, such as material and size

            ## However, please also follow these additional labeling guidelines:

            *If the product information does not mention a specification in the query, it is acceptable to keep the label as “E”.

            *If the product includes additional features or bundled items beyond what was requested in the query, you may still consider the label to be “E”.
            
            Query: {query}
            
            Product Details:
            
            Title: {title}
            
            Description: {description}
            
            Task:
            
            Does this product satisfy all the specifications of the user’s query? Answer according to the following classes:
            Exact (E): the item is relevant for the query, and satisfies all the query specifications (e.g., water bottle matching all attributes of a query “plastic water bottle 24oz”, such as material and size)

            Substitute (S): the item is somewhat relevant: it fails to fulfill some aspects of the query but the item can be used as a functional substitute (e.g., fleece for a “sweater” query)
            
            Complement (C): the item does not fulfill the query, but could be used in combination with an exact item (e.g., track pants for “running shoe” query)
            
            Irrelevant (I): the item is irrelevant, or it fails to fulfill a central aspect of the query (e.g. socks for a “pant” query)

            * Important
            Product Brand is only relevant if the user query explicity requests it.
            Additonally Brand and Product name can be different since Brand may have many different product lines.
            Example:
            ACDelco is a product line under Powermax
            
            If you answer False, please explain which query requirement(s) the product fails to meet.
            """
        }
        ]

        lookup = {'task1': task1}
        return lookup[task]

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def chat_completion(self, task, query, title, description):
        messages = self.generate_message(task, query, title, description)
        response = client.beta.chat.completions.parse( 
            model=self.chat_completion_model,
            messages=messages,
            response_format=ClassifierOutput,
            
        )
        return response.choices[0].message.parsed

class Controller():
    def __init__(self):
        self.task_manager = Tasks()

    def run_tasks(self, query, title, description, label):
        task3_new_query = None
        task1 = self.task_manager.chat_completion('task1', query, title, description, label, None)
        task2 = self.task_manager.chat_completion('task2', query, title, description, label, task1)
        if not task2.exact_match:
            task3 = self.task_manager.chat_completion('task3', query, title, description, None, None)
            task3_new_query = task3.new_query
        return task1.answer, task1.explanation, task2.exact_match, task2.explanation, task3_new_query
            