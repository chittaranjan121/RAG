import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional,List 
from pydantic import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]
llm=ChatOpenAI()

class Person(BaseModel):
    """Information about a person"""
    name:Optional[str]=Field(
        default=None,description="The Name of the Person"
    )
    lastname:Optional[str]=Field(
        default=None,description="The lastname of the Person"
    )
    country:Optional[str]=Field(
        default=None,description="The country of the person"
    )

class data(BaseModel):
    """Extracted about 2 people"""
    people: List[Person]
prompt=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "you are an expert extraction algorithim"
                "only extract the relevant information"
                "if you don't know the value asked to extract"
                "return null for the attributes value"
            ),
            ("human","{text}")
        ]
    )
chain=prompt|llm.with_structured_output(schema=data)

comment="John Doe is a software engineer from Canada.Meanwhile Stankey kennedy from Australia is a data scientist"
res=chain.invoke({"text":comment})
print(res)

