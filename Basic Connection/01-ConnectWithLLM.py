import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]
llmmodel=ChatOpenAI()

#### Invoke a simple question to LLM and gets teh response
response=llmmodel.invoke("Tell me about Virat Kohli")
print(response.content)
print("")

#### Streaming Output
print("###################################")
for i in llmmodel.stream("Tell me one fun fact of Virat Kohli"):
    print(i.content,end="",flush=True)


#### Now lets create a chatModel prompt
print("")
print("###################################")
print("Now the chat prompt Model")
print("")

messages=[
    ("system","You are a cricket expert in the history and having huge knowledge on cricket"),
    ("human","tell me curious thing about Sourav Ganguly")
]

res=llmmodel.invoke(messages)
print(res.content)

###To check metadata of content
print(res.response_metadata)
print("###################################")
print("")
print(res.schema())


print("###################################")
#####Older way
messages_1=[
    SystemMessage(content="You are a cricket expert and having huge knowledge on cricket"),
    HumanMessage(content="Tell me date of birth of sourav ganguly")
]

res_1=llmmodel.invoke(messages_1)
print(res_1.content)