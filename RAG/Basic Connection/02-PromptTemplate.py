import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate

load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]
llmmodel=ChatOpenAI()

prompt_template=PromptTemplate.from_template(
    "Tell me a {adjective} story about {topic}."
)
llmmodelformat=prompt_template.format(
    adjective="curious",
    topic="The horror movie"
)
res=llmmodel.invoke(llmmodelformat)
print(res.content)

##### Now lets check the ChatPromptTemplate

chat_template=ChatPromptTemplate.from_template(    
    "You are a {profession} expert on {topic}. "
    "Hello Mr. {profession}, can you please answer a question? "
    "Sure. {user_input}"
    
)

messages=chat_template.format_messages(
    profession="Historian",
    topic="The Kennedy family",
    user_input="How many grandchilder Kennedy family has"
)


response=llmmodel.invoke(messages)
print(response.content)