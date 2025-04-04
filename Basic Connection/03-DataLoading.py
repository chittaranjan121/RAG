import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader,CSVLoader,PyPDFLoader,WikipediaLoader

load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]

###Text Loader
loader=TextLoader(r"E:\Git-Repos\RAG\Basic Connection\cloudpandith.txt")
load_file=loader.load()
print(load_file)

#### Loading csv file
loader_1=CSVLoader(r"E:\Git-Repos\RAG\Basic Connection\countries.csv")
loader_csv=loader_1.load()
print(loader_csv)

### PDF Loader
loader_2=PyPDFLoader(r"E:\Git-Repos\RAG\Basic Connection\Rogers-2025-03-19.pdf")
load_pdf=loader_2.load_and_split()
print(load_pdf[0].page_content)

#### Wikipedia Loader
loader_3=WikipediaLoader(query='tesla',load_max_docs=1)
loaded_wiki=loader_3.load()[0].page_content
print(loaded_wiki)