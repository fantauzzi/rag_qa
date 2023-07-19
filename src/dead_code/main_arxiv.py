import pickle
import shutil
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

config_path = Path('../config')
load_dotenv(config_path / '.env')

dataset_path = Path('~/dataset/arxiv')
data_path = Path('../data')
pickled_path = data_path / 'pickled'
from langchain.document_loaders import PyPDFLoader

pdf_file = dataset_path / '2306.17766.pdf'
pickled_file = (pickled_path / pdf_file.name).with_suffix('.pickle')
if pickled_file.exists():
    with open(pickled_file, 'br') as pickled:
        pages = pickle.load(pickled)
else:
    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()
    with open(pickled_file, 'wb') as pickled:
        pickle.dump(pages, pickled, pickle.HIGHEST_PROTOCOL)

tokenizer = tiktoken.get_encoding('cl100k_base')


def tokenized_length(s: str) -> int:
    """
    Returns the length in tokens in a given string after tokenization.
    :param s: the given string.
    :return: the count of tokens in the tokenized string.
    """
    tokenized = tokenizer.encode(s)
    return len(tokenized)


splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                          chunk_overlap=150,
                                          separators=["\n\n", "\n", "(?<=\. )", " ", ""],
                                          length_function=tokenized_length)
chunks = splitter.split_documents(pages)

embeddings_path = data_path / 'chroma'
if embeddings_path.exists():
    shutil.rmtree(embeddings_path)
embeddings_path.mkdir(exist_ok=False)

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=str(embeddings_path)
)

question = 'How can researchers make changes to learning tasks to study their effects on reinforcement learning algorithm performance?'
results = vectordb.max_marginal_relevance_search(query=question, k=3)

llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever())
result = qa_chain({"query": question})
print(result['result'])
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
qa_chain_prompt = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_chain_prompt}
)

result = qa_chain({"query": question})

print(result['result'])
print(result['source_documents'][0])

"""
TODO
Consider concatenating all pages in a given PDF so that chunks can then extend across pages; it looks like that requires
    giving up on maintaining the original page number in the chunk metadata.
Remove page numbers, perhaps headers and footers if possible, or give h/f their own place at the end of the article (so
    they don't come in the way of regular text chunks).
Try out compression after retrieval using a ContextualCompressionRetriever
Try other vector stores such as FAISS and Milvus
Try with refine chain type
"""
