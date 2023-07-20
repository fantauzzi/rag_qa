import pickle
from multiprocessing import Pool
from pathlib import Path
from time import time

import tiktoken
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from omegaconf import OmegaConf

from utils import info

config_path = Path('../config')
load_dotenv(config_path / '.env')

data_path = Path('../data')
dataset_path = data_path / 'dataset/dataset.pickle'

# Load the questions and answers for model validation
movies_qa_path = data_path / 'movies_qa.yaml'
movies_qa = OmegaConf.load(movies_qa_path)
movies_qa = [{'Question': item.Question.rstrip('\n )'), 'Answer': item.Answer.rstrip('\n ')} for item in movies_qa]

# Instantiate the tokenizer
tokenizer = tiktoken.get_encoding('cl100k_base')  # This is right for GPT-3.5


def tokenized_length(s: str) -> int:
    """
    Returns the length in tokens in a given string after tokenization.
    :param s: the given string.
    :return: the count of tokens in the tokenized string.
    """
    tokenized = tokenizer.encode(s)
    return len(tokenized)


embeddings_path = data_path / 'chroma'
embedding = OpenAIEmbeddings()

# If embeddings have been previously saved, then reload them...
if embeddings_path.exists():
    info(f'Reloading embeddings from {embeddings_path}')
    vectordb = Chroma(persist_directory=str(embeddings_path), embedding_function=embedding)
# Otherwise, chunk the corpus of text, compute the embeddings and save them
else:
    info('Chunking text')
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                              chunk_overlap=150,
                                              separators=["\n\n", "\n", "(?<=\. )", " ", ""],
                                              length_function=tokenized_length)

    # Load the corpus of text to be encoded as embeddings
    with open(dataset_path, 'rb') as dataset_f:
        dataset = pickle.load(dataset_f)

    # Convert the text and its metadata into langchain Documents
    start_time = time()
    docs = [(text, dataset['metadata'].loc[pageid].to_dict()) for (pageid, text) in dataset['data'].items()]
    docs, metadata = list(zip(*docs))
    docs = splitter.create_documents(docs, metadata)

    # Do the chunking
    chunks = splitter.split_documents(docs)

    end_chunking = time()
    info(f'Completed chunking in {int(end_chunking - start_time)} sec')

    # Encode the chunked text as embeddings
    info(f'Making embeddings and saving them into {embeddings_path}')
    embeddings_path.mkdir(exist_ok=False)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(embeddings_path)
    )
    end_time = time()
    print(f'Completed embeddings in {int(end_time - end_chunking)} sec')

llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)
qa_chain_no_context = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
template = """Use also the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use four sentences maximum. Keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
qa_chain_prompt = PromptTemplate.from_template(template)
qa_chain_with_context = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_chain_prompt}
)
movies_qa_processed = []
"""
for qa in tqdm(movies_qa):
    question = qa['Question']

    answer_no_context = qa_chain_no_context({"query": question})

    answer_with_context = qa_chain_with_context({"query": question})

    source_chunks = [doc.page_content for doc in answer_with_context['source_documents']]
    movies_qa_processed.append({'Question': qa['Question'],
                                'Answer': qa['Answer'],
                                'Answer_without_context': answer_no_context['result'],
                                'Answer_with_context': answer_with_context['result'],
                                'Context': source_chunks
                                })
"""

queries = [{'query': qa['Question']} for qa in movies_qa]


def run_query_no_context(query: dict) -> dict:
    res = qa_chain_no_context(query)
    info(f'Query with no context completed')
    return res


def run_query_with_context(query: dict) -> dict:
    res = qa_chain_with_context(query)
    info(f'Query with context completed')
    return res


pool_size = min(16, len(queries))
with Pool(pool_size) as pool:
    results_no_context = pool.map(run_query_no_context, queries)
    results_with_context = pool.map(run_query_with_context, queries)

for qa, answer_no_context, answer_with_context in zip(movies_qa, results_no_context, results_with_context):
    source_chunks = [doc.page_content for doc in answer_with_context['source_documents']]

    movies_qa_processed.append({'Question': qa['Question'],
                                'Answer': qa['Answer'],
                                'Answer_without_context': answer_no_context['result'],
                                'Answer_with_context': answer_with_context['result'],
                                'Context': source_chunks
                                })

processed_qa_file = data_path / 'processed_qa.yaml'
OmegaConf.save(movies_qa_processed, processed_qa_file)

"""
TODO
Speed-up queries to OpenAI
Leverage metadata
Checkout why some contexts have leading and trailing \ at every line in the output .yaml

Try out compression after retrieval using a ContextualCompressionRetriever
Try other vector stores such as FAISS and Milvus
Try with refine chain type
"""
