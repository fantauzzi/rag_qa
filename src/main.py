import pickle
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
from tqdm import tqdm

from utils import info

config_path = Path('../config')
load_dotenv(config_path / '.env')

data_path = Path('../data')
dataset_path = data_path / 'dataset/dataset.pickle'

movies_qa_path = data_path / 'movies_qa.yaml'
movies_qa = OmegaConf.load(movies_qa_path)
movies_qa = [{'Question': item.Question.rstrip('\n )'), 'Answer': item.Answer.rstrip('\n ')} for item in movies_qa]

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

if embeddings_path.exists():
    info(f'Reloading embeddings from {embeddings_path}')
    vectordb = Chroma(persist_directory=str(embeddings_path), embedding_function=embedding)
else:
    info('Chunking text')
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                              chunk_overlap=150,
                                              separators=["\n\n", "\n", "(?<=\. )", " ", ""],
                                              length_function=tokenized_length)


    with open(dataset_path, 'rb') as dataset_f:
        dataset = pickle.load(dataset_f)

    start_time = time()
    docs = [(text, dataset['metadata'].loc[pageid].to_dict()) for (pageid, text) in dataset['data'].items()]
    docs, metadata = list(zip(*docs))
    docs = splitter.create_documents(docs, metadata)

    chunks = splitter.split_documents(docs)
    end_chunking = time()

    info(f'Completed chunking in {int(end_chunking - start_time)} sec')

    info(f'Making embeddings and saving them into {embeddings_path}')
    embeddings_path.mkdir(exist_ok=False)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(embeddings_path)
    )
    end_time = time()
    print(f'Completed embeddings in {int(end_time - end_chunking)} sec')

movies_qa_processed = []
for qa in tqdm(movies_qa):
    question = qa['Question']
    results = vectordb.max_marginal_relevance_search(query=question, k=3)

    llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever())
    answer_no_context = qa_chain({"query": question})

    # print(answer_no_context['result'])
    template = """Use also the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use four sentences maximum. Keep the answer as concise as possible. 
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

    answer_with_context = qa_chain({"query": question})

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
Consider concatenating all pages in a given PDF so that chunks can then extend across pages; it looks like that requires
    giving up on maintaining the original page number in the chunk metadata.
Remove page numbers, perhaps headers and footers if possible, or give h/f their own place at the end of the article (so
    they don't come in the way of regular text chunks).
Try out compression after retrieval using a ContextualCompressionRetriever
Try other vector stores such as FAISS and Milvus
Try with refine chain type
"""
