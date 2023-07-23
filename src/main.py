import pickle
from pathlib import Path
from time import time

import tiktoken
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain.vectorstores import Qdrant
from omegaconf import OmegaConf
from tqdm import tqdm

from utils import info

config_path = Path('../config')
load_dotenv(config_path / '.env')

data_path = Path('../data')
dataset_path = data_path / 'dataset/dataset.pickle'

# Load the questions and answers for model validation
movies_qa_path = data_path / 'movies_qa.yaml'
movies_qa = OmegaConf.load(movies_qa_path)
movies_qa = [{'Question': item.Question.rstrip('\n )'),
              'Answer': item.Answer.rstrip('\n ')} for i, item in enumerate(movies_qa)]

embeddings_path = data_path / 'qdrant_vector_store.pickle'
# embeddings_path = data_path / 'chroma'


# If embeddings have been previously saved, then reload them...
if embeddings_path.exists():
    info(f'Reloading embeddings from {embeddings_path}')
    # vectordb = Chroma(persist_directory=str(embeddings_path), embedding_function=embedding)
    with open(embeddings_path, 'rb') as pickled:
        vectordb = pickle.load(pickled)
# Otherwise, chunk the corpus of text, compute the embeddings and save them
else:
    info('Chunking text')
    tokenizer = tiktoken.get_encoding('cl100k_base')  # This is right for GPT-3.5


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

    embedding = OpenAIEmbeddings()

    # Encode the chunked text as embeddings
    info(f'Making embeddings and saving them into {embeddings_path}')

    vectordb = Qdrant.from_documents(
        docs,
        embedding,
        tokenizer=tokenizer,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="my_documents",
    )

    with open(embeddings_path, 'bw') as pickled:
        pickle.dump(vectordb, pickled, protocol=pickle.HIGHEST_PROTOCOL)

    end_time = time()
    print(f'Completed embeddings in {int(end_time - end_chunking)} sec')

llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

prompt = ChatPromptTemplate.from_template("{query}")
qa_chain_no_context = LLMChain(llm=llm, prompt=prompt)

qa_chain_with_context = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
template = """Use also the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use four sentences maximum. Keep the answer as concise as possible. Always end the answer with 'Thank you for asking!' 
{context}
Question: {question}
Helpful Answer:"""
qa_chain_prompt = PromptTemplate.from_template(template)

metadata_field_info = [
    AttributeInfo(name='title',
                  description='The movie title',
                  type='string'),
    AttributeInfo(name='year',
                  description='The movie release year',
                  type='integer'),
    AttributeInfo(name='id',
                  description='The movide unique ID within Wikipedia',
                  type='integer'),
    AttributeInfo(name='revision_id',
                  description='The movie unique revision ID within Wikipedia',
                  type='integer')
]
document_content_description = 'The movie plot or synopsis'
retriever = SelfQueryRetriever.from_llm(llm,
                                        vectordb,
                                        document_content_description,
                                        metadata_field_info,
                                        verbose=True)

qa_chain_with_filter = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_chain_prompt}
)


queries = [{'query': qa['Question']} for qa in movies_qa]

results_no_context = []
results_with_filter = []
results_with_context = []
for query in tqdm(queries):
    res_no_context = qa_chain_no_context.run(query)
    results_no_context.append(res_no_context)
    res_with_context = qa_chain_with_context(query)
    results_with_context.append(res_with_context)
    res_with_filter = qa_chain_with_filter(query)
    results_with_filter.append(res_with_filter)



movies_qa_processed = []
for qa, answer_no_context, answer_with_context, answer_with_filter in zip(movies_qa,
                                                                          results_no_context,
                                                                          results_with_context,
                                                                          results_with_filter):
    source_chunks = [doc.page_content for doc in answer_with_filter['source_documents']] if answer_with_filter.get(
        'source_documents') else None

    movies_qa_processed.append({'Question': qa['Question'],
                                'Answer': qa['Answer'],
                                'Answer_no_context': answer_no_context,
                                'Answer_with_context': answer_with_context['result'],
                                'Answer_with_filter': answer_with_filter['result'],
                                'Context': source_chunks
                                })

processed_qa_file = data_path / 'processed_qa.yaml'
OmegaConf.save(movies_qa_processed, processed_qa_file)

"""
TODO

Checkout why some contexts have leading and trailing \ at every line in the output .yaml
Try out compression after retrieval using a ContextualCompressionRetriever
Try other vector stores such as FAISS and Milvus
Try with refine chain type

Leverage metadata -> Done
Speed-up queries to OpenAI -> Done, but undone because makes debugging much harder
"""
