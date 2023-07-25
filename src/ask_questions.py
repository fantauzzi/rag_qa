import os
import pickle
from pathlib import Path

import hydra
import wandb
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils import info, log_into_wandb

config_path = Path('../config')


@hydra.main(version_base='1.3', config_path=str(config_path), config_name='params.yaml')
def main(params: DictConfig) -> None:
    wandb_project = params.wandb.project

    load_dotenv(config_path / '.env')
    log_into_wandb()

    wandb_run = wandb.init(project=wandb_project,
                           notes="Asks questions to the language model and collects answers",
                           config={'params': OmegaConf.to_object(params)})

    langchain_project = os.environ.get('LANGCHAIN_PROJECT')
    if not langchain_project:
        langchain_project = f'{wandb_project}/{wandb_run.name}'
        os.environ['LANGCHAIN_PROJECT'] = langchain_project

    params = params.ask_questions

    data_path = Path('../data')
    # dataset_path = data_path / 'dataset' / params.pickled_dataset
    embeddings_path = data_path / params.vector_store_filename

    if params.vector_store_artifact:
        vector_store_artifact = wandb_run.use_artifact(params.vector_store_artifact)
        info(f'Downloading vector store artifact into {str(data_path.parent)}')
        vector_store_artifact.download(root=str(data_path.parent))

    # Load the questions and answers for model validation
    movies_qa_path = data_path / params.qa_filenames[0]
    movies_qa = OmegaConf.load(movies_qa_path)
    movies_qa = [{'Question': item.Question.rstrip('\n )'),
                  'Answer': item.Answer.rstrip('\n ')} for i, item in enumerate(movies_qa)]

    # If embeddings have been previously saved, then reload them...
    with open(embeddings_path, 'rb') as pickled:
        vectordb = pickle.load(pickled)

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

    langsmith_metadata = {'wandb_project': wandb_project,
                          'wandb_run_id': wandb_run.id,
                          'wandb_run_name': wandb_run.name,
                          'wandb_run_path': wandb_run.path}
    for query in tqdm(queries):
        res_no_context = qa_chain_no_context.run(query, metadata=langsmith_metadata)
        results_no_context.append(res_no_context)
        res_with_context = qa_chain_with_context(query, metadata=langsmith_metadata)
        results_with_context.append(res_with_context)
        res_with_filter = qa_chain_with_filter(query, metadata=langsmith_metadata)
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

    processed_qa_path = movies_qa_path.with_suffix('.processed.yaml')
    info(f'Saving queries and results in {processed_qa_path}')
    OmegaConf.save(movies_qa_processed, processed_qa_path)

    dataset_artifact = wandb.Artifact(name='questions_and_answers',
                                      type='conversation',
                                      description='Questions from the dataset and their answers from the language model'
                                      )
    dataset_artifact.add_file(processed_qa_path)
    wandb.log_artifact(dataset_artifact)
    wandb_run.finish()


if __name__ == '__main__':
    main()

"""
TODO

Checkout why some contexts have leading and trailing \ at every line in the output .yaml
Try out compression after retrieval using a ContextualCompressionRetriever
Try other vector stores such as FAISS and Milvus
Try with refine chain type

Leverage metadata -> Done
Speed-up queries to OpenAI -> Done, but undone because makes debugging much harder
"""
