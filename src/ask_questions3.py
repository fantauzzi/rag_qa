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

from utils import info, error, log_into_wandb

config_path = Path('../config')


@hydra.main(version_base='1.3', config_path=str(config_path), config_name='params.yaml')
def main(params: DictConfig) -> None:
    wandb_project = params.wandb.project

    load_dotenv(config_path / '.env')
    log_into_wandb()

    params = params.ask_questions

    if not params.get('processed_qa_filename'):
        error('Parameter `processed_qa_filename` not found')
        exit(-1)

    if not params.get('processed_qa_artifact'):
        error('Parameter `processed_qa_artifact` not found')
        exit(-1)

    data_path = Path('../data')
    qa_path = data_path / 'qa'
    llm_name = params.llm_name
    temperature = params.temperature
    processed_qa_file = data_path / params.processed_qa_filename
    processed_qa_artifact_name = params.processed_qa_artifact

    wandb_run = wandb.init(project=wandb_project,
                           notes="Asks questions to the language model and collects answers",
                           config={'params': OmegaConf.to_object(params)})

    # TODO stick the below into its own subroutine
    langchain_project = os.environ.get('LANGCHAIN_PROJECT')
    if not langchain_project:
        langchain_project = f'{wandb_project}/{wandb_run.name}'
        os.environ['LANGCHAIN_PROJECT'] = langchain_project

    embeddings_file = data_path / params.vector_store_filename

    if params.vector_store_artifact:
        vector_store_artifact = wandb_run.use_artifact(params.vector_store_artifact)
        info(f'Downloading vector store artifact into {data_path}')
        vector_store_artifact.download(root=str(data_path))

    with open(embeddings_file, 'rb') as pickled:
        vectordb = pickle.load(pickled)

    llm = ChatOpenAI(model_name=llm_name, temperature=temperature)

    prompt_no_context = ChatPromptTemplate.from_template("{query}")
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
    self_query_retriever = SelfQueryRetriever.from_llm(llm,
                                                       vectordb,
                                                       document_content_description,
                                                       metadata_field_info,
                                                       verbose=True)

    qa_chain_no_context = LLMChain(llm=llm, prompt=prompt_no_context)
    qa_chain_with_context = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
    qa_chain_with_filter = RetrievalQA.from_chain_type(llm,
                                                       retriever=self_query_retriever,
                                                       return_source_documents=True,
                                                       chain_type_kwargs={"prompt": qa_chain_prompt}
                                                       )

    params_qa_artifact = params.get('qa_artifact')
    if params_qa_artifact:
        qa_artifact = wandb_run.use_artifact(params.qa_artifact)
        info(f'Downloading Q&A artifact into {qa_path}')
        qa_artifact.download(root=str(qa_path))
    if params_qa_artifact or not params.get('qa_filenames'):
        qa_files = list(qa_path.glob('*.yaml'))
    else:
        qa_files = [qa_path / filename for filename in params.qa_filenames]

    # Load the questions and answers for model validation
    movies_qa_list = [OmegaConf.load(qa_file) for qa_file in qa_files]
    movies_qa = [{'Question': item.Question.rstrip('\n )'),
                  'Answer': item.Answer.rstrip('\n ')} for movies_qa in movies_qa_list for item in movies_qa]

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

    info(f'Saving queries and results in {processed_qa_file}')
    OmegaConf.save(movies_qa_processed, processed_qa_file)

    processed_qa_artifact = wandb.Artifact(name=processed_qa_artifact_name,
                                           type='conversation',
                                           description='Questions from the dataset and their answers from the language model'
                                           )
    processed_qa_artifact.add_file(processed_qa_file)
    wandb.log_artifact(processed_qa_artifact)
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
