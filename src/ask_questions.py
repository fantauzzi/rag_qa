import os
import pickle
from pathlib import Path

import hydra
import wandb
from dotenv import load_dotenv
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
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
    retriever = vectordb.as_retriever(search_type='similarity', k=3)  # TODO try mmr and similarity_score_threshold

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

    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""

    results_from_context = []
    for query in tqdm(queries):
        relevant_documents = retriever.get_relevant_documents(query['query'])
        relevant_documents_content = []
        for relevant_doc in relevant_documents:
            metadata_and_content = f'Movie title: {relevant_doc.metadata["title"]}\nMovie year: {relevant_doc.metadata["year"]}\nMovie plot:\n{relevant_doc.page_content}'
            relevant_documents_content.append(metadata_and_content)
        context = '\n'.join(relevant_documents_content)
        qa_chain_prompt = PromptTemplate.from_template(template)
        filled_in_prompt = qa_chain_prompt.format(context=context, question=query['query'])
        response_from_context = llm.call_as_llm(filled_in_prompt)
        results_from_context.append(response_from_context)

    movies_qa_processed = []
    for qa, result_from_context in zip(movies_qa,
                                       results_from_context):
        source_chunks = None
        movies_qa_processed.append({'Question': qa['Question'],
                                    'Answer': qa['Answer'],
                                    'Answer_from_context': result_from_context,
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
