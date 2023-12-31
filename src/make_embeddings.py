import pickle
from pathlib import Path
from time import time

import hydra
import tiktoken
import wandb
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from omegaconf import DictConfig, OmegaConf

from utils import info, error, log_into_wandb

tokenizer = tiktoken.get_encoding('cl100k_base')  # This is right for GPT-3.5


def tokenized_length(s: str) -> int:
    """
    Returns the length in tokens in a given string after tokenization.
    :param s: the given string.
    :return: the count of tokens in the tokenized string.
    """
    tokenized = tokenizer.encode(s)
    return len(tokenized)


config_path = Path('../config')


@hydra.main(version_base='1.3', config_path=str(config_path), config_name='params.yaml')
def main(params: DictConfig) -> None:
    wandb_project = params.wandb.project
    params = params.make_embeddings
    load_dotenv(config_path / '.env')
    log_into_wandb()

    if not params.get('vector_store_artifact'):
        error(f'Parameter `make_embeddings.vector_store_artifact` not set')
        exit(-1)

    data_path = Path('../data')
    dataset_file = data_path / params.pickled_dataset
    embeddings_file = data_path / params.vector_store_filename

    if embeddings_file.exists():
        info(f'Going to overwrite embeddings in {embeddings_file}')
    info('Chunking text')

    run = wandb.init(project=wandb_project,
                     notes="Chunks the text in the dataset, converts the chunks into embeddings and save the embeddings \
                        into a vectore store along with their metadata",
                     config={'params': OmegaConf.to_object(params)})

    if params.dataset_artifact:
        dataset_artifact = run.use_artifact(params.dataset_artifact)
        info(f'Downloading dataset artifact into {data_path}')
        dataset_artifact.download(str(data_path))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                              chunk_overlap=150,
                                              separators=["\n\n", "\n", "(?<=\. )", " ", ""],
                                              length_function=tokenized_length)

    # Load the corpus of text to be encoded as embeddings
    with open(dataset_file, 'rb') as dataset_f:
        dataset = pickle.load(dataset_f)

    # Convert the text and its metadata into langchain Documents
    start_time = time()
    docs = [(text, dataset['metadata'].loc[pageid].to_dict()) for (pageid, text) in dataset['data'].items()]
    docs, metadata = list(zip(*docs))
    docs = splitter.create_documents(docs, metadata)

    # Do the chunking
    chunks = splitter.split_documents(docs)
    pickled_chunks_file = None
    if params.get('pickled_chunks_filename') is not None and params.pickled_chunks_filename:
        pickled_chunks_file = data_path / params.pickled_chunks_filename
        info(f'Saving chunked docs with their metadata in {pickled_chunks_file}')
        with open(pickled_chunks_file, 'bw') as pickled:
            pickle.dump(chunks, pickled, protocol=pickle.HIGHEST_PROTOCOL)

    end_chunking = time()
    info(f'Completed chunking in {int(end_chunking - start_time)} sec')
    # chunks = chunks[:10]

    embedding = OpenAIEmbeddings()

    # Encode the chunked text as embeddings
    info(f'Making embeddings and saving them into {embeddings_file}')

    vectordb = Qdrant.from_documents(
        chunks,
        embedding,
        tokenizer=tokenizer,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="my_documents",
    )

    with open(embeddings_file, 'bw') as pickled:
        pickle.dump(vectordb, pickled, protocol=pickle.HIGHEST_PROTOCOL)

    end_time = time()
    print(f'Completed embeddings in {int(end_time - end_chunking)} sec')
    dataset_artifact = wandb.Artifact(name=params.vector_store_artifact,
                                      type='dataset',
                                      description='Output of the chunking end encoding into embedding of the \
                                                  source dataset')
    if pickled_chunks_file:
        dataset_artifact.add_file(str(pickled_chunks_file))
    dataset_artifact.add_file(embeddings_file)
    wandb.log_artifact(dataset_artifact)
    run.finish()


if __name__ == '__main__':
    main()
