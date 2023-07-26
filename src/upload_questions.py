from pathlib import Path

import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from utils import log_into_wandb, info, error

config_path = Path('../config')


@hydra.main(version_base='1.3', config_path=str(config_path), config_name='params.yaml')
def main(params: DictConfig) -> None:
    wandb_project = params.wandb.project

    load_dotenv(config_path / '.env')
    log_into_wandb()

    params = params.upload_questions

    if not params.get('qa_artifact'):
        error(f'parameter `upload_questions.qa_artifact` not found')
        exit(-1)

    data_path = Path('../data')
    qa_path = data_path / 'qa'

    if params.get('qa_filenames'):
        qa_files = [qa_path / filename for filename in params.qa_filenames]
        info(f'Uploading Q&A from {len(qa_files)} file(s) listed in parameter `upload_questions.qa_filenames`')
    else:
        qa_files = list(qa_path.glob('*.yaml'))
        info(f'Uploading Q&A from {len(qa_files)} file(s), all `.yaml` files in {qa_path}')

    with wandb.init(project=wandb_project,
                    notes="Uploads artifact with questions for the model",
                    config={'params': OmegaConf.to_object(params)}):
        # qa_files = [qa_path / filename for filename in params.qa_filenames]
        qa_artifact = wandb.Artifact(name=params.qa_artifact,
                                     type='dataset',
                                     description='Questions to be submitted to the language model, optionally with \
                                     expected answers.'
                                     )
        for qa_file in qa_files:
            qa_artifact.add_file(qa_file)
        wandb.log_artifact(qa_artifact)


if __name__ == '__main__':
    main()
