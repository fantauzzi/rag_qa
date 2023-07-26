from pathlib import Path

import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from utils import log_into_wandb

config_path = Path('../config')


@hydra.main(version_base='1.3', config_path=str(config_path), config_name='params.yaml')
def main(params: DictConfig) -> None:
    wandb_project = params.wandb.project

    load_dotenv(config_path / '.env')
    log_into_wandb()

    params = params.upload_questions

    data_path = Path('../data')
    qa_path = data_path / 'qa'

    with wandb.init(project=wandb_project,
                    notes="Uploads artifact with questions for the model",
                    config={'params': OmegaConf.to_object(params)}):
        qa_files = [qa_path / filename for filename in params.qa_filenames]
        qa_artifact = wandb.Artifact(name='questions',
                                     type='dataset',
                                     description='Questions to be submitted to the language model, optionally with \
                                     expected answers.'
                                     )
        for qa_file in qa_files:
            qa_artifact.add_file(qa_file)
        wandb.log_artifact(qa_artifact)


if __name__ == '__main__':
    main()
