import os
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pyrootutils

from datasets.dataset_utils import Dataset
from models.model_utils import Model
from utils.utils import task_wrapper, extras

import torch

torch.use_deterministic_algorithms(True)

# project root setup
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@task_wrapper
def run(cfg: DictConfig) -> None:
    """Runs the main task.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.
    """
    print("====START====")
    dataset: Dataset = instantiate(cfg.dataset)
    dataset.load_data()
    print(f"Dataset name: {dataset.dataset_name}, nodes: {dataset.num_nodes}, edges: {dataset.adj.nonzero()}")
    model: Model = instantiate(cfg.model, encoder={"in_channels": dataset.features.shape[0]})
    logger = instantiate(cfg.logger)
    trainer = instantiate(
        cfg.trainer,
        dataset=dataset,
        model=model,
        logger=logger,
        training_objective=cfg.training_objective)
    trainer.train()
    logger.finalize()

@hydra.main(version_base=None, config_path='config', config_name='main')
def main(cfg: DictConfig) -> None:
    """Main entry point for running tasks.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    run(cfg)

if __name__ == '__main__':
    main()