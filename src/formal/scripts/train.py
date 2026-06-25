from formal.training.config import TrainingConfig
from formal.utils import hydra

hydra.setup(stage_name="train", config_cls=TrainingConfig)


@hydra.main(stage_name="train")
def main(config: TrainingConfig) -> None:
    from formal.training.train import train

    train(config)
