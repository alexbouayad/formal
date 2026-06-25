from formal.tokenization.config import TokenizationConfig
from formal.utils import hydra

hydra.setup(stage_name="tokenize", config_cls=TokenizationConfig)


@hydra.main(stage_name="tokenize")
def main(config: TokenizationConfig) -> None:
    from formal.tokenization.tokenize import tokenize

    tokenize(config)
