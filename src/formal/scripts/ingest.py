from formal.ingestion.config import IngestionConfig
from formal.utils import hydra

hydra.setup(stage_name="ingest", config_cls=IngestionConfig)


@hydra.main(stage_name="ingest")
def main(config: IngestionConfig) -> None:
    from formal.ingestion.ingest import ingest

    ingest(config)
