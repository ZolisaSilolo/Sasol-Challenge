import argparse
from src.utils import seed_everything, setup_logger
from src.pipeline import run_pipeline

logger = setup_logger()

def main():
    logger.info('Starting Sasol Pipeline Execution...')
    seed_everything(42)
    run_pipeline()
    logger.info('Pipeline finished.')

if __name__ == '__main__':
    main()
