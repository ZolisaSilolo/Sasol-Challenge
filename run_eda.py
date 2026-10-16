import polars as pl
from src.utils import setup_logger

logger = setup_logger()

def main():
    logger.info("Starting Data Discovery")
    
    from pathlib import Path
    train_path = Path('data/raw/train.csv')
    test_path = Path('data/raw/test.csv')
    sub_path = Path('data/raw/SampleSubmission.csv')

    if not train_path.exists():
        logger.warning(
            f"train.csv not found at '{train_path.resolve()}'. "
            "Note: train.csv is not included in git due to size constraints. "
            "Please place the competition train.csv in data/raw/ to inspect training distributions."
        )

    try:
        if train_path.exists():
            train = pl.read_csv(str(train_path), ignore_errors=True)
            logger.info(f"Train shape: {train.shape}")
        else:
            train = None

        if test_path.exists():
            test = pl.read_csv(str(test_path), ignore_errors=True)
            logger.info(f"Test shape: {test.shape}")
        if sub_path.exists():
            sub = pl.read_csv(str(sub_path))
            logger.info(f"SampleSubmission shape: {sub.shape}")
        if train is not None:
            logger.info("\n--- Target Distributions ---")
            targets = ['CLV_fuel', 'CLV_nonfuel', 'Opportunity']
            for col in targets:
                if col in train.columns:
                    logger.info(f"\nStats for {col}:")
                    if train[col].dtype == pl.String or train[col].dtype == pl.Categorical or train[col].n_unique() < 10:
                        print(train[col].value_counts())
                    else:
                        print(train[col].describe())
    except Exception as e:
        logger.error(f"Error during EDA: {e}")

if __name__ == '__main__':
    main()
