import polars as pl
from src.utils import setup_logger

logger = setup_logger()

def main():
    logger.info("Starting Data Discovery")
    
    try:
        train = pl.read_csv('data/raw/train.csv', ignore_errors=True)
        test = pl.read_csv('data/raw/test.csv', ignore_errors=True)
        sub = pl.read_csv('data/raw/SampleSubmission.csv')
        
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")
        logger.info(f"SampleSubmission shape: {sub.shape}")
        
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
