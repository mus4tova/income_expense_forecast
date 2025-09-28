import argparse
from loguru import logger

from src.train import ModelTrainer
from src.dataset import DataPreprocessor
from src.predict import ModelPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Train or predict with antifraud models."
    )
    parser.add_argument("--income", action="store_true")
    parser.add_argument("--outcome", action="store_true")
    parser.add_argument("--difference", action="store_true")
    args = parser.parse_args()

    if args.income:
        logger.info("=======Income Predictor Training=======")
        mode = "Inсome"
        in_ = DataPreprocessor(mode=mode).create_income_dataset()
        ModelTrainer(dataset=in_, mode=mode).model_trainer()
        ModelPredictor(dataset=in_, mode=mode).predictor()
    elif args.outcome:
        logger.info("=======Outcome Predictor Training=======")
        mode = "Outcome"
        out_ = DataPreprocessor(mode=mode).create_outcome_dataset()
        ModelTrainer(dataset=out_, mode=mode).model_trainer()
        ModelPredictor(dataset=out_, mode=mode).predictor()
    elif args.difference:
        logger.info("=======Difference Predictor Training=======")
        mode = "Difference"
        diff_ = DataPreprocessor(mode=mode).create_difference_dataset()
        ModelTrainer(dataset=diff_, mode=mode).model_trainer()
        ModelPredictor(dataset=diff_, mode=mode).predictor()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
