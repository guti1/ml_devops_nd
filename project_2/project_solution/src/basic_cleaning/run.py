#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
from numpy import row_stack
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Apply some basic data cleaning to the raw dataset and export the result to a new artifact.
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Loading {artifact_local_path}")
    sample_dataframe = pd.read_csv(artifact_local_path).copy()

    logger.info(f"Cleaning {artifact_local_path}")

    # Drop price outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = sample_dataframe["price"].between(min_price, max_price)
    sample_dataframe_cleaned = sample_dataframe[idx].copy()
    logger.info(
        f"Dropped {(len(sample_dataframe_cleaned) - len(idx)) * -1} outliers, DataFrame shape: {sample_dataframe_cleaned.shape}"
    )

    # Fix date type
    sample_dataframe_cleaned["last_review"] = pd.to_datetime(
        sample_dataframe_cleaned["last_review"]
    )
    
    # Drop null values
    shape_with_nulls = sample_dataframe_cleaned.shape
    sample_dataframe_cleaned = sample_dataframe_cleaned.dropna()
    shape_non_null = sample_dataframe_cleaned.shape
    logger.info(f"Dropped {(shape_with_nulls[0] - shape_non_null[0])} null values, DataFrame shape: {sample_dataframe_cleaned.shape}")

    idx = sample_dataframe_cleaned['longitude'].between(-74.25, -73.50) & sample_dataframe_cleaned['latitude'].between(40.5, 41.2)
    sample_dataframe_cleaned = sample_dataframe_cleaned[idx].copy()
    # save the cleaned dataframe
    logger.info(f"Saving {args.output_artifact}")
    sample_dataframe_cleaned.to_csv(args.output_artifact, index=False)

    # upload it to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file(args.output_artifact)

    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", type=str, help="Name of the input artifact", required=True
    )

    parser.add_argument(
        "--output_artifact", type=str, help="Name of the output artifact", required=True
    )

    parser.add_argument(
        "--output_type", type=str, help="Type of the output artifact", required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True,
    )

    parser.add_argument("--min_price", type=int, help="Minimum price", required=True)

    parser.add_argument("--max_price", type=int, help="Maximum price", required=True)

    args = parser.parse_args()

    go(args)
