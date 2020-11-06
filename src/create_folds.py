import pandas as pd
from sklearn import model_selection
import config


def create_folds(num_folds):
    # Load in initial training data
    df = pd.read_csv(config.TRAINING_FILE)

    # Shuffle dataset rows
    df = df.sample(frac=1).reset_index(drop=True)

    # Create a new column for kfolds and fill it with -1
    df["kfold"] = -1

    # Fetch targets
    y = df.label.values

    # Initiate kfold class from model selection
    kf = model_selection.StratifiedKFold(n_splits=num_folds)

    # Fill in new kfold col
    for kfold, (train_indices, validation_indices) in enumerate(kf.split(X=df, y=y)):
        print(f"kfold: {kfold}")
        print(f"train_indices: {validation_indices}")
        df.loc[validation_indices, "kfold"] = kfold

    # Save training data input with folds
    df.to_csv(config.TRAINING_FILE_KFOLD, index=False)


if __name__ == "__main__":
    create_folds(num_folds=5)
