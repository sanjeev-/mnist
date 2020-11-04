import pandas as pd
from sklearn import metrics
from sklearn import tree
import config


def run(fold):
    # Read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # Training data is where kfold is not equal to current fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # Validation data is where kfold is equal to fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Drop the label column and convert to numpy array using .values
    x_train = df_train.drop("labels", axis=1).values
    y_train = df_train.labels.values

    x_valid = df_valid.drop("labels", axis=1).values
    y_valid = df.valid.labels.values

    # Initialize simple decision tree classifier from SKLearn
    clf = tree.DecisionTreeClassifier()

    # Fit model on training data
    clf.fit(x_train, y_train)

    # Create predictions from validation samples
    preds = clf.predict(x_valid)

