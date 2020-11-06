import pandas as pd
import joblib
from sklearn import metrics
from sklearn import tree
import config


def run(fold):
    # Read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE_KFOLD)

    # Training data is where kfold is not equal to current fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # Validation data is where kfold is equal to fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Drop the label column and convert to numpy array using .values
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    # Initialize simple decision tree classifier from SKLearn
    clf = tree.DecisionTreeClassifier()

    # Fit model on training data
    clf.fit(x_train, y_train)

    # Create predictions from validation samples
    preds = clf.predict(x_valid)

    # Calculate and print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"fold: {fold}, accuracy: {accuracy}")

    # Save model
    model_output_filepath = f"{config.MODEL_OUTPUT}dt_{fold}.bin"
    joblib.dump(clf, model_output_filepath)


if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)
