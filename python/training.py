import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict


def plot_corr_matrix(df):
    fig, ax = plt.subplots(figsize=(15, 15))  # Sample figsize in inches
    sns.heatmap(df.corr(), annot=True, linewidths=.5, ax=ax)
    plt.show()


def print_metrics(confusion_matrix):
    print(confusion_matrix)
    tp = confusion_matrix[1][1]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tn = confusion_matrix[0][0]
    print("True positive: ", tp)
    print("False positive: ", fp)
    print("False negative: ", fn)
    print("True negative: ", tn)
    print("Precision: ", tp / (tp + fp))
    print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
    print("Recall: ", tp / (tp + fn))
    print("False negative rate: ", fn / (tp + fn))
    print("False positive rate: ", fp / (fp + tn))


def aggregation(application_pred, network_pred, or_flag=True, and_flag=True):
    if len(application_pred) != len(network_pred):
        raise Exception("application length != network length")

    if or_flag & and_flag:  # entrambi
        or_agg = np.array([])
        for i in range(len(application_pred)):
            or_agg = np.append(or_agg, application_pred[i] or network_pred[i])

        and_agg = np.array([])
        for i in range(len(application_pred)):
            and_agg = np.append(and_agg, application_pred[i] or network_pred[i])

        return or_agg, and_agg

    elif or_flag & (not and_flag):  # solo or aggregation
        or_agg = np.array([])
        for i in range(len(application_pred)):
            or_agg = np.append(or_agg, application_pred[i] or network_pred[i])

        return or_agg

    elif (not or_flag) & and_flag:  # solo and aggregation
        and_agg = np.array([])
        for i in range(len(application_pred)):
            and_agg = np.append(and_agg, application_pred[i] or network_pred[i])

        return and_agg


def train_model_data_aggregation(name_classifier, classifier, X, y):
    print("\n" + name_classifier + " data-aggregation" + "-" * 60)
    print_metrics(confusion_matrix(y, cross_val_predict(classifier, X, y, cv=10)))


def train_model_and_or_aggregation(name_classifier, classifier, application_df, network_df, y):
    print("\n" + name_classifier + "-" * 60)
    pred_or_agg_application = cross_val_predict(classifier, application_df, y, cv=10)
    pred_or_agg_network = cross_val_predict(classifier, network_df, y, cv=10)
    or_agg, and_agg = aggregation(pred_or_agg_application, pred_or_agg_network)
    print(name_classifier + "OR-aggregation" + "-" * 10)
    print_metrics(confusion_matrix(y, or_agg))
    print(name_classifier + "AND-aggregation" + "-" * 10)
    print_metrics(confusion_matrix(y, and_agg))
