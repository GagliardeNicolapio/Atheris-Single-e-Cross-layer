import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from data_cleaning import get_dataframe


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


if __name__ == "__main__":

    df, X, y = get_dataframe("./dataset/dataset.csv")

    # J48 data-aggregation
    print("j48 data-aggregation " + "-" * 60)
    j48_data_agg = DecisionTreeClassifier()
    j48_pred_data_agg = cross_val_predict(j48_data_agg, X, y, cv=10)
    print_metrics(confusion_matrix(y, j48_pred_data_agg))

    # Naive Bayes data-aggregation
    print("Naive Bayes data-aggregation " + "-" * 60)
    nby = GaussianNB()  # valori continui
    nby_pred = cross_val_predict(nby, X, y, cv=10)
    print_metrics(confusion_matrix(y, nby_pred))

    # SVM data-aggregation
    print("Support Vector Machine data-aggregation" + "-" * 60)
    svm_al = svm.SVC()
    svm_pred = cross_val_predict(svm_al, X, y, cv=10)
    print_metrics(confusion_matrix(y, svm_pred))

    # Logicistic data-aggregation
    print("Logicistic data-aggregation" + "-" * 60)
    lcs = LogisticRegression()
    lcs_pred = cross_val_predict(lcs, X, y, cv=10)
    print_metrics(confusion_matrix(y, lcs_pred))

    # split dataset per OR, AND e XOR aggregation
    application_df = X[
        ['URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS', 'CHARSET', 'SERVER', 'CONTENT_LENGTH', 'WHOIS_COUNTRY',
         'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE', 'DNS_QUERY_TIMES']]
    network_df = X[
        ['TCP_CONVERSATION_EXCHANGE', 'DIST_REMOTE_TCP_PORT', 'REMOTE_IPS', 'APP_BYTES', 'SOURCE_APP_PACKETS',
         'REMOTE_APP_PACKETS', 'SOURCE_APP_BYTES', 'REMOTE_APP_BYTES', 'APP_PACKETS']]

    # J48 OR e AND-aggregation
    print("J48 OR e AND-aggregation " + "-" * 60)
    j48_pred_or_agg_application = cross_val_predict(DecisionTreeClassifier(), application_df, y, cv=10)
    j48_pred_or_agg_network = cross_val_predict(DecisionTreeClassifier(), network_df, y, cv=10)
    aggregation_or, aggregation_and = aggregation(j48_pred_or_agg_application, j48_pred_or_agg_network)
    print("J48 OR-aggregation " + "-" * 10)
    print_metrics(confusion_matrix(y, aggregation_or))
    print("J48 AND-aggregation " + "-" * 10)
    print_metrics(confusion_matrix(y, aggregation_and))

    # Naive Bayes OR e AND-aggregation
    print("Naive Bayes OR e AND-aggregation " + "-" * 60)
    nby_pred_or_agg_application = cross_val_predict(GaussianNB(), application_df, y, cv=10)
    nby_pred_or_agg_network = cross_val_predict(GaussianNB(), network_df, y, cv=10)
    aggregation_or, aggregation_and = aggregation(nby_pred_or_agg_application, nby_pred_or_agg_network)
    print("Naive Bayes OR-aggregation " + "-" * 10)
    print_metrics(confusion_matrix(y, aggregation_or))
    print("Naive Bayes AND-aggregation " + "-" * 10)
    print_metrics(confusion_matrix(y, aggregation_and))

    # SVM OR e AND-aggregation
    print("Support vector machine OR e AND-aggregation " + "-" * 60)
    svm_pred_or_agg_application = cross_val_predict(svm.SVC(), application_df, y, cv=10)
    svm_pred_or_agg_network = cross_val_predict(svm.SVC(), network_df, y, cv=10)
    aggregation_or, aggregation_and = aggregation(svm_pred_or_agg_application, svm_pred_or_agg_network)
    print("Support vector machine OR-aggregation " + "-" * 10)
    print_metrics(confusion_matrix(y, aggregation_or))
    print("Support vector machine AND-aggregation " + "-" * 10)
    print_metrics(confusion_matrix(y, aggregation_and))

    # Logicistic OR e AND-aggregation
    print("Logicistic OR e AND-aggregation " + "-" * 60)
    lcs_pred_or_agg_application = cross_val_predict(LogisticRegression(), application_df, y, cv=10)
    lcs_pred_or_agg_network = cross_val_predict(LogisticRegression(), network_df, y, cv=10)
    aggregation_or, aggregation_and = aggregation(lcs_pred_or_agg_application, lcs_pred_or_agg_network)
    print("Logicistic OR-aggregation " + "-" * 10)
    print_metrics(confusion_matrix(y, aggregation_or))
    print("Logicistic AND-aggregation " + "-" * 10)
    print_metrics(confusion_matrix(y, aggregation_and))
