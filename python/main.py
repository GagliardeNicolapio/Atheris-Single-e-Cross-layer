from data_cleaning import get_dataframe, data_balancing, df_to_arff
from training import train_model_data_aggregation, train_model_and_or_aggregation
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import pandas as pd


def without_feature_selection():
    print("WITHOUT FEATURE SELECTION\n\n\n\n")

    df = get_dataframe("../dataset/dataset.csv")

    df_to_arff(df, "datasetDataCleaningScaling")

    X, y = data_balancing(df)

    train_model_data_aggregation("J48", DecisionTreeClassifier(), X, y)
    train_model_data_aggregation("Naive Bayes", GaussianNB(), X, y)
    train_model_data_aggregation("Support Vector Machine", svm.SVC(), X, y)
    train_model_data_aggregation("Logistic Regression", LogisticRegression(), X, y)

    # split dataset per OR, AND e XOR aggregation
    application_df = X[
        ['URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS', 'CHARSET', 'SERVER', 'CONTENT_LENGTH', 'WHOIS_COUNTRY',
         'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE', 'DNS_QUERY_TIMES']]
    network_df = X[
        ['TCP_CONVERSATION_EXCHANGE', 'DIST_REMOTE_TCP_PORT', 'REMOTE_IPS', 'APP_BYTES', 'SOURCE_APP_PACKETS',
         'REMOTE_APP_PACKETS', 'SOURCE_APP_BYTES', 'REMOTE_APP_BYTES', 'APP_PACKETS']]

    train_model_and_or_aggregation("J48", DecisionTreeClassifier(), application_df, network_df, y)
    train_model_and_or_aggregation("Naive Bayes", GaussianNB(), application_df, network_df, y)
    train_model_and_or_aggregation("Support Vector Machine", svm.SVC(), application_df, network_df, y)
    train_model_and_or_aggregation("Logistic Regression", LogisticRegression(), application_df, network_df, y)


def subset_eval_selection():
    print("\n\n\n\n SUBSET EVAL SELECTION \n\n\n\n")

    df_subset_eval = pd.read_csv("../dataset/datasetSubsetEval.csv")

    X_eval, y_eval = data_balancing(df_subset_eval)

    train_model_data_aggregation("J48", DecisionTreeClassifier(), X_eval, y_eval)
    train_model_data_aggregation("Naive Bayes", GaussianNB(), X_eval, y_eval)
    train_model_data_aggregation("Support Vector Machine", svm.SVC(), X_eval, y_eval)
    train_model_data_aggregation("Logistic Regression", LogisticRegression(), X_eval, y_eval)

    application_df = X_eval[
        ['CONTENT_LENGTH', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']]
    network_df = X_eval[['DIST_REMOTE_TCP_PORT', 'REMOTE_APP_PACKETS']]

    train_model_and_or_aggregation("J48", DecisionTreeClassifier(), application_df, network_df, y_eval)
    train_model_and_or_aggregation("Naive Bayes", GaussianNB(), application_df, network_df, y_eval)
    train_model_and_or_aggregation("Support Vector Machine", svm.SVC(), application_df, network_df, y_eval)
    train_model_and_or_aggregation("Logistic Regression", LogisticRegression(), application_df, network_df, y_eval)


def info_gain_selection():
    print("\n\n\n\n INFO GAIN SELECTION \n\n\n\n")

    df_info_gain = pd.read_csv("../dataset/infoGainDataset.csv")

    X_info, y_info = data_balancing(df_info_gain)

    train_model_data_aggregation("J48", DecisionTreeClassifier(), X_info, y_info)
    train_model_data_aggregation("Naive Bayes", GaussianNB(), X_info, y_info)
    train_model_data_aggregation("Support Vector Machine", svm.SVC(), X_info, y_info)
    train_model_data_aggregation("Logistic Regression", LogisticRegression(), X_info, y_info)

    application_df = X_info[
        ['CONTENT_LENGTH', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']]
    network_df = X_info[['SOURCE_APP_BYTES', 'REMOTE_APP_PACKETS']]

    train_model_and_or_aggregation("J48", DecisionTreeClassifier(), application_df, network_df, y_info)
    train_model_and_or_aggregation("Naive Bayes", GaussianNB(), application_df, network_df, y_info)
    train_model_and_or_aggregation("Support Vector Machine", svm.SVC(), application_df, network_df, y_info)
    train_model_and_or_aggregation("Logistic Regression", LogisticRegression(), application_df, network_df, y_info)


if __name__ == "__main__":
    without_feature_selection()
    subset_eval_selection()
    info_gain_selection()