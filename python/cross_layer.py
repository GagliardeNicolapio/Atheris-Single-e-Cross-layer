from data_cleaning import cleaning_dataframe, data_balancing, df_to_arff, nan_values
from training import train_model_data_aggregation, train_model_and_or_aggregation
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn import svm
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def without_feature_selection():

    print("WITHOUT FEATURE SELECTION\n\n\n\n")

    df = pd.read_csv("../dataset/dataset.csv")

    df = cleaning_dataframe(df)

    df_to_arff(df, "datasetDataCleaningScaling")

    X, y = data_balancing(df)

    train_model_data_aggregation("J48", DecisionTreeClassifier(), X, y)

    train_model_data_aggregation("GAUSSIAN Naive Bayes", GaussianNB(), X, y)
    train_model_data_aggregation("MULTINOMIAL Naive Bayes", MultinomialNB(), X, y)
    train_model_data_aggregation("COMPLEMENT Naive Bayes", ComplementNB(), X, y)

    train_model_data_aggregation("Support Vector Machine", svm.SVC(kernel="poly"), X, y)
    train_model_data_aggregation("Logistic Regression", LogisticRegression(), X, y)

    # split dataset per OR, AND e XOR aggregation
    application_df = X[
        ['URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS', 'CHARSET', 'SERVER', 'CONTENT_LENGTH', 'WHOIS_COUNTRY',
         'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE', 'DNS_QUERY_TIMES']]
    network_df = X[
        ['TCP_CONVERSATION_EXCHANGE', 'DIST_REMOTE_TCP_PORT', 'REMOTE_IPS', 'APP_BYTES', 'SOURCE_APP_PACKETS',
         'REMOTE_APP_PACKETS', 'SOURCE_APP_BYTES', 'REMOTE_APP_BYTES', 'APP_PACKETS']]

    train_model_and_or_aggregation("J48", DecisionTreeClassifier(), application_df, network_df, y)

    train_model_and_or_aggregation("GAUSSIAN Naive Bayes", GaussianNB(), application_df, network_df, y)
    train_model_and_or_aggregation("MULTINOMIAL Naive Bayes", MultinomialNB(), application_df, network_df, y)
    train_model_and_or_aggregation("COMPLEMENT Naive Bayes", ComplementNB(), application_df, network_df, y)

    train_model_and_or_aggregation("Support Vector Machine", svm.SVC(kernel="poly"), application_df, network_df, y)
    train_model_and_or_aggregation("Logistic Regression", LogisticRegression(), application_df, network_df, y)


def subset_eval_selection():
    print("\n\n\n\n SUBSET EVAL SELECTION \n\n\n\n")

    df_subset_eval = pd.read_csv("../dataset/datasetSubsetEval.csv")

    X_eval, y_eval = data_balancing(df_subset_eval)

    train_model_data_aggregation("J48", DecisionTreeClassifier(), X_eval, y_eval)

    train_model_data_aggregation("GAUSSIAN Naive Bayes", GaussianNB(), X_eval, y_eval)
    train_model_data_aggregation("MULTINOMIAL Naive Bayes", MultinomialNB(), X_eval, y_eval)
    train_model_data_aggregation("COMPLEMENT Naive Bayes", ComplementNB(), X_eval, y_eval)

    train_model_data_aggregation("Support Vector Machine", svm.SVC(kernel="poly"), X_eval, y_eval)
    train_model_data_aggregation("Logistic Regression", LogisticRegression(), X_eval, y_eval)

    application_df = X_eval[
        ['CONTENT_LENGTH', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']]
    network_df = X_eval[['DIST_REMOTE_TCP_PORT', 'REMOTE_APP_PACKETS']]

    train_model_and_or_aggregation("J48", DecisionTreeClassifier(), application_df, network_df, y_eval)

    train_model_and_or_aggregation("GAUSSIAN Naive Bayes", GaussianNB(), application_df, network_df, y_eval)
    train_model_and_or_aggregation("MULTINOMIAL Naive Bayes", MultinomialNB(), application_df, network_df, y_eval)
    train_model_and_or_aggregation("COMPELEMNT Naive Bayes", ComplementNB(), application_df, network_df, y_eval)

    train_model_and_or_aggregation("Support Vector Machine", svm.SVC(kernel="poly"), application_df, network_df, y_eval)
    train_model_and_or_aggregation("Logistic Regression", LogisticRegression(), application_df, network_df, y_eval)


def info_gain_selection():
    print("\n\n\n\n INFO GAIN SELECTION \n\n\n\n")

    df_info_gain = pd.read_csv("../dataset/infoGainDataset.csv")

    X_info, y_info = data_balancing(df_info_gain)

    train_model_data_aggregation("J48", DecisionTreeClassifier(), X_info, y_info)

    train_model_data_aggregation("GAUSSIAN Naive Bayes", GaussianNB(), X_info, y_info)
    train_model_data_aggregation("MULTINOMIAL Naive Bayes", MultinomialNB(), X_info, y_info)
    train_model_data_aggregation("COMPLEMENT Naive Bayes", ComplementNB(), X_info, y_info)

    train_model_data_aggregation("Support Vector Machine", svm.SVC(kernel="poly"), X_info, y_info)
    train_model_data_aggregation("Logistic Regression", LogisticRegression(), X_info, y_info)

    application_df = X_info[
        ['CONTENT_LENGTH', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']]
    network_df = X_info[['SOURCE_APP_BYTES', 'REMOTE_APP_PACKETS']]

    train_model_and_or_aggregation("J48", DecisionTreeClassifier(), application_df, network_df, y_info)

    train_model_and_or_aggregation("GAUSSIAN Naive Bayes", GaussianNB(), application_df, network_df, y_info)
    train_model_and_or_aggregation("MULTINOMIAL Naive Bayes", MultinomialNB(), application_df, network_df, y_info)
    train_model_and_or_aggregation("COMPLEMENT Naive Bayes", ComplementNB(), application_df, network_df, y_info)

    train_model_and_or_aggregation("Support Vector Machine", svm.SVC(kernel="poly"), application_df, network_df, y_info)
    train_model_and_or_aggregation("Logistic Regression", LogisticRegression(), application_df, network_df, y_info)


def pca_selection():
    print("\n\n\n\n PCA \n\n\n\n")

    df_pca = pd.read_csv("../dataset/dataset.csv")
    df_pca = cleaning_dataframe(df_pca, scaling=False, knn_imputer=False)

    scaler = MinMaxScaler()
    df_pca = pd.DataFrame(scaler.fit_transform(df_pca), columns=df_pca.columns)

    imputer = KNNImputer(missing_values=np.nan)
    df_pca = pd.DataFrame(imputer.fit_transform(df_pca), columns=df_pca.columns)

    df_pca_application = df_pca[
        ['URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS', 'CHARSET', 'SERVER', 'CONTENT_LENGTH', 'WHOIS_COUNTRY',
         'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE', 'DNS_QUERY_TIMES']]
    df_pca_network = df_pca[
        ['TCP_CONVERSATION_EXCHANGE', 'DIST_REMOTE_TCP_PORT', 'REMOTE_IPS', 'APP_BYTES', 'SOURCE_APP_PACKETS',
         'REMOTE_APP_PACKETS', 'SOURCE_APP_BYTES', 'REMOTE_APP_BYTES', 'APP_PACKETS']]

    y = df_pca['Type'].astype('int')

    pca_application = PCA(n_components=7)
    df_pca_application = pd.DataFrame(pca_application.fit_transform(df_pca_application))

    pca_network = PCA(n_components=2)
    df_pca_network = pd.DataFrame(pca_network.fit_transform(df_pca_network))

    smote = SMOTE()
    X_pca, y_pca = smote.fit_resample(pd.concat([df_pca_application, df_pca_network], axis=1), y)

    train_model_data_aggregation("J48", DecisionTreeClassifier(), X_pca, y_pca)
    train_model_data_aggregation("GAUSSIAN Naive Bayes", GaussianNB(), X_pca, y_pca)
    train_model_data_aggregation("Support Vector Machine", svm.SVC(kernel="poly"), X_pca, y_pca)
    train_model_data_aggregation("Logistic Regression", LogisticRegression(), X_pca, y_pca)

    X_pca_application = X_pca.iloc[:, :7]
    X_pca_network = X_pca.iloc[:, -2:]

    train_model_and_or_aggregation("J48", DecisionTreeClassifier(), X_pca_application, X_pca_network, y_pca)
    train_model_and_or_aggregation("GAUSSIAN Naive Bayes", GaussianNB(), X_pca_application, X_pca_network, y_pca)
    train_model_and_or_aggregation("Support Vector Machine", svm.SVC(kernel="poly"), X_pca_application, X_pca_network, y_pca)
    train_model_and_or_aggregation("Logistic Regression", LogisticRegression(), X_pca_application, X_pca_network, y_pca)


if __name__ == "__main__":
    print("\n\n\n CROSS-LAYER \n\n\n")
    pca_selection()
    without_feature_selection()
    subset_eval_selection()
    info_gain_selection()


