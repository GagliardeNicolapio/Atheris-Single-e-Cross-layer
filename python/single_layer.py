import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from data_cleaning import cleaning_dataframe, data_balancing
from training import train_model_single_layer


def subset_eval_single_layer():
    print("\n\n\n\n SUBSET EVAL \n\n\n\n")

    df_sub_eval_application = pd.read_csv("../dataset/subsetEvalDatasetSingleLayerApplication.csv")
    df_sub_eval_network = pd.read_csv("../dataset/subsetEvalDatasetSingleLayerNetwork.csv")

    X_subset_application, y_subset_application = data_balancing(df_sub_eval_application)
    X_subset_network, y_subset_network = data_balancing(df_sub_eval_network)

    train_model_single_layer("GAUSSIAN NB", GaussianNB(), X_subset_network, X_subset_application, y_subset_network,
                             y_subset_application)
    train_model_single_layer("MULTINOMIAL NB", MultinomialNB(), X_subset_network, X_subset_application,
                             y_subset_network, y_subset_application)
    train_model_single_layer("COMPLEMENT NB", ComplementNB(), X_subset_network, X_subset_application, y_subset_network,
                             y_subset_application)

    train_model_single_layer("J48", DecisionTreeClassifier(), X_subset_network, X_subset_application, y_subset_network,
                             y_subset_application)
    train_model_single_layer("SVM", svm.SVC(), X_subset_network, X_subset_application, y_subset_network,
                             y_subset_application)
    train_model_single_layer("LOGISTIC REGRESSION", LogisticRegression(), X_subset_network, X_subset_application,
                             y_subset_network, y_subset_application)


def info_gain_single_layer():
    print("\n\n\n\n INFO GAIN \n\n\n\n")

    df_info_gain_application = pd.read_csv("../dataset/infoGainDatasetSingleLayerApplication.csv")
    df_info_gain_network = pd.read_csv("../dataset/infoGainDatasetSingleLayerNetwork.csv")

    X_info_gain_application, y_info_gain_application = data_balancing(df_info_gain_application)
    X_info_gain_network, y_info_gain_network = data_balancing(df_info_gain_network)

    train_model_single_layer("GAUSSIAN NB", GaussianNB(), X_info_gain_network, X_info_gain_application,
                             y_info_gain_network,
                             y_info_gain_application)
    train_model_single_layer("MULTINOMIAL NB", MultinomialNB(), X_info_gain_network, X_info_gain_application,
                             y_info_gain_network, y_info_gain_application)
    train_model_single_layer("COMPLEMENT NB", ComplementNB(), X_info_gain_network, X_info_gain_application,
                             y_info_gain_network,
                             y_info_gain_application)

    train_model_single_layer("J48", DecisionTreeClassifier(), X_info_gain_network, X_info_gain_application,
                             y_info_gain_network,
                             y_info_gain_application)
    train_model_single_layer("SVM", svm.SVC(), X_info_gain_network, X_info_gain_application, y_info_gain_network,
                             y_info_gain_application)
    train_model_single_layer("LOGISTIC REGRESSION", LogisticRegression(), X_info_gain_network, X_info_gain_application,
                             y_info_gain_network, y_info_gain_application)


def pca_selection_single_layer():
    print("\n\n\n\n PCA \n\n\n\n")

    df_pca = pd.read_csv("../dataset/dataset.csv")
    df_pca = cleaning_dataframe(df_pca)

    df_pca_application = df_pca[
        ['URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS', 'CHARSET', 'SERVER', 'CONTENT_LENGTH', 'WHOIS_COUNTRY',
         'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']]
    df_pca_network = df_pca[
        ['TCP_CONVERSATION_EXCHANGE', 'DIST_REMOTE_TCP_PORT', 'REMOTE_IPS', 'APP_BYTES', 'SOURCE_APP_PACKETS',
         'REMOTE_APP_PACKETS', 'SOURCE_APP_BYTES', 'REMOTE_APP_BYTES', 'APP_PACKETS', 'DNS_QUERY_TIMES']]

    y = df_pca['Type'].astype('int')

    pca_application = PCA(n_components=7)
    df_pca_application = pd.DataFrame(pca_application.fit_transform(df_pca_application))

    pca_network = PCA(n_components=5)
    df_pca_network = pd.DataFrame(pca_network.fit_transform(df_pca_network))

    smote = SMOTE()
    X_pca_application, y_pca_application = smote.fit_resample(df_pca_application, y)

    smote = SMOTE()
    X_pca_network, y_pca_network = smote.fit_resample(df_pca_network, y)

    print("\n" + "NAIVE BAYES PCA SINGLE LAYER" + "-" * 60)
    train_model_single_layer("GAUSSIAN NB", GaussianNB(), X_pca_network, X_pca_application, y_pca_network,
                             y_pca_application)
    train_model_single_layer("J48", DecisionTreeClassifier(), X_pca_network, X_pca_application, y_pca_network,
                             y_pca_application)
    train_model_single_layer("SVM", svm.SVC(), X_pca_network, X_pca_application, y_pca_network, y_pca_application)
    train_model_single_layer("LOGISTIC REGRESSION", LogisticRegression(), X_pca_network, X_pca_application,
                             y_pca_network, y_pca_application)


def without_feature_selection_single_layer():
    print("\n\n\n\n WITHOUT FEATURE SELECTION \n\n\n\n")

    df = pd.read_csv("../dataset/dataset.csv")
    df = cleaning_dataframe(df)

    df_application = df[
        ['URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS', 'CHARSET', 'SERVER', 'CONTENT_LENGTH', 'WHOIS_COUNTRY',
         'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE', 'Type']]
    df_network = df[
        ['TCP_CONVERSATION_EXCHANGE', 'DIST_REMOTE_TCP_PORT', 'REMOTE_IPS', 'APP_BYTES', 'SOURCE_APP_PACKETS',
         'REMOTE_APP_PACKETS', 'SOURCE_APP_BYTES', 'REMOTE_APP_BYTES', 'APP_PACKETS', 'DNS_QUERY_TIMES', 'Type']]

    X_application, y_application = data_balancing(df_application)
    X_network, y_network = data_balancing(df_network)

    train_model_single_layer("GAUSSIAN NB", GaussianNB(), X_network, X_application,
                             y_network,
                             y_application)
    train_model_single_layer("MULTINOMIAL NB", MultinomialNB(), X_network, X_application,
                             y_network, y_application)
    train_model_single_layer("COMPLEMENT NB", ComplementNB(), X_network, X_application,
                             y_network,
                             y_application)

    train_model_single_layer("J48", DecisionTreeClassifier(), X_network, X_application,
                             y_network,
                             y_application)
    train_model_single_layer("SVM", svm.SVC(), X_network, X_application, y_network,
                             y_application)
    train_model_single_layer("LOGISTIC REGRESSION", LogisticRegression(), X_network,
                             X_application,
                             y_network, y_application)


if __name__ == "__main__":
    print("\n\n\n SINGLE LAYER \n\n\n")
    pca_selection_single_layer()
    subset_eval_single_layer()
    info_gain_single_layer()
    without_feature_selection_single_layer()
