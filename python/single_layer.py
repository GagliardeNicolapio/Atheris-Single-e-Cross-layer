from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from data_cleaning import cleaning_dataframe
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from training import print_metrics


def pca_selection_single_layer():
    print("\n\n\n\n PCA \n\n\n\n")

    df_pca = pd.read_csv("../dataset/dataset.csv")
    df_pca = cleaning_dataframe(df_pca, scaling=False, knn_imputer=False)

    scaler = StandardScaler()
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

    pca_network = PCA(n_components=5)
    df_pca_network = pd.DataFrame(pca_network.fit_transform(df_pca_network))

    smote = SMOTE()
    X_pca_application, y_pca_application = smote.fit_resample(pd.concat(df_pca_application, axis=1), y)

    smote = SMOTE()
    X_pca_network, y_pca_network = smote.fit_resample(pd.concat(df_pca_network, axis=1), y)

    print("\n" + "NAIVE BAYES PCA SINGLE LAYER" + "-" * 60)
    #print_metrics(confusion_matrix(y, cross_val_predict(GaussianNB(), X, y, cv=10)))

if __name__ == "__main__":
    print("\n\n\n SINGLE LAYER \n\n\n")