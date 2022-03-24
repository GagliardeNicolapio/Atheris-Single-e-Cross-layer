import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression


# stampa nomaColonna->numValMancanti
def missing_values(df):
    print("Valori mancanti")
    for col in df.columns:
        print(f"{col}->{df[col].isin(['NA', 'None', 'none']).sum()}")
    print("\n\n")


# stampa nomeColonna -> num valori nan
def nan_values(df):
    for col in df.columns:
        print(f"{col}->{df[col].isna().sum()}")


# Sostituisce i country code con il nome esteso
def replace_states_cc(df):
    states_code_dict = {"qld": "queensland", "nsw": "new south wales", "ab": "alberta", "on": "ontario",
                        "al": "alabama",
                        "qc": "quebec", "ny": "new york", "ca": "california", "fl": "florida", "ma": "massachusetts",
                        "ct": "connecticut",
                        "mo": "missouri", "dc": "district of columbia", "wa": "washington", "az": "arizona",
                        "kg": "kavango",
                        "de": "delaware", "ga": "georgia", "mi": "michigan", "tx": "texas", "nj": "new jersey",
                        "il": "illinois",
                        "ut": "utah", "bc": "british columbia", "va": "virginia", "oh": "ohio", "pa": "pennsylvania",
                        "la": "louisiana",
                        "ks": "kansas", "co": "colorado", "wv": "west virginia", "nv": "nevada", "ok": "oklahoma",
                        "tn": "tamil nadu", "rm": "rome",
                        "vt": "vermont", "ak": "alaska", "vi": "victoria", "or": "oregon", "wi": "wisconsin",
                        "md": "maryland", "sk": "saskatchewan",
                        "zh": "zuid-holland", "nh": "new hampshire", "nc": "north carolina", "hr": "haryana",
                        "me": "maine", "mb": "manitoba"}
    df["WHOIS_STATEPRO"].replace(states_code_dict, inplace=True)


# stampa gli states più frequenti per ogni country
def print_groupby_sort(name_file, df):
    f = open(name_file, "w")
    result = df.groupby(["WHOIS_COUNTRY", "WHOIS_STATEPRO"]).size().reset_index(name="Time").sort_values(
        by=['WHOIS_COUNTRY'])
    f.write(result.to_string())
    f.close()


# stampa in file le columns_list di df
def print_columns(name_file, columns_list, df):
    f = open(name_file, "w")
    f.write(df[columns_list].to_string())
    f.close()


def plot_corr_matrix(df):
    # df['WHOIS_COUNTRY'] = df['WHOIS_COUNTRY'].astype('category').cat.codes
    # df['WHOIS_STATEPRO'] = df['WHOIS_STATEPRO'].astype('category').cat.codes
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

    if or_flag & and_flag: # entrambi
        aggregation_or = np.array([])
        for i in range(len(application_pred)):
            aggregation_or = np.append(aggregation_or, application_pred[i] or network_pred[i])

        aggregation_and = np.array([])
        for i in range(len(application_pred)):
            aggregation_and = np.append(aggregation_and, application_pred[i] or network_pred[i])

        return aggregation_or, aggregation_and

    elif or_flag & (not and_flag): # solo or aggregation
        aggregation_or = np.array([])
        for i in range(len(application_pred)):
            aggregation_or = np.append(aggregation_or, application_pred[i] or network_pred[i])

        return aggregation_or

    elif (not or_flag) &  and_flag: # solo and aggregation
        aggregation_and = np.array([])
        for i in range(len(application_pred)):
            aggregation_and = np.append(aggregation_and, application_pred[i] or network_pred[i])

        return aggregation_and


if __name__ == "__main__":
    df = pd.read_csv('./dataset/dataset.csv')

    # Elimino la colonna URL
    df.pop('URL')

    # lowercasing di WHOIS_STATEPRO e WHOIS_COUNTRY
    df["WHOIS_STATEPRO"] = df["WHOIS_STATEPRO"].astype('str').str.lower()
    df["WHOIS_COUNTRY"] = df["WHOIS_COUNTRY"].astype('str').str.lower()

    # sostituisco i none con 'us' (valore piu frequente) in WHOIS_COUNTRY
    df["WHOIS_COUNTRY"] = df["WHOIS_COUNTRY"].replace(['none'], df['WHOIS_COUNTRY'].value_counts().index.tolist()[0])

    # Sostituisce i country code con il nome esteso
    replace_states_cc(df)

    most_freq_states_none_replace = {"no": "rogaland", "lu": "luxembourg", "jp": "osaka", "il": "israel",
                                     "ph": "manila",
                                     "se": "indal", "ua": "ukraine",
                                     "hk": "hong kong", "ch": "zug", "cn": "zhejiang", "cz": "praha", "br": "brazil",
                                     "de": "berlin", "be": "antwerp",
                                     "kr": "korea", "gb": "london", "uk": "united kingdom", "fr": "paris",
                                     "au": "queensland", "cy": "cyprus", "us": "california"}

    # sostituisco i none in WHOIS_STATEPRO con i states più frequenti
    dict_index_state = {}
    for index, row in df.iterrows():
        if (row["WHOIS_COUNTRY"] != "none") & (row["WHOIS_STATEPRO"] == "none"):
            dict_index_state[index] = most_freq_states_none_replace.get(row["WHOIS_COUNTRY"])

    new_column = pd.Series(dict_index_state.values(), name='WHOIS_STATEPRO', index=dict_index_state.keys())
    df.update(new_column)

    missing_values(df)

    # Sostituisco i none con NaN nelle colonne cha hanno ancora campi vuoti
    df["WHOIS_REGDATE"].replace('None', np.nan, inplace=True)
    df["WHOIS_UPDATED_DATE"].replace('None', np.nan, inplace=True)
    df["CHARSET"].replace('None', np.nan, inplace=True)
    df["SERVER"].replace('None', np.nan, inplace=True)

    # lowercasing di CHARSET e SERVER
    df["CHARSET"] = df["CHARSET"].astype('str').str.lower()
    df["SERVER"] = df["SERVER"].astype('str').str.lower()

    # Applico il LabelEncoder solo alle colonne string conservando i NaN
    list_col_str = []
    for col in df.select_dtypes(include='object').columns:
        list_col_str.append(col)
    df[list_col_str] = df[list_col_str].apply(lambda series: pd.Series(
        LabelEncoder().fit_transform(series[series.notnull()]),
        index=series[series.notnull()].index
    ))

    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    imputer = KNNImputer(missing_values=np.nan)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    nan_values(df)

    sns.countplot(x="Type", data=df)
    plt.show()
    smote = SMOTE()
    X, y = smote.fit_resample(df[df.columns.values.tolist()[:-1]], df['Type'])
    # df = pd.DataFrame(X, columns=df.columns.values.tolist()[:-1])

    # sns.countplot(x="Type", data=df)
    # plt.show()

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
