import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
import arff, os


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


# stampa il dataframe features+target nel file file_name
def print_dataframe_to_csv(file_name, features, target, columns):
    file = open(file_name, "w")

    for column in columns[:-1]:
        file.write(column + ",")
    file.write("Type\n")

    for i in range(len(features.index)):
        for column in columns[:-1]:
            file.write(str(features._get_value(features.index[i], column)) + ",")
        file.write(str(target._get_value(target.index[i])) + "\n")
    file.close()


def df_to_csv(file_name, df):
    f = open(file_name, "w+")
    f.write(df.to_csv(index=False))
    f.flush()
    f.close()


def df_to_arff(df, file_name):
    arff.dump('../dataset/' + file_name + '.arff',
              df.values, relation=file_name, names=df.columns)

    fin = open('../dataset/' + file_name + ".arff", "rt")
    fout = open('../dataset/' + file_name + "Final.arff", "wt")
    for line in fin:
        fout.write(line.replace('Type real', 'Type {0.0,1.0}'))
    fin.close()
    fout.close()
    os.remove('../dataset/' + file_name + ".arff")


# sostituisco i none in WHOIS_STATEPRO con i states più frequenti
def replace_none_statepro(df, most_freq_states_none_replace):
    dict_index_state = {}
    for index, row in df.iterrows():
        if (row["WHOIS_COUNTRY"] != "none") & (row["WHOIS_STATEPRO"] == "none"):
            dict_index_state[index] = most_freq_states_none_replace.get(row["WHOIS_COUNTRY"])

    new_column = pd.Series(dict_index_state.values(), name='WHOIS_STATEPRO', index=dict_index_state.keys())
    df.update(new_column)


# ritorna il dataframe dopo aver applicato data imputation e feature scaling
def get_dataframe(path):
    df = pd.read_csv(path)

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
    replace_none_statepro(df, most_freq_states_none_replace)

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

    return df


# SMOTE data balancing
def data_balancing(df):
    smote = SMOTE()
    X, y = smote.fit_resample(df[df.columns.values.tolist()[:-1]], df['Type'])
    return X, y
