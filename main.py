import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# stampa nomaColonna->numValMancanti
def missing_values(df):
    print("Valori mancanti")
    for col in df.columns:
        print(f"{col}->{df[col].isin(['NA', 'None', 'none']).sum()}")
    print("\n\n")


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


if __name__ == "__main__":
    df = pd.read_csv('./dataset/dataset.csv')

    plot_corr_matrix(df)

    # DATA CLEANING
    # lowercasing di WHOIS_STATEPRO e WHOIS_COUNTRY
    df["WHOIS_STATEPRO"] = df["WHOIS_STATEPRO"].astype('str').str.lower()
    df["WHOIS_COUNTRY"] = df["WHOIS_COUNTRY"].astype('str').str.lower()
    # sostituisco i none con 'us' in WHOIS_COUNTRY
    df["WHOIS_COUNTRY"] = df["WHOIS_COUNTRY"].replace(['none'], df['WHOIS_COUNTRY'].value_counts().index.tolist()[0])

    #Sostituisce i country code con il nome esteso
    replace_states_cc(df)


    most_freq_states_none_replace = {"no": "rogaland", "lu": "luxembourg", "jp": "osaka", "il": "israel",
                                     "ph": "manila",
                                     "se": "indal", "ua": "ukraine",
                                     "hk": "hong kong", "ch": "zug", "cn": "zhejiang", "cz": "praha", "br": "brazil",
                                     "de": "berlin", "be": "antwerp",
                                     "kr": "korea", "gb": "london", "uk": "united kingdom", "fr": "paris",
                                     "au": "queensland", "cy": "cyprus", "us": "california"}


    #sostituisco i none in WHOIS_STATEPRO con i states più frequenti
    dict_index_state = {}
    for index, row in df.iterrows():
        if (row["WHOIS_COUNTRY"] != "none") & (row["WHOIS_STATEPRO"] == "none"):
            dict_index_state[index] = most_freq_states_none_replace.get(row["WHOIS_COUNTRY"])

    new_column = pd.Series(dict_index_state.values(), name='WHOIS_STATEPRO', index=dict_index_state.keys())
    df.update(new_column)

    print_groupby_sort("prova",df)
    missing_values(df)
