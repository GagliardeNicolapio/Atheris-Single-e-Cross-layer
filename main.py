import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#stampa nomeColonna -> numValoriMancanti
def missingValues():
    print("Valori mancanti")
    for col in df.columns:
       print(f"{col}->{df[col].isin(['NA', 'None']).sum()}")


#Sostituisce i country code con il nome esteso, e applica la lower sulla colonna
def replaceStatesCC():
    statesCodeDict = {"QLD":"QUEENSLAND","NSW":"NEW SOUTH WALES","AB":"ALBERTA","ON":"ONTARIO","AL":"Alabama",
                       "QC":"QUEBEC","NY":"NEW YORK","CA":"CALIFORNIA","FL":"FLORIDA","MA":"Massachusetts","CT":"Connecticut",
                       "MO":"Missouri","DC":"District of Columbia","WA":"Washington","AZ":"Arizona","KG":"Kavango",
                       "DE":"Delaware","GA":"Georgia","MI":"Michigan","TX":"Texas","NJ":"New Jersey","IL":"Illinois",
                      "UT":"Utah","BC":"British Columbia","VA":"Virginia","OH":"Ohio","PA":"Pennsylvania","LA":"Louisiana",
                      "KS":"Kansas","CO":"Colorado","WV":"West Virginia","NV":"Nevada","OK":"Oklahoma","TN":"Tamil Nadu","RM":"Rome",
                      "VT":"Vermont","AK":"Alaska","VI":"Victoria","OR":"Oregon","WI":"Wisconsin","MD":"Maryland","SK":"Saskatchewan",
                      "ZH":"Zuid-Holland","NH":"New Hampshire","NC":"North Carolina","HR":"Haryana","ME":"Maine","MB":"Manitoba"}
    df["WHOIS_STATEPRO"].replace(statesCodeDict,inplace=True)
    df["WHOIS_STATEPRO"] = df["WHOIS_STATEPRO"].astype('str').str.lower()

def printGroupbySort(nameFile):
    f = open(nameFile, "w")
    result = df.groupby(["WHOIS_COUNTRY","WHOIS_STATEPRO"]).size().reset_index(name="Time").sort_values(by=['WHOIS_COUNTRY'])
    f.write(result.to_string())
    f.close()

if __name__ == "__main__":
    df = pd.read_csv('./dataset/dataset.csv')

    #DATA CLEANING
    replaceStatesCC()
    printGroupbySort("stateProResult")

    mostFreqStatesNoneReplace = {"NO":"rogaland","LU":"Luxembourg","JP":"osaka","IL":"israel","PH":"manila","SE":"indal","UA":"Ukraine",
                     "HK":"hong kong","CH":"zug","CN":"zhejiang","CZ":"praha","BR":"Brazil","DE":"berlin","BE":"antwerp",
                     "KR":"Korea","GB":"london","UK":"united kingdom","FR":"paris","AU":"queensland","CY":"Cyprus"}

