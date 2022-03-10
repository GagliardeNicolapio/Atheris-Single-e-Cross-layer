import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./dataset/dataset.csv')

#DATA CLEANING
#Missing values
print("Valori mancanti")
for col in df.columns:
   print(f"{col}->{df[col].isin(['NA', 'None']).sum()}")


#Replace country
statesCodeDict = {"QLD":"QUEENSLAND","NSW":"NEW SOUTH WALES","AB":"ALBERTA","ON":"ONTARIO","AL":"Alabama",
                   "QC":"QUEBEC","NY":"NEW YORK","CA":"CALIFORNIA","FL":"FLORIDA","MA":"Massachusetts","CT":"Connecticut",
                   "MO":"Missouri","DC":"District of Columbia","WA":"Washington","AZ":"Arizona","KG":"Kavango",
                   "DE":"Delaware","GA":"Georgia","MI":"Michigan","TX":"Texas","NJ":"New Jersey","IL":"Illinois",
                  "UT":"Utah","BC":"British Columbia","VA":"Virginia","OH":"Ohio","PA":"Pennsylvania","LA":"Louisiana",
                  "KS":"Kansas","CO":"Colorado","WV":"West Virginia","NV":"Nevada","OK":"Oklahoma","TN":"Tamil Nadu","RM":"Rome",
                  "VT":"Vermont","AK":"Alaska","VI":"Victoria","OR":"Oregon","WI":"Wisconsin","MD":"Maryland","SK":"Saskatchewan",
                  "ZH":"Zuid-Holland","NH":"New Hampshire","NC":"North Carolina","HR":"Haryana","ME":"Maine","MB":"Manitoba"}
df["WHOIS_STATEPRO"].replace(statesCodeDict,inplace=True)

for col in df["WHOIS_STATEPRO"]:
   if len(col) <= 3:
      print(col)