import math
import unittest
import pandas as pd
import numpy as np
from python.data_cleaning import replace_none_statepro, label_encoder, cleaning_dataframe


class TestDataCleaning(unittest.TestCase):
    def test_replace_none_statepro(self):
        df = pd.DataFrame({"WHOIS_COUNTRY": ["jp", "cz", "fr", "none"],
                           "WHOIS_STATEPRO": ["none", "none", "paris", "none"]})
        dict_most_freq = {"jp": "osaka", "cz": "praha"}
        replace_none_statepro(df, dict_most_freq)
        self.assertEqual(df["WHOIS_STATEPRO"].isin(["none"]).sum(), 1)
        self.assertEqual(df["WHOIS_STATEPRO"].iloc[0], "osaka")
        self.assertEqual(df["WHOIS_STATEPRO"].iloc[1], "praha")
        self.assertEqual(df["WHOIS_COUNTRY"].iloc[3], "none")
        self.assertEqual(df["WHOIS_STATEPRO"].iloc[3], "none")

    def test_label_encoder(self):
        df = pd.DataFrame({"col1": ["str1", np.nan, "str1", "str2"], "col2": [np.nan, np.nan, "str3", "str4"]})
        label_encoder(df)
        self.assertEqual(df["col2"].isna().sum(), 2)
        self.assertEqual(df["col1"].isna().sum(), 1)
        self.assertEqual(math.isnan(df["col1"].iloc[1]), True)
        self.assertEqual(math.isnan(df["col2"].iloc[0]), True)
        self.assertEqual(math.isnan(df["col2"].iloc[1]), True)
        self.assertEqual(df["col1"].iloc[0] != "str1", True)
        self.assertEqual(df["col1"].iloc[0] == df["col1"].iloc[2], True)
        self.assertEqual(df["col1"].iloc[0] != df["col1"].iloc[3], True)

    def test_cleaning_dataset(self):
        df = pd.DataFrame({
            "URL": ["url1", "url2", "url3", "url4", "url5"],
            "WHOIS_COUNTRY": ["US", "GB", "None", "US", "US"],
            "WHOIS_STATEPRO": ["CO", "None", "None", "CA", "Tennesse"],
            "WHOIS_REGDATE": ["dAte1", "none", "dATe2", "none", "DATE2"],
            "WHOIS_UPDATED_DATE": ["uP1", "up2", "Up1", "none", "UP2"],
            "CHARSET": ["none", "none", "Ch2", "cH1", "none"],
            "SERVER": ["sErVer1", "none", "SERVEr2", "none", "SERVER3"],

        })
        df_oracle = pd.DataFrame({
            "WHOIS_COUNTRY": ["us", "gb", "us", "us", "us"],
            "WHOIS_STATEPRO": ["colorado", "london", "california", "california", "tennesse"],
            "WHOIS_REGDATE": ["date1", np.nan, "date2", np.nan, "date2"],
            "WHOIS_UPDATED_DATE": ["up1", "up2", "up1", np.nan, "up2"],
            "CHARSET": [np.nan, np.nan, "ch2", "ch1", np.nan],
            "SERVER": ["server1", np.nan, "server2", np.nan, "server3"]
        })

        df_cleaned = cleaning_dataframe(df, scaling=False, knn_imputer=False, encoder=False)

        print(df_cleaned.to_string())
        print(df_oracle.to_string())
        self.assertEqual(df_cleaned.to_string() == df_oracle.to_string(), True)


if __name__ == '__main__':
    unittest.main()
