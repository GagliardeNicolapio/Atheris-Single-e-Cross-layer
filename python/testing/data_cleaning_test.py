import math
import unittest
import pandas as pd
import numpy as np
from python.data_cleaning import replace_none_statepro, label_encoder


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

if __name__ == '__main__':
    unittest.main()
