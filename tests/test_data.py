"""
test_data.py

Unit tests specifically for data loading and preprocessing.
"""

import unittest
import os
import tempfile
import pandas as pd
import numpy as np

from hsi_don_pipeline.src.data.load_data import load_data
from hsi_don_pipeline.src.data.preprocess import preprocess_data

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_csv = os.path.join(self.temp_dir.name, "test_data.csv")

        # Create mock data
        rows = 10
        cols = 5
        df = pd.DataFrame(np.random.rand(rows, cols),
                          columns=[f"band_{i}" for i in range(cols)])
        df["vomitoxin_ppb"] = np.random.randint(0, 2000, size=rows)
        df["hsi_id"] = np.arange(rows)

        df.to_csv(self.test_csv, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_data(self):
        df = load_data(self.test_csv)
        self.assertEqual(df.shape[0], 10)  # 10 rows
        self.assertIn("vomitoxin_ppb", df.columns)

    def test_preprocess_data(self):
        df = load_data(self.test_csv)
        X_train, X_test, y_train, y_test = preprocess_data(df, nir_band="band_1", red_band="band_2")
        self.assertTrue(X_train.shape[0] > 0)
        self.assertTrue(X_test.shape[0] > 0)

if __name__ == '__main__':
    unittest.main()
