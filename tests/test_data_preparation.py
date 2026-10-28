"""Data-free, deterministic tests for the two input-preparation scripts."""
import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from data.prepare_load import prepare, compare, OUTPUT_COLUMNS
from data.prepare_price import prepare_price

EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "examples"

class DataPreparationTests(unittest.TestCase):
    def test_synthetic_load_and_device_rule(self):
        x=prepare(pd.read_csv(EXAMPLES/"synthetic_raw_15min.csv"))
        self.assertEqual(len(x),2)
        self.assertEqual(list(x.columns),list(OUTPUT_COLUMNS))
        r=x.iloc[0]
        self.assertEqual(r["time"],"27/07/2018 15:00")
        for col,val in {"air":0.225,"car":1.0,"clotheswasher":0.0,"dishwasher":1.5,"dry":0.0,"non-shiftable":0.2,"total":2.925}.items():
            self.assertAlmostEqual(r[col],val)
        self.assertEqual(x.iloc[1]["total"],0.0)

    def test_raw_schema_and_offset_required(self):
        a=pd.read_csv(EXAMPLES/"synthetic_raw_15min.csv")
        with self.assertRaisesRegex(ValueError,"missing required columns"):
            prepare(a.drop(columns=["air1"]))
        a["local_15min"]=a["local_15min"].str.replace("-05","",regex=False)
        with self.assertRaisesRegex(ValueError,"UTC offset"):prepare(a)

    def test_reference_mismatch_detected(self):
        x=prepare(pd.read_csv(EXAMPLES/"synthetic_raw_15min.csv"))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"hourly.csv"
            x.to_csv(p,index=False)
            self.assertEqual(compare(x,p)["mismatching_cells_above_1e_9"],0)
            altered=x.copy()
            altered.loc[0,"total"]+=1
            altered.to_csv(p,index=False)
            self.assertEqual(compare(x,p)["mismatching_cells_above_1e_9"],1)

    def test_price_coverage_and_negative_values(self):
        times=pd.date_range("2018-01-01",periods=8760,freq="h")
        y=pd.DataFrame({"timestamp":times.strftime("%d/%m/%Y %H:%M"),"Price":np.full(8760,27.55)})
        y.loc[12,"Price"]=-4
        self.assertEqual(prepare_price(y).loc[12,"Price"],-4)
        with self.assertRaisesRegex(ValueError,"coverage incomplete"):prepare_price(y.drop(index=100))

    def test_wrong_price_file_rejected(self):
        with self.assertRaisesRegex(ValueError,"Expected timestamp,Price"):
            prepare_price(pd.DataFrame({"datetime":[],"price":[]}))

if __name__=="__main__":unittest.main()
