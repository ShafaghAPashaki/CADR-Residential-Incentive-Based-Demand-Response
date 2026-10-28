"""Data-free tests for raw EnergyOnline preprocessing (no licensed records)."""
import unittest
import pandas as pd
from data.prepare_price import prepare_price

class PriceRawPreparationTests(unittest.TestCase):
    def setUp(self):
        times=pd.date_range("2018-01-01","2018-12-31 23:00",freq="h")
        gaps={pd.Timestamp("2018-03-11 02:00"),pd.Timestamp("2018-05-22 05:00"),pd.Timestamp("2018-06-04 05:00")}
        times=[t for t in times if t not in gaps]
        self.raw=pd.DataFrame({
            "Date":[t.strftime("%m/%d/%Y %I:%M:%S %p") for t in times],
            "Price":[float(10+(i%30)) for i in range(len(times))],
            "Zone":["LZ_AEN"]*len(times)
        })

    def test_historical_missing_hours_forward_filled(self):
        r=prepare_price(self.raw,zone="LZ_AEN").set_index("timestamp")["Price"]
        self.assertEqual(len(r),8760)
        for date,prior in [("11/03/2018 02:00","11/03/2018 01:00"),
                           ("22/05/2018 05:00","22/05/2018 04:00"),
                           ("04/06/2018 05:00","04/06/2018 04:00")]:
            self.assertEqual(r[date],r[prior])

    def test_requires_zone(self):
        with self.assertRaisesRegex(ValueError,"--zone"):prepare_price(self.raw)

    def test_rejects_incorrect_zone(self):
        with self.assertRaisesRegex(ValueError,"LZ_AEN"):prepare_price(self.raw,zone="LZ_SOUTH")

    def test_unexpected_gap_is_error(self):
        bad=self.raw[self.raw["Date"]!="01/01/2018 03:00:00 AM"]
        with self.assertRaisesRegex(ValueError,"Unexpected missing"):prepare_price(bad,zone="LZ_AEN")

    def test_boundary_hour_is_source_supplied(self):
        extra=pd.DataFrame({"Date":["01/01/2019 12:00:00 AM"],"Price":[14.89],"Zone":["LZ_AEN"]})
        r=prepare_price(self.raw,zone="LZ_AEN",boundary_df=extra)
        self.assertEqual(len(r),8761)
        self.assertEqual(r.iloc[-1]["timestamp"],"01/01/2019 00:00")
        self.assertEqual(r.iloc[-1]["Price"],14.89)

    def test_prepared_file_accepted(self):
        extra=pd.DataFrame({"Date":["01/01/2019 12:00:00 AM"],"Price":[14.89],"Zone":["LZ_AEN"]})
        prepared=prepare_price(self.raw,zone="LZ_AEN",boundary_df=extra)
        pd.testing.assert_frame_equal(prepared,prepare_price(prepared))

if __name__=="__main__":
    unittest.main()
