import pandas as pd
from os.path import exists
from finvizfinance.quote import finvizfinance
from datetime import datetime
from pathlib import Path
import pickle5 as pickle

from src.tools import functions as f0


class Recommendations5(object):
    def __init__(self, today_stamp):
        self.today_stamp = today_stamp
        self.saveMonth = str(today_stamp)[:7]
        self.saveDay = str(datetime.now())[8:10]

        self.saveRec = Path(f"data/recommenders/{str(self.today_stamp)[:4]}/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveRec.exists():
            self.saveRec.mkdir(parents=True)

    def run_rec5(self):

        if exists(self.saveRec / "recommender_05_return_dataFrame.pkl"):
            return

        else:
            with open(
                str(self.saveRec) + "/recommender_04_return_dataFrame.pkl", "rb"
            ) as fh:
                fd = pickle.load(fh)
            df = pd.DataFrame(fd).reset_index()
            df = df[df["my_score"] > 49.99]
            cde = pd.DataFrame(df).set_index(["rank", "my_score"])
            cde.to_pickle(self.saveRec / "recommender_05_return_dataFrame.pkl")
            return


if __name__ == "__main__":
    Recommendations5("2021-07-28").run_rec5()