from finvizfinance.quote import finvizfinance
from finviz import get_stock
import pandas as pd
from pathlib import Path
from datetime import datetime
from os.path import exists
import time


class Recommendations1(object):
    def __init__(self, today_stamp):
        self.today_stamp = today_stamp
        self.saveMonth = str(today_stamp)[:7]
        self.saveScreeners = Path(f"data/screeners/{self.saveMonth}/{self.today_stamp}/")
        self.saveTickers = Path(f"data/tickers/{self.saveMonth}/{self.today_stamp}/")
        self.saveRec = Path(f"data/recommenders/{str(self.today_stamp)[:4]}/{self.saveMonth}/{self.today_stamp}/")
        self.saveRaw = Path(f"data/raw/{self.saveMonth}/{self.today_stamp}/")
        self.a = "__" * 25
        self.b = "- " * 5

    def run_rec1(self):

        if exists(self.saveRec / f"recommender_01_return_dataFrame.pkl"):
            return

        elif exists(self.saveRec / f"recommender_00_return_dataFrame.pkl"):
            self.ticker_list = sorted(
                list(
                    pd.read_pickle(
                        self.saveRec / f"recommender_00_return_dataFrame.pkl"
                    )["Symbol"]
                )
            )

        print(
            f"{self.a}\n{self.a} \n {self.b}Pre-Stage-Report {self.b} \n {self.a}\n{self.a}"
        )
        print(f"   > [STAGE 001] Analyst Recommendation (Averaged)")
        print(f"     * Total Assets In Evaluation: {len(self.ticker_list)}")
        print(f"     * Current Low-Watermark-Score: x < 2.51 \n\n")

        c0 = 0
        self.recommendations = []
        for s in self.ticker_list:
            try:
                recommendation = get_stock(s)["Recom"]
                if recommendation == "-":
                    recommendation = 6.0
            except Exception:
                recommendation = 6.0

            c0 += 1
            print(f"{c0}) {s} - analyst recom == [{recommendation}]")
            self.recommendations.append(round(float(recommendation), 2))
            time.sleep(0.13)

        dataframe = pd.DataFrame(
            list(zip(self.ticker_list, self.recommendations)),
            columns=["Company", "Recommendations"],
        ).sort_values("Recommendations")

        dataframe_02 = dataframe[dataframe["Recommendations"] < 2.6]
        dataframe_02.columns = ["Symbol", "Score"]
        dataframe_02["rank"] = range(1, len(dataframe_02["Symbol"]) + 1)
        dataframe_02 = dataframe_02.set_index("rank")
        dataframe_02.to_pickle(self.saveRec / "recommender_01_return_dataFrame.pkl")
        return

    def run_rec1_personal_port(self, personal_port):
        self.ticker_list = personal_port
        self.recommendations = []

        for s in self.ticker_list:
            try:
                recommendation = finvizfinance(s).TickerFundament()["Recom"]
            except Exception:
                recommendation = 6.0

            if recommendation == "-":
                recommendation = 6.0
            self.recommendations.append(round(float(recommendation), 2))

        dataframe = pd.DataFrame(
            list(zip(self.ticker_list, self.recommendations)),
            columns=["Company", "Recommendations"],
        ).sort_values("Recommendations")

        dataframe_02 = dataframe[dataframe["Recommendations"] < 2.6]
        dataframe_02.columns = ["Symbol", "Score"]
        return dataframe, dataframe_02


if __name__ == "__main__":
    Recommendations1(today_stamp="2021-11-11").run_rec1()