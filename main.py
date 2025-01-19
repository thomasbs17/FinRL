# https://colab.research.google.com/github/AI4Finance-Foundation/FinRL-Tutorials/blob/master/1-Introduction/Stock_NeurIPS2018_SB3.ipynb#scrollTo=yCKm4om-s9kE

import itertools
from typing import Literal, get_args

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl import config, config_tickers
from finrl.config import INDICATORS
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from helpers import StockReturnsComputing
from finrl.agents.stablebaselines3.models import DRLAgent


SUPPORTED_MODELS = Literal["a2c", "ppo", "ddpg", "td3", "sac", "erl", "rllib"]


class FinRlTest:
    processed_full: pd.DataFrame
    mvo_df: pd.DataFrame  # mean-variance optimization dataframe
    train_df: pd.DataFrame
    trade_df: pd.DataFrame

    def __init__(
        self,
        train_start_date: str,
        train_end_date: str,
        trade_start_date: str,
        trade_end_date: str,
        initial_portfolio_value=1000000,
        tickers: list[str] = None,
        indicators: list[str] = None,
        use_vix: bool = False,
        use_turbulence: bool = False,
    ):
        check_and_make_directories(
            [
                config.DATA_SAVE_DIR,
                config.TRAINED_MODEL_DIR,
                config.TENSORBOARD_LOG_DIR,
                config.RESULTS_DIR,
            ]
        )

        if not tickers:
            self.tickers = config_tickers.DOW_30_TICKER
        if not indicators:
            self.indicators = INDICATORS
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.trade_start_date = trade_start_date
        self.trade_end_date = trade_end_date
        self.initial_portfolio_value = initial_portfolio_value
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.pre_process()

    def get_data(self, use_saved: bool) -> pd.DataFrame:
        if use_saved:
            return pd.read_csv("datasets/dow_30_2021.csv")
        df = YahooDownloader(
            start_date=self.trade_start_date,
            end_date=self.trade_end_date,
            ticker_list=self.tickers,
        ).fetch_data()
        df.to_csv("datasets/dow_30_2021.csv")
        return df

    def pre_process(self):
        df = self.get_data(use_saved=True)
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=self.indicators,
            use_vix=self.use_vix,
            use_turbulence=self.use_turbulence,
            user_defined_feature=False,
        )
        processed = fe.preprocess_data(df)

        list_ticker = processed["tic"].unique().tolist()
        list_date = list(
            pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
        )
        combination = list(itertools.product(list_date, list_ticker))

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
            processed, on=["date", "tic"], how="left"
        )
        processed_full = processed_full[processed_full["date"].isin(processed["date"])]
        processed_full = processed_full.sort_values(["date", "tic"])

        self.processed_full = processed_full.fillna(0)
        self.mvo_df = processed_full.sort_values(["date", "tic"], ignore_index=True)[
            ["date", "tic", "close"]
        ]
        self.train_df = data_split(
            processed_full, self.train_start_date, self.train_end_date
        )
        self.train_df = data_split(
            processed_full, self.trade_start_date, self.trade_end_date
        )

    def train_agent(self, model_name: SUPPORTED_MODELS):
        training_env = self.get_training_env()
        agent = DRLAgent(env=training_env)
        model_params = dict()
        if model_name == "a2c":
            model_params = config.A2C_PARAMS
        elif model_name == "ddpg":
            model_params = config.DDPG_PARAMS
        elif model_name == "ppo":
            model_params = config.PPO_PARAMS
        elif model_name == "td3":
            model_params = config.TD3_PARAMS
        elif model_name == "sac":
            model_params = config.SAC_PARAMS
        model = agent.get_model(model_name=model_name, model_kwargs=model_params)
        tmp_path = f"{config.RESULTS_DIR}/{model_name}"
        model_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model.set_logger(model_logger)
        return agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=50000
        )

    def set_turbulence_threshold(self):
        """
        Set the turbulence threshold to be greater than the maximum of insample turbulence data.
        If current turbulence index is greater than the threshold, then we assume that the current market is volatile
        """
        data_risk_indicator = self.processed_full[
            (self.processed_full.date < self.trade_end_date)
            & (self.processed_full.date >= self.trade_start_date)
        ]
        insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=["date"])

    @property
    def env_kwargs(self) -> dict:
        stock_dimension = len(self.train_df.tic.unique())
        state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
        buy_cost_list = sell_cost_list = [0.001] * stock_dimension
        num_stock_shares = [0] * stock_dimension
        return {
            "hmax": 100,
            "initial_amount": self.initial_portfolio_value,
            "num_stock_shares": num_stock_shares,
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
        }

    def get_training_env(self) -> DummyVecEnv:
        e_train_gym = StockTradingEnv(df=self.train_df, **self.env_kwargs)
        training_env, _ = e_train_gym.get_sb_env()
        return training_env

    def get_predictions(self, model_name: SUPPORTED_MODELS):
        """
        :return: account value and actions
        """
        trained_model = self.train_agent(model_name)
        trading_env = StockTradingEnv(
            df=self.trade_df,
            turbulence_threshold=70,
            risk_indicator_col="vix",
            **self.env_kwargs,
        )
        return DRLAgent.DRL_prediction(model=trained_model, environment=trading_env)

    def mean_variance_optimization(self) -> pd.DataFrame:
        fst = self.mvo_df.copy()
        fst = fst.iloc[0 * 29 : 0 * 29 + 29, :]
        tic = fst["tic"].tolist()

        mvo = pd.DataFrame()

        for k in range(len(tic)):
            mvo[tic[k]] = 0

        for i in range(self.mvo_df.shape[0] // 29):
            n = self.mvo_df.copy()
            n = n.iloc[i * 29 : i * 29 + 29, :]
            date = n["date"][i * 29]
            mvo.loc[date] = n["close"].tolist()

        # Obtain optimal portfolio sets that maximize return and minimize risk
        # extract asset prices
        stock_data = mvo.head(mvo.shape[0] - 336)
        trade_data = mvo.tail(336)
        trade_data.to_numpy()

        # compute asset returns
        ar_stock_prices = np.asarray(stock_data)
        [rows, cols] = ar_stock_prices.shape
        ar_returns = StockReturnsComputing(ar_stock_prices, rows, cols)

        # compute mean returns and variance covariance matrix of returns
        mean_returns = np.mean(ar_returns, axis=0)
        cov_returns = np.cov(ar_returns, rowvar=False)

        ef_mean = EfficientFrontier(mean_returns, cov_returns, weight_bounds=(0, 0.5))
        ef_mean.max_sharpe()
        cleaned_weights_mean = ef_mean.clean_weights()
        mvo_weights = np.array(
            [self.initial_portfolio_value * cleaned_weights_mean[i] for i in range(29)]
        )

        last_price = np.array([1 / p for p in stock_data.tail(1).to_numpy()[0]])
        initial_portfolio = np.multiply(mvo_weights, last_price)

        portfolio_assets = trade_data @ initial_portfolio
        return pd.DataFrame(portfolio_assets, columns=["Mean Var"])

    def backtest(self):
        all_models = get_args(SUPPORTED_MODELS)
        # all_results = self.mean_variance_optimization()
        all_results = pd.DataFrame()
        for model_name in all_models:
            df_account_value, _ = self.get_predictions(model_name)
            model_results = df_account_value.set_index(df_account_value.columns[0])
            model_results["model_name"] = model_name
            all_results.append([all_results, model_results])

        # df_dji_ = get_baseline(
        #     ticker="^DJI", start=self.trade_start_date, end=self.trade_end_date
        # )

        # baseline stats
        # print("==============Get Baseline Stats===========")
        # stats = backtest_stats(df_dji_, value_col_name="close")
        # df_dji = pd.DataFrame()
        # df_dji["date"] = df_account_value["date"]
        # df_dji["account_value"] = (
        #     df_dji_["close"] / df_dji_["close"][0] * self.env_kwargs["initial_amount"]
        # )
        # df_dji.to_csv("df_dji.csv")
        # df_dji = df_dji.set_index(df_dji.columns[0])
        # df_dji.to_csv("df_dji+.csv")
        # result = pd.merge(
        #     result, df_dji, left_index=True, right_index=True, suffixes=("", "_dji")
        # )
        all_results.to_csv("all_backtest_results.csv")


if __name__ == "__main__":
    finrl = FinRlTest(
        trade_start_date="2010-01-01",
        train_end_date="2021-10-01",
        train_start_date="2021-10-01",
        trade_end_date="2023-03-01",
    )
    finrl.backtest()
