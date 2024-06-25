import pandas as pd
import yfinance as yf

class StockDataDownloader:
    def __init__(self):
        pass

    def download_stock_data(self, ticker):
        start_date = pd.to_datetime("today") - pd.DateOffset(years=3)
        end_date = pd.to_datetime("today")

        # Загрузка данных
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        # Добавление столбца с датой и расчет средней цены
        stock_data['date'] = stock_data.index
        stock_data = stock_data.reset_index(drop=True)
        stock_data['price'] = (stock_data['High'] + stock_data['Low']) / 2

        # Удаление ненужных столбцов
        columns_drop = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        stock_data = stock_data.drop(columns=columns_drop)

        return stock_data

    def download_multiple_stocks_data(self, tickers):
        result_data = {}
        for ticker in tickers:
            stock_data = self.download_stock_data(ticker)
            result_data[ticker] = stock_data
        return result_data