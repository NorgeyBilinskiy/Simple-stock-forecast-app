# Импорт модулей
from stocks_data import StockDataDownloader
from lin_model import Predictor
from save_predict import ResultSaver

# Импорт переменных
from config import (
    tickers,
    target_column,
    window_sizes_3_day,
    test_size_3_day,
    window_sizes_7_day,
    test_size_7_day,
    window_sizes_14_day,
    test_size_14_day,
    window_sizes_31_day,
    test_size_31_day
)

# Получение котировок акций по тикерам
stock_downloader = StockDataDownloader()
stocks_data = stock_downloader.download_multiple_stocks_data(tickers)
for ticker, data_frame in stocks_data.items():
    globals()[f"{ticker}"] = data_frame

# Рассчеёт прогнозных значений
ds_predict_3_day, ds_predict_7_day, ds_predict_14_day, ds_predict_31_day, today = \
    Predictor.calculate_and_save_predictions(
        tickers, stocks_data, target_column,
        window_sizes_3_day, test_size_3_day,
        window_sizes_7_day, test_size_7_day,
        window_sizes_14_day, test_size_14_day,
        window_sizes_31_day, test_size_31_day
    )

# Сохранение прогнозных значений в CSV-файлы
result_saver = ResultSaver()
result_saver.save_results(tickers, ds_predict_3_day, ds_predict_7_day, ds_predict_14_day, ds_predict_31_day, today)