import numpy as np

tickers = ["AAPL", "AMZN", "BABA", "GOOGL", "JNJ", "META", "MSFT", "NFLX", "NVDA", "TSLA"]

target_column = 'price'

window_sizes_3_day = np.arange(2, 15)
test_size_3_day = 3

window_sizes_7_day = np.arange(3, 18)
test_size_7_day = 7

window_sizes_14_day = np.arange(5, 25)
test_size_14_day  = 14

window_sizes_31_day  = np.arange(15, 31)
test_size_31_day = 31