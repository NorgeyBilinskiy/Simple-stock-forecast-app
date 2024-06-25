import os
import pandas as pd

class ResultSaver:
    def __init__(self, save_folder_path='Simple_stock_forecast_app\\Program\\frontend\\data_predictions'):
        self.save_folder_path = save_folder_path

        # Проверяем, существует ли папка, если нет, создаем ее
        if not os.path.exists(self.save_folder_path):
            os.makedirs(self.save_folder_path)

    def save_results(self, tickers, ds_predict_3_day, ds_predict_7_day, ds_predict_14_day, ds_predict_31_day, today):
        for ticker in tickers:
            # Создаем переменные с именами, содержащими тикеры и временные интервалы
            globals()[f'{ticker}_3_day'] = pd.DataFrame(ds_predict_3_day[ticker]['predict'])
            globals()[f'{ticker}_7_day'] = pd.DataFrame(ds_predict_7_day[ticker]['predict'])
            globals()[f'{ticker}_14_day'] = pd.DataFrame(ds_predict_14_day[ticker]['predict'])
            globals()[f'{ticker}_31_day'] = pd.DataFrame(ds_predict_31_day[ticker]['predict'])
            
            # Создаем пути для сохранения файлов
            file_path_3_day = os.path.join(self.save_folder_path, f'{ticker}_3_day.csv')
            file_path_7_day = os.path.join(self.save_folder_path, f'{ticker}_7_day.csv')
            file_path_14_day = os.path.join(self.save_folder_path, f'{ticker}_14_day.csv')
            file_path_31_day = os.path.join(self.save_folder_path, f'{ticker}_31_day.csv')
            
            # Сохраняем датафреймы в CSV файлы
            globals()[f'{ticker}_3_day'].to_csv(file_path_3_day, index=False)
            globals()[f'{ticker}_7_day'].to_csv(file_path_7_day, index=False)
            globals()[f'{ticker}_14_day'].to_csv(file_path_14_day, index=False)
            globals()[f'{ticker}_31_day'].to_csv(file_path_31_day, index=False)

        today_df = pd.DataFrame({'date': [today]})
        file_path_today = os.path.join(self.save_folder_path, 'today.csv')
        today_df.to_csv(file_path_today, index=False)

