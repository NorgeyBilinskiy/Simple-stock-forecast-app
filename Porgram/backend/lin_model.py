import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

class Predictor:
    # Функция для прогноза на test_size дней вперёд с помощью SMA и линейной регрессии
    # Создаём фнкцию для генерации новых фичей (уходящих в будущее на test_size суток) на основе SMA
    def predict_sma(df_original, target_column=['price'], window_sizes=np.arange(2, 31), test_size=31):
     # функция принимает на вход:
     # df_original - дата-фрейм содержащий информацию о измнении цены во времени,
     # target_column - поле цены акции (по умолчанию 'price'),
     # window_sizes=np.arange(2, 51) - размер временных окон (по умолчанию от 2 до 50),
     # test_size=31 - размер периода на который делаем прогноз (по умолчанию на 31 день)

        # Блокаем какие-то бесполезные pandas-сообщения, пофиксить которые пока я не смог
        pd.options.mode.chained_assignment = None

        # Проверяем, достаточно ли строк в датафрейме
        if len(df_original) <= test_size:
            raise ValueError("Длина датафрейма меньше или равна размеру тестовой выборки.")

        # Создаём новый дата-фрейм только с нужными полями: датой и таргетом
        df = pd.DataFrame()
        df['date'] = df_original['date'].copy()
        df['target'] = df_original[target_column].copy()

        # Разделяем полученный дата-фрейм на тренировочную и тестовую выборки
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]

        # Запускаем цикл перебора по размеру окна для SMA
        for window_size in window_sizes:
            # Иницилизируем промежуточный дата-фрейм в каждой итерации
            intermediate_df = pd.DataFrame()
            # Добавляем дату и таргет из начально дата-фрейма в промежуточный
            intermediate_df['date'] = train_df['date'].copy()
            intermediate_df['target'] = train_df['target'].copy()

            # Запускаем цикл для рекурсвиного предсказания SMA на test_size дней
            for _ in range(test_size):
              # Считаем SMA в одном временном окне по тергету из промежутоного дата-фрейма
                intermediate_df['sma'] = intermediate_df['target'].rolling(window=window_size).mean()
                # Получаем последние дату и значение из sma из промежутоного дата-фрейма
                last_date = intermediate_df['date'].iloc[-1]
                last_sma_value = intermediate_df['sma'].iloc[-1]
                # Увеличиваем дату на один день вперёд
                new_date = last_date + pd.DateOffset(days=1)
                # Заполняем новую строку перед добавлением
                new_row = pd.Series({'date': new_date, 'target': last_sma_value, 'sma': np.nan})
                # Добавляем новую строку в промежуточный дата-фрейм
                intermediate_df = pd.concat([intermediate_df, pd.DataFrame([new_row], columns=intermediate_df.columns)], ignore_index=True)
                # Удаляем столбец sma, чтобы использовать его в следующей итерации
                intermediate_df.drop(columns=['sma'], inplace=True)

            # Оставляем в промежуточнм дата-фрейме test_size строк с конца
            intermediate_df = intermediate_df.tail(test_size)
            # Создаём переменную для корректного названия нового столбца
            sma_column_name = f'sma_{window_size}'
            # Добавляем в тестовый дата-фрейм столбец из промежуточного дата-фрейма
            test_df = pd.concat([test_df, intermediate_df['target'].copy().rename(sma_column_name)], axis=1)
            # Удаляем промежуточный дата-фрейм, чтобы использовать его в следующей итерации
            intermediate_df = None

        # Занимаемся поиском лучшей фичи для линейной модели на основе лучших скоров
        # Инициализируем словарь для названий столбцов и МАЕ в них
        errors = {}
        # Задаём список столбцов для цикла
        pred_columns = [f'sma_{i}' for i in window_sizes]

        # Рассчитаем MAE для каждого столбца-предикта
        for pred_column in pred_columns:
            error = test_df['target'] - test_df[pred_column]
            absolute_error = error.abs()
            MAE = absolute_error.mean()
            # Занесём значение в словарь
            errors[pred_column] = MAE

        # Ищем лучшую фичу с минимальным МАЕ при прогнозе на test_size дней
        best_feature_str = min(errors, key=errors.get)
        best_feature_MAE = errors[best_feature_str]
        # Удаляем префикс "sma_" и преобразуем в число
        best_feature = int(best_feature_str.replace("sma_", ""))

        # Сохраняем сообщение о лучшей фиче и её МАЕ
        text_sma = f"Лучшая по наименьшей ошибке SMA с размером окна {best_feature}" \
            f"\nMAE SMA для предиката на {test_size} суток: {best_feature_MAE:_.2f}"

        # Cоздаём дата-фрейм для теста лин.модели
        # К нему ниже добавим прогноз по SMA с лучшим окном на на test_size суток
        df_future_test = df.iloc[-366*2:-test_size]

        # Запускаем цикл для рекурсвиного предсказания SMA с лучшим времянным окном на test_size дней
        for _ in range(test_size):
            # Считаем SMA в одном временном окне по тергету
            df_future_test['sma'] = df_future_test['target'].rolling(window=best_feature).mean()
            # Получаем последние дату и значение из sma
            last_date = df_future_test['date'].iloc[-1]
            last_sma_value = df_future_test['sma'].iloc[-1]
            # Увеличиваем дату на один день вперёд
            new_date = last_date + pd.DateOffset(days=1)
            # Заполняем новую строку перед добавлением
            new_row = pd.Series({'date': new_date, 'target': last_sma_value, 'sma': np.nan})
            # Добавляем новую строку
            df_future_test = pd.concat([df_future_test, pd.DataFrame([new_row], columns=df_future_test.columns)], ignore_index=True)
            # Удаляем столбец sma, чтобы использовать его в следующей итерации
            df_future_test.drop(columns=['sma'], inplace=True)
        df_future_test['SMA'] = df_future_test['target'].rolling(window=best_feature).mean()

        # Разделяем полученный дата-фрейм на выборки для тренировки и теста
        model_train_df = df_future_test.iloc[best_feature:-test_size]
        model_test_df = df_future_test.iloc[len(model_train_df)+best_feature:]

        # Обучаем модель для получения скоров на тестовой выборки
        model_test = LinearRegression()
        x_train = model_train_df[['SMA']]
        y_train = model_train_df[['target']]
        model_test.fit(x_train, y_train)

        # Добавляем в тестовый дата-фрейм истинные значения таргета
        copy_df = df.iloc[-test_size:]

        # Получаем прогнозные значения для теста
        x_test = model_test_df[['SMA']]
        true_target = copy_df[['target']]
        predict_target = model_test.predict(x_test)

        # Считаем скоры
        mae_score = mean_absolute_error(true_target, predict_target)
        mse_score = mean_squared_error(true_target, predict_target)
        rmse_score = np.sqrt(mse_score)
        mape_score = mean_absolute_percentage_error(true_target, predict_target)
        r2_score_value = r2_score(true_target, predict_target)

        # Сохраняем сообщение о скорах обученной линейнокй  модели
        text_model = f"\n\nМетрики качества обученной модели:" \
            f"\nMean Absolute Error (MAE): {mae_score:_.2f}" \
            f"\nRoot Mean Squared Error (RMSE): {rmse_score:_.2f}" \
            f"\nMean Absolute Percentage Error (MAPE): {mape_score:_.2f}%" \
            f"\nR-squared (R2): {r2_score_value:_.2f} \n"

        # Собираем текстовое сообщение о скорах фичи и модели на тесте
        text = text_sma + text_model

        # Cоздаём дата-фрейм для прогноза лин.модели
        # К нему ниже добавим прогноз по SMA с лучшим окном на на test_size суток
        df_future = df.iloc[-366*2:]

        # Запускаем цикл для рекурсвиного предсказания SMA с лучшим времянным окном на test_size дней
        for _ in range(test_size):
            # Считаем SMA в одном временном окне по тергету
            df_future['sma'] = df_future['target'].rolling(window=best_feature).mean()
            # Получаем последние дату и значение из sma
            last_date = df_future['date'].iloc[-1]
            last_sma_value = df_future['sma'].iloc[-1]
            # Увеличиваем дату на один день вперёд
            new_date = last_date + pd.DateOffset(days=1)
            # Заполняем новую строку перед добавлением
            new_row = pd.Series({'date': new_date, 'target': last_sma_value, 'sma': np.nan})
            # Добавляем новую строку
            df_future = pd.concat([df_future, pd.DataFrame([new_row], columns=df_future.columns)], ignore_index=True)
            # Удаляем столбец sma, чтобы использовать его в следующей итерации
            df_future.drop(columns=['sma'], inplace=True)
        df_future['SMA'] = df_future['target'].rolling(window=best_feature).mean()

        # Разделяем полученный дата-фрейм на выборки для тренировки, теста, итоговой тренировки и предикта
        total_model_train_df = df_future.iloc[best_feature:-test_size]
        model_predict_df =  df_future[-test_size:]
        model_predict_df.drop(columns=['target'], inplace=True)

        # Обучаем модель на полных имеющихся данных
        model_pred = LinearRegression()
        x_train = total_model_train_df[['SMA']]
        y_train = total_model_train_df[['target']]
        model_pred.fit(x_train, y_train)

        # Получаем финальные прогнозные значения на будущее на test_size суток
        x_pred = model_predict_df[['SMA']]
        predict_target = model_pred.predict(x_pred)
        model_predict_df['predict'] = predict_target

        # Для красивой зарисовки графиков скопируем информацию о поведение таргета в последние test_size/2 суток
        recentvalues_df =  total_model_train_df[-test_size//2:]
        recentvalues_df['SMA'] = np.NAN
        recentvalues_df['predict'] = np.NAN
        # Получаем индекс последней строки
        last_index = recentvalues_df.index[-1]
        # Копируем значение в последней строке из поля target в SMA и predict
        target_value = recentvalues_df.at[last_index, 'target']
        last_date = recentvalues_df.at[last_index, 'date']
        recentvalues_df.at[last_index, 'SMA'] = target_value
        recentvalues_df.at[last_index, 'predict'] = target_value

        # Присоединим к дата-фрейму с предиктом на test_size суток данные о предыдущих test_size/2 суток
        result_df = pd.concat([recentvalues_df, model_predict_df], ignore_index=False)

        # Возвращаем итоговый дата-фрейм, сегодняшнее число, а также сообщение о лучшей фиче и скорах SMA и лин.модели
        return result_df, last_date, text


    # Функция для создания словаря, в котором содержится прогноз на все акции на заданное количество дней
    def predict_sma_tickers(tickers, stocks_data, target_column=['price'], window_sizes=np.arange(2, 31), test_size=31):
        result_dict = {}  # Словарь для сохранения результатов
        for ticker, data_frame in stocks_data.items():
            result_predict, today_result, text_result = Predictor.predict_sma(data_frame, target_column, window_sizes, test_size)
            # Сохранение результатов в словарь
            result_dict[ticker] = {
                "predict": result_predict,
            }
            # Вывод текстового сообщения
            print(f'{ticker}:\n')
            print(text_result,'\n')
            print(f'---'*25, '\n')
        return result_dict, today_result


    # Функция для создания словарей с прогнозами на 3, 7, 14 и 31 день
    def calculate_and_save_predictions(
        tickers, stocks_data, target_column,
        window_sizes_3_day, test_size_3_day,
        window_sizes_7_day, test_size_7_day,
        window_sizes_14_day, test_size_14_day,
        window_sizes_31_day, test_size_31_day
    ):
        ds_predict_3_day, today = Predictor.predict_sma_tickers(tickers, stocks_data, target_column, window_sizes_3_day, test_size_3_day)
        ds_predict_7_day, today = Predictor.predict_sma_tickers(tickers, stocks_data, target_column, window_sizes_7_day, test_size_7_day)
        ds_predict_14_day, today = Predictor.predict_sma_tickers(tickers, stocks_data, target_column, window_sizes_14_day, test_size_14_day)
        ds_predict_31_day, today = Predictor.predict_sma_tickers(tickers, stocks_data, target_column, window_sizes_31_day, test_size_31_day)
        return ds_predict_3_day, ds_predict_7_day, ds_predict_14_day, ds_predict_31_day, today