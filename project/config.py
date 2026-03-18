"""
Это файл конфигурации для запуска экспериментов и гиперпараметров для обучения, также здесь указаны пути для скачки датасета.
"""

from pathlib import Path

# указала пути к основной папке и папке с результатами
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# скачивание датасета и указание его как переменной
DATASET = "m4_monthly"          # M4 Monthly скачиваю с forecastingdata.org

M4_TSF_URL = "https://zenodo.org/record/4656064/files/m4_monthly_dataset.zip"

TSF_FILE_NAME = "m4_monthly_dataset.tsf"

N_SERIES = 500                  # кол-во серий для сэмплирования (взяла побольше)
RANDOM_SEED = 42
MIN_SERIES_LEN = 36             # убираю серии короче минимума
SEASONALITY = 12                # беру стандартный период в один год
ACF_LAG = 12                    # лаг для проверки seasonal ACF peak
ACF_THRESHOLD = 0.3             # порог для проверки сезонности

# горизонты прогнозирования
HORIZONS = {
    "short": 3,
    'medium': 6,
    "long": 18,
}
DEFAULT_HORIZON = 18            # горизонт для таблицы со сравнениями (взяла стандартный)

# здесь указала варианты фич
FEATURE_VARIANTS = [
    "lags_only",
    "lags_seasonal",
    "lags_calendar",
    "lags_fourier",
    "lags_seasonal_calendar",
    "lags_fourier_calendar",
]

# установка лагов
N_LAGS = 13                     # прямые лаги 1..N_LAGS
SEASONAL_LAGS = [12, 24]        # сезонные лаги для месячных данных
N_FOURIER_TERMS = 2             # количество sin/cos пар Фурье на период

# проверка доступности видеокарты
import torch
HAS_GPU = torch.cuda.is_available()


# гиперпараметры для кэтбуста
CATBOOST_PARAMS = {
    "iterations": 600,
    "learning_rate": 0.03,
    "depth": 4,
    "l2_leaf_reg": 5.0,
    "loss_function": "RMSE",
    "eval_metric": "MAE",
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "allow_writing_files": False, # чтобы не создавались лишние файлы логов
}

# если видеокарта найдена, то добавляю специфичные параметры
if HAS_GPU:
    CATBOOST_PARAMS.update({
        "task_type": "GPU",
        "devices": "0",
        "bootstrap_type": "Bernoulli", # Bernoulli лучше всего подходит для GPU в CatBoost
        # на GPU лучше работает RMSE, но буду следить за MAE
        "loss_function": "RMSE", 
        "eval_metric": "MAE", 
        "metric_period": 10,  # считаем метрики каждые 10 шагов для ускорения
        "pinned_memory_size": "2GB" # ускоряет передачу данных в Colab
    })
    print("Есть видеокарта, CatBoost на GPU.")
else:
    print("Видеокарта не обнаружена, CatBoost на CPU")


# бины для силы сезонности
STRENGTH_BINS = {
    "weak":   (ACF_THRESHOLD, 0.5),
    "strong": (0.5, 1.0),
}
