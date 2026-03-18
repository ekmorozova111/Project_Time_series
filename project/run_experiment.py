"""
Здесь полный пайплайн для запуска и проведения экспериментов с сезонностью

Содержание и функции:
  1. Загрузка и сэмплирование M4 Monthly датасета
  2. Фильтрую серии с видимой сезонностью (на основе ACF)
  3. Обучение и оценка бейзлайнов: Naive, SeasonalNaive, AutoTheta, AutoETS
  4. Обучение и оценка CatBoost с 6 вариантами фич
  5. Анализ результатов по силе сезонности и горизонту
  6. Сохранение всех результатов в папку results
"""

# сделаю импорт необходимых либ
import warnings
import sys
import json
import requests
import zipfile
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf
from catboost import CatBoostRegressor
from statsforecast import StatsForecast
from statsforecast.models import (
    Naive,
    SeasonalNaive,
    AutoTheta,
    AutoETS,
)
from tqdm import tqdm

# указываю пути к папкам и конфигам
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from config import (
    RESULTS_DIR, DATA_DIR,
    M4_TSF_URL, TSF_FILE_NAME,
    N_SERIES, RANDOM_SEED, MIN_SERIES_LEN,
    SEASONALITY, ACF_LAG, ACF_THRESHOLD,
    HORIZONS, DEFAULT_HORIZON,
    FEATURE_VARIANTS,
    N_LAGS, SEASONAL_LAGS, N_FOURIER_TERMS,
    CATBOOST_PARAMS, STRENGTH_BINS,
)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(RANDOM_SEED)

# загрузка и подготовка датасета
def convert_tsf_to_dataframe(full_file_path_and_name):
    """Парсер .tsf формата с защитой от ошибок кодировки"""
    all_data = {}
    
    # Использую utf-8 с игнорированием ошибок (у меня появлялась ошибка кодировки)
    with open(full_file_path_and_name, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("@"):
                # в формате TSF у нас такой вид данных -  series_id:start_timestamp:values
                # мне нужен первый элемент (ID) и последний (значения)
                parts = line.split(":")
                if len(parts) >= 2:
                    series_id = parts[0]
                    # данные в последней части строки, числа идут через запятую
                    try:
                        raw_values = parts[-1].split(",")
                        series_values = np.array([float(x) for x in raw_values if x.strip()])
                        all_data[series_id] = series_values
                    except ValueError:
                        # пропускаю строки, которые не являются данными
                        continue
    return all_data

def load_m4_monthly_tsf() -> dict:
    """Так как не получалось загрузить напрямую с сайта, то 
    сделала загрузку и парсинг файла .tsf из папки data (вручную его закинула в папку)"""
    tsf_path = DATA_DIR / TSF_FILE_NAME

    print(f"Файл найден. Начинаю парсинг {TSF_FILE_NAME}")
    
    # запускаю функцию парсинга датасета
    full_series = convert_tsf_to_dataframe(str(tsf_path))
    
    return full_series

def sample_and_filter(full_series: dict) -> dict:
    """Сэмплирую серии и убираю слишком короткие"""
    keys = list(full_series.keys())
    # фильтрую по длине сразу, чтобы сэмплировать из валидных рядов
    valid_keys = [
        k for k in keys 
        if len(full_series[k]) >= MIN_SERIES_LEN + DEFAULT_HORIZON
    ]
    
    chosen_keys = rng.choice(
        valid_keys, 
        size=min(N_SERIES, len(valid_keys)), 
        replace=False
    )
    return {k: full_series[k] for k in chosen_keys}


# функция для определения сезонности на основе ACF
def acf_seasonal_strength(series: np.ndarray, lag: int = ACF_LAG) -> float:
    """возвращаю |ACF(lag)| в качестве прокси для определения силы сезонности"""
    if len(series) < lag + 2:
        return 0.0
    acf_vals = acf(series, nlags=lag, fft=True)
    return abs(acf_vals[lag])


def annotate_series(series_dict: dict) -> pd.DataFrame:
    """делаю датафрейм с метадата с силой ACF на каждую серию"""
    records = []
    for sid, vals in series_dict.items():
        strength = acf_seasonal_strength(vals)
        records.append({"series_id": sid, "length": len(vals), "acf_strength": strength})
    return pd.DataFrame(records).set_index("series_id")


def filter_seasonal(series_dict: dict, meta: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """беру только серии где |ACF(12)| >= ACF_THRESHOLD"""
    keep = meta[meta["acf_strength"] >= ACF_THRESHOLD].index
    print(f"  Серии с видимой сезонностью: {len(keep)} / {len(meta)}")
    filtered = {sid: series_dict[sid] for sid in keep}
    return filtered, meta.loc[keep]


# Функции с расчетами метрик

def mase(actual: np.ndarray, forecast: np.ndarray, train: np.ndarray,
         seasonality: int = SEASONALITY) -> float:
    """Mean Absolute Scaled Error (scaled by seasonal naive in-sample MAE)."""
    naive_errors = np.abs(train[seasonality:] - train[:-seasonality])
    scale = np.mean(naive_errors)
    if scale == 0:
        return np.nan
    return np.mean(np.abs(actual - forecast)) / scale


def rmsse(actual: np.ndarray, forecast: np.ndarray, train: np.ndarray,
          seasonality: int = SEASONALITY) -> float:
    """Root Mean Squared Scaled Error."""
    naive_errors = (train[seasonality:] - train[:-seasonality]) ** 2
    scale = np.mean(naive_errors)
    if scale == 0:
        return np.nan
    return np.sqrt(np.mean((actual - forecast) ** 2) / scale)


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric MAPE (%)."""
    denom = (np.abs(actual) + np.abs(forecast)) / 2
    mask = denom > 0
    return np.mean(np.abs(actual[mask] - forecast[mask]) / denom[mask]) * 100


def compute_metrics(actual, forecast, train) -> dict:
    return {
        "MASE":  mase(actual, forecast, train),
        "RMSSE": rmsse(actual, forecast, train),
        "sMAPE": smape(actual, forecast),
    }


# Функции для инжиниринга фичей

def make_features(series: np.ndarray, horizon: int, variant: str) -> tuple:
    """
    Построение (X, y) для рекурсивного пошагового обучения  по выбранному варианту с фичами
    Функция возращает X (датафрейм), y (массив), и последние значения лагов для прогнозирования.
    """
    series_log = np.log1p(series)
    
    n = len(series_log)
    X_rows, y_vals = [], []
    
    # Определяем минимальный отступ, чтобы не было NaN
    needed_lags = [N_LAGS]
    if "seasonal" in variant:
        needed_lags.extend(SEASONAL_LAGS)
    max_lag = max(needed_lags)

    for t in range(max_lag, n - horizon + 1):
        row = {}
        # Целевое значение тоже в лог-шкале
        target = series_log[t + horizon - 1] 

        # Прямые лаги
        for lag in range(1, N_LAGS + 1):
            row[f"lag_{lag}"] = series_log[t - lag]

        if "seasonal" in variant:
            for slag in SEASONAL_LAGS:
                row[f"slag_{slag}"] = series_log[t - slag]

        if "calendar" in variant:
            month = (t % SEASONALITY)
            row["month"] = int(month)
            row["quarter"] = int(month // 3)

        if "fourier" in variant:
            for k in range(1, N_FOURIER_TERMS + 1):
                row[f"sin_{k}"] = np.sin(2 * np.pi * k * t / SEASONALITY)
                row[f"cos_{k}"] = np.cos(2 * np.pi * k * t / SEASONALITY)

        X_rows.append(row)
        y_vals.append(target)

    X = pd.DataFrame(X_rows) # fillna больше не нужен, так как мы не берем пустые окна
    y = np.array(y_vals)
    return X, y


def predict_recursive_global(model, series_train, horizon, variant, sid):
    """
    Функция генерации прогноза на h шагов с помощью direct стратегии (ставить одну модель на
    каждый шаг горизонта это слишком затратно по вычислениям — я использовала одну модель на весь горизонт).
    Для упрощения я взяла единый директ прогноз и выстроила прогноз на основе последнего известного 
    значения
    """
    series_extended = list(np.log1p(series_train)) # работаем в логах
    t = len(series_extended)

    for h in range(1, horizon + 1):
        row = {}
        # Собираем фичи из расширяющегося списка (с учетом предсказаний)
        for lag in range(1, N_LAGS + 1):
            row[f"lag_{lag}"] = series_extended[t - lag]

        if "seasonal" in variant:
            for slag in SEASONAL_LAGS:
                row[f"slag_{slag}"] = series_extended[t - slag]

        if "calendar" in variant:
            month = (t % SEASONALITY)
            row["month"] = month
            row["quarter"] = month // 3

        if "fourier" in variant:
            for k in range(1, N_FOURIER_TERMS + 1):
                row[f"sin_{k}"] = np.sin(2 * np.pi * k * t / SEASONALITY)
                row[f"cos_{k}"] = np.cos(2 * np.pi * k * t / SEASONALITY)

        # Добавляем идентификатор серии
        row['series_id_feat'] = sid

        feat_df = pd.DataFrame([row])
        
        # выравнивание столбцов в трейне (заполнение пропуском нулем)
        pred_log = model.predict(feat_df)[0]
        series_extended.append(pred_log)
        t += 1
        
    # возвращаем в исходную шкалу через exp
    preds_log = np.array(series_extended[-horizon:])
    return np.expm1(preds_log)


# Запуск бейзлайнов 

def run_baselines(series_dict: dict, horizon: int) -> pd.DataFrame:
    """Запуск бейзлайнов - Naive, SeasonalNaive, AutoTheta, AutoETS на всех рядах"""
    print(f"  Запуск бейзлайнов (h={horizon}) ")
    records = []

    # построение длинного датафрейма в формате для StatsForecast
    rows = []
    for sid, vals in series_dict.items():
        train_vals = vals[:-horizon]
        for i, v in enumerate(train_vals):
            rows.append({"unique_id": sid, "ds": i + 1, "y": v})
    df_long = pd.DataFrame(rows)

    sf = StatsForecast(
        models=[Naive(), SeasonalNaive(season_length=SEASONALITY),
                AutoTheta(season_length=SEASONALITY),
                AutoETS(season_length=SEASONALITY)],
        freq=1,
        n_jobs=-1,
    )
    sf.fit(df_long)
    forecasts = sf.predict(h=horizon)

    for sid, vals in series_dict.items():
        train = vals[:-horizon]
        actual = vals[-horizon:]
        f = forecasts[forecasts["unique_id"] == sid]
        for model_name in ["Naive", "SeasonalNaive", "AutoTheta", "AutoETS"]:
            col = model_name if model_name in f.columns else model_name.replace("Auto", "")
            if col not in f.columns:
                continue
            fc = f[col].values
            if len(fc) != len(actual):
                continue
            m = compute_metrics(actual, fc, train)
            m.update({"series_id": sid, "model": model_name, "horizon": horizon})
            records.append(m)

    return pd.DataFrame(records)


# Запуск экспериментов с Кэтбустом

def run_catboost_variant(series_dict: dict, horizon: int, variant: str) -> pd.DataFrame:
    """Обучение одной модели Кэтбуст на каждую серию и оценка"""
    records = []
    cat_features = ["month", "quarter"] if "calendar" in variant else []

    for sid, vals in tqdm(series_dict.items(), desc=f"  CatBoost/{variant}", leave=False):
        train = vals[:-horizon]
        actual = vals[-horizon:]

        if len(train) < N_LAGS + horizon + 5:
            continue

        X_tr, y_tr = make_features(train, horizon=1, variant=variant)

        # Кэтбуст с категориальными фичами
        params = {**CATBOOST_PARAMS}
        cat_idx = [i for i, c in enumerate(X_tr.columns) if c in cat_features]

        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, cat_features=cat_idx if cat_idx else None)

        fc = predict_recursive(model, train, horizon, variant)

        m = compute_metrics(actual, fc, train)
        m.update({"series_id": sid, "model": f"CatBoost_{variant}", "horizon": horizon})
        records.append(m)

    return pd.DataFrame(records)


def run_all_catboost(series_dict: dict, horizon: int) -> pd.DataFrame:
    """Глобальное обучение CatBoost на всех рядах сразу (а то ждала по 1 часу
    пока все обучится)"""
    all_results = []
    
    for variant in FEATURE_VARIANTS:
        print(f"Обучение глобальной модели (вариант: {variant})...")
        
        X_train_list, y_train_list = [], []
        test_data = {}

        # сбор данных со всех рядов
        for sid, vals in series_dict.items():
            train = vals[:-horizon]
            actual = vals[-horizon:]
            
            if len(train) < N_LAGS + horizon + 5:
                continue
                
            X_tr, y_tr = make_features(train, horizon=1, variant=variant)
            
            # добавляем ID серии как признак, чтобы модель их различала
            X_tr['series_id_feat'] = sid 
            
            X_train_list.append(X_tr)
            y_train_list.append(y_tr)
            test_data[sid] = (train, actual)

        # объединяем в один большой датафрейм
        X_global = pd.concat(X_train_list, ignore_index=True)
        y_global = np.concatenate(y_train_list)
        
        # однократное обучение на GPU
        cat_features = ['series_id_feat']
        if "calendar" in variant:
            cat_features.extend(["month", "quarter"])
            
        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(X_global, y_global, cat_features=cat_features)

        # рекурсивный прогноз для каждой серии
        for sid, (train, actual) in test_data.items():
            # predict_recursive должен уметь прокидывать series_id_feat
            fc = predict_recursive_global(model, train, horizon, variant, sid)
            
            m = compute_metrics(actual, fc, train)
            m.update({"series_id": sid, "model": f"CatBoost_{variant}", "horizon": horizon})
            all_results.append(m)

    return pd.DataFrame(all_results)


# Функции для визуализации (таблицы и графики)

def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    # если переданы сырые метрики — агрегируем
    if 'MASE' in results.columns:
        tbl = (
            results
            .groupby("model")[['MASE', 'RMSSE', 'sMAPE']]
            .agg(["mean", "median"])
            .round(4)
        )
        tbl.columns = ["_".join(c) for c in tbl.columns]
        return tbl.sort_values("MASE_mean")
    
    # если переданы уже агрегированные (из CSV) — просто сортируем
    return (
        results
        .set_index("model")
        [['MASE_mean', 'MASE_median', 'RMSSE_mean', 'RMSSE_median', 'sMAPE_mean', 'sMAPE_median']]
        .round(4)
        .sort_values("MASE_mean")
    )


def plot_metric_comparison(results: pd.DataFrame, metric: str = "MASE",
                           title: str = "", save_path: Path = None,
                           show: bool = False):
    order = results.groupby("model")[metric].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=results, x="model", y=metric, order=order, ax=ax,
                palette="coolwarm", fliersize=2)
    ax.set_title(title or f"{metric} by model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_ylabel(metric)
    plt.tight_layout()
    
    if save_path:
        # Убеждаемся, что папка существует и удаляем старый файл, если он мешает
        if save_path.exists():
            save_path.unlink() 
        fig.savefig(save_path, dpi=150)
        print(f"График сохранен в: {save_path}")
    
    if show:
        plt.show()
    
    #  закрываем фигуру и очищаем память
    plt.close(fig)
    plt.close('all')


def plot_acf_distribution(meta: pd.DataFrame, save_path: Path = None, show: bool = False):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(meta["acf_strength"], bins=20, edgecolor="white", color="steelblue")
    ax.axvline(ACF_THRESHOLD, color="red", linestyle="--", label=f"threshold={ACF_THRESHOLD}")
    ax.set_xlabel(f"|ACF(lag={ACF_LAG})|")
    ax.set_ylabel("Кол-во рядов")
    ax.set_title("Распределение силы сезонности (ACF)")
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        # убеждаемся, что папка существует и удаляем старый файл, если он мешает
        if save_path.exists():
            save_path.unlink() 
        fig.savefig(save_path, dpi=150)
        print(f"График сохранен в: {save_path}")
    
    if show:
        plt.show()
    
    #  закрываем фигуру и очищаем память
    plt.close(fig)
    plt.close('all')


def analyse_by_strength(results_cb: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Сравнение вариантов Кэтбуста по слабым и сильным сезонным сериям"""
    merged = results_cb.merge(meta[["acf_strength"]], left_on="series_id", right_index=True)
    records = []
    for label, (lo, hi) in STRENGTH_BINS.items():
        subset = merged[(merged["acf_strength"] >= lo) & (merged["acf_strength"] < hi)]
        grp = subset.groupby("model")[["MASE", "RMSSE", "sMAPE"]].mean().round(4)
        grp["strength_group"] = label
        records.append(grp)
    return pd.concat(records)


def analyse_by_horizon(series_dict: dict, meta: pd.DataFrame) -> pd.DataFrame:
    """Запуск Кэтбуста для каждого горизонта и сравнение результатов"""
    all_frames = []
    for h_name, h_val in HORIZONS.items():
        print(f"\n  Горизонт = {h_name} ({h_val} шагов)")
        fr = run_all_catboost(series_dict, h_val)
        fr["horizon_name"] = h_name
        all_frames.append(fr)
    return pd.concat(all_frames, ignore_index=True)


def correlation_strength_vs_improvement(results_cb: pd.DataFrame,
                                        meta: pd.DataFrame, save_path: Path = None, show: bool = False) -> None:
    """
    Функция расчитывает улучшение для каждой серии на основе лучшего сезонного варианта 
    через лаги. Затем коррелирует улучшение через силу ACF
    """
    pivot = results_cb.pivot_table(index="series_id", columns="model", values="MASE")
    if "CatBoost_lags_only" not in pivot.columns:
        return
    best_seasonal = pivot[[c for c in pivot.columns if c != "CatBoost_lags_only"]].min(axis=1)
    improvement = pivot["CatBoost_lags_only"] - best_seasonal  # если позитивное значение то сезонность лучше

    merged = improvement.rename("improvement").to_frame().join(meta["acf_strength"])
    merged = merged.dropna()
    r, p = pearsonr(merged["acf_strength"], merged["improvement"])
    print(f"\n  Pearson r (Сила ACF против улучшения MASE): {r:.3f}  (p={p:.4f})")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(merged["acf_strength"], merged["improvement"], alpha=0.5, s=20)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel(f"|ACF(lag={ACF_LAG})|  (seasonality strength)")
    ax.set_ylabel("MASE улучшение через lags_only")
    ax.set_title(f"Сильная сезонность приносит пользу?  r={r:.2f}")
    plt.tight_layout()
    
    fig.savefig(RESULTS_DIR / "strength_vs_improvement.png", dpi=150)

    # если путь не передан явно, использую стандартный из RESULTS_DIR
    if save_path is None:
        save_path = RESULTS_DIR / "strength_vs_improvement.png"

    if save_path.exists():
        save_path.unlink() 
    fig.savefig(save_path, dpi=150)
    print(f"График сохранен в: {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)

def plot_improvement_by_strength(results_cb: pd.DataFrame, meta: pd.DataFrame, save_path: Path = None):
    """
    Проверяю зависит ли выгода от сезонных фич от силы сезонности (ACF).
    Добавляю расчет корреляции Пирсона на график.
    """
    pivot = results_cb.pivot_table(index="series_id", columns="model", values="MASE")
    if "CatBoost_lags_only" not in pivot.columns:
        print("Пропуск: модель 'CatBoost_lags_only' не найдена в результатах.")
        return
        
    # считаю улучшение: (ошибка базовой модели) - (ошибка лучшей сезонной)
    seasonal_cols = [c for c in pivot.columns if c != "CatBoost_lags_only"]
    best_seasonal = pivot[seasonal_cols].min(axis=1)
    improvement = pivot["CatBoost_lags_only"] - best_seasonal 

    merged = improvement.rename("improvement").to_frame().join(meta["acf_strength"]).dropna()
    
    # расчет корреляции
    r_val, p_val = pearsonr(merged["acf_strength"], merged["improvement"])
    
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.regplot(data=merged, x='acf_strength', y='improvement', 
                scatter_kws={'alpha':0.4, 's':20}, line_kws={'color':'red'}, ax=ax)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_title(f"Связь силы сезонности и профита от фич\nPearson r = {r_val:.3f} (p={p_val:.4f})")
    ax.set_xlabel("Сила сезонности (ACF lag 12)")
    ax.set_ylabel("Улучшение MASE (выше 0 = сезонные фичи лучше)")
    
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_improvement_by_horizon(series_dict: dict, meta: pd.DataFrame, save_dir: Path):
    """
    Проверяю как меняется точность моделей при увеличении горизонта
    """
    horizon_results = []
    # использую горизонты из конфига ({'Short': 3, 'Medium': 6, 'Long': 18})
    for h_name, h_val in HORIZONS.items():
        print(f"  Эксперимент для горизонта {h_name} (h={h_val})...")
        res = run_all_catboost(series_dict, h_val)
        res['horizon_val'] = h_val
        horizon_results.append(res)
    
    df_horizons = pd.concat(horizon_results)
    summary = df_horizons.groupby(['horizon_val', 'model'])['MASE'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=summary, x='horizon_val', y='MASE', hue='model', marker='o', linewidth=2.5, ax=ax)
    
    ax.set_title("Устойчивость моделей к увеличению горизонта прогноза")
    ax.set_xlabel("Горизонт (количество месяцев)")
    ax.set_ylabel("Mean MASE")
    ax.grid(True, alpha=0.3)
    
    fig.savefig(save_dir / "horizon_analysis_lineplot.png", dpi=150)
    plt.close(fig)

def main():
    print(f"Запуск эксперимента (Seed: {RANDOM_SEED}) ")
    
    # Очистка старых результатов (гарантирует обновление файлов)
    for f in RESULTS_DIR.glob("*.png"):
        f.unlink()
    
    # загрузка и подготовка
    full_series = load_m4_monthly_tsf()
    sampled_series = sample_and_filter(full_series)
    meta = annotate_series(sampled_series)
    
    # визуализация распределения сезонности
    plot_acf_distribution(meta, save_path=RESULTS_DIR / "acf_distribution.png")
    
    # фильтрация только сезонных рядов
    seasonal_series, seasonal_meta = filter_seasonal(sampled_series, meta)
    
    # запуск моделей
    # Бейзлайны (Naive, ETS, Theta)
    results_baselines = run_baselines(seasonal_series, DEFAULT_HORIZON)
    
    # CatBoost со всеми вариантами фич
    results_catboost = run_all_catboost(seasonal_series, DEFAULT_HORIZON)
    
    # Объединение всех результатов для финального сравнения
    all_results = pd.concat([results_baselines, results_catboost], ignore_index=True)
    
    # сохранение финального графика
    # используем обновленную функцию с plt.close() внутри
    plot_metric_comparison(
        all_results, 
        metric="MASE", 
        title=f"Сравнение всех моделей (Horizon={DEFAULT_HORIZON})",
        save_path=RESULTS_DIR / "all_models_comparison.png"
    )
    
    plot_metric_comparison(
        all_results, 
        metric="sMAPE", 
        title=f"Сравнение sMAPE (Horizon={DEFAULT_HORIZON})",
        save_path=RESULTS_DIR / "smape_comparison.png"
    )
    
    # Дополнительный анализ
    print("\nИтоговая таблица результатов ")
    summary = summary_table(all_results)
    print(summary)
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv")
    
    # Анализ по силе сезонности
    strength_analysis = analyse_by_strength(results_catboost, seasonal_meta)
    print("\n Анализ CatBoost по силе сезонности ")
    print(strength_analysis)
    
    # Корреляция
    correlation_strength_vs_improvement(results_catboost, seasonal_meta)

    print(f"\nЭксперимент завершен. Все графики в папке: {RESULTS_DIR}")
    
    # анализ по силе сезонности (уже обученные данные)
    print("\n[Анализ] Проверка зависимости от силы сезонности...")
    plot_improvement_by_strength(
        results_catboost, 
        seasonal_meta, 
        save_path=RESULTS_DIR / "improvement_by_strength.png"
    )
    
    # анализ по горизонтам (это запустит повторное обучение для h=3, 12, 18)
    print("\n[Анализ] Проверка зависимости от горизонта прогнозирования...")
    plot_improvement_by_horizon(
        seasonal_series, 
        seasonal_meta, 
        save_dir=RESULTS_DIR
    )
    

if __name__ == "__main__":
    main()
