import pandas as pd
import numpy as np

def indicators_generation(df_chunk: pd.DataFrame) -> pd.DataFrame:
    df = df_chunk.copy()
    eps = 1e-9
    dt = df["deltatime"]

    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["dist"] = np.hypot(df["dx"], df["dy"])

    df["speed"] = df["dist"] / dt
    df["acc"] = df["speed"].diff() / dt
    df["jerk"] = df["acc"].diff() / dt

    df["theta"] = np.arctan2(df["dy"], df["dx"])
    df["angle_vel"] = df["theta"].diff().pipe(lambda x: np.arctan2(np.sin(x), np.cos(x))) / dt

    # 4. Micro-shake
    df["micro_shake"] = (df["speed"].diff().abs() + df["angle_vel"].diff().abs())

    # 5. Curvature
    x1, y1 = df["x"].shift(3), df["y"].shift(3)
    x2, y2 = df["x"].shift(6), df["y"].shift(6)
    a = np.hypot(x1 - x2, y1 - y2)
    b = np.hypot(df["x"] - x1, df["y"] - y1)
    c = np.hypot(df["x"] - x2, df["y"] - y2)
    s = (a + b + c) / 2
    area = np.sqrt(np.maximum(0, s * (s - a) * (s - b) * (s - c)))
    df["curvature"] = np.where(a*b*c > eps, (4 * area) / (a * b * c + eps), 0)

    # 6. 비선형성 (energy_impact)
    df["energy_impact"] = df["acc"] * df["jerk"]
    df["low_speed_const_acc"] = np.where(df["speed"] < 1.0, df["acc"].rolling(5).std(), 1.0)
    
    # 결측치 처리
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df