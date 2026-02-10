import pandas as pd
import numpy as np

def indicators_generation(df_chunk: pd.DataFrame) -> pd.DataFrame:
    df = df_chunk.copy()

    for col in ['x', 'y', 'deltatime']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)

    dt = df["deltatime"].replace(0, 1e-6)

    # ===== 기본 이동 =====
    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["dist"] = np.hypot(df["dx"], df["dy"])

    df["speed"] = df["dist"] / dt
    df["acc"] = df["speed"].diff() / dt
    df["jerk"] = df["acc"].diff() / dt

    # ===== 방향 =====
    df["theta"] = np.arctan2(df["dy"], df["dx"])
    unwrapped = np.unwrap(df["theta"].fillna(0).values)
    df["angular_speed"] = (pd.Series(unwrapped, index=df.index).diff() / dt)
    df["direction_change"] = df["theta"].diff().abs()

    # ===== 🔥 매크로 분리용 핵심 =====

    # 1️⃣ micro shaking (인간은 미세 진동 많음)
    df["micro_shake"] = (df["dx"].diff().abs() + df["dy"].diff().abs())

    # 2️⃣ jerk window std (인간은 변동 큼)
    df["jerk_std"] = df["jerk"].rolling(5).std()

    # 3️⃣ speed window std
    df["speed_std"] = df["speed"].rolling(5).std()

    # 4️⃣ dt 변동성
    df["dt_std"] = dt.rolling(5).std()

    # 5️⃣ 방향 변화율
    df["direction_change_rate"] = df["direction_change"].rolling(5).mean()

    # 6️⃣ 선형성 점수 (직선이면 매크로 확률↑)
    total_dist = df["dist"].rolling(10).sum()
    straight_dist = np.hypot(df["x"].diff(10), df["y"].diff(10))
    df["linearity"] = straight_dist / (total_dist + 1e-6)

    # 7️⃣ 속도 자기상관 (매크로는 패턴 일정)
    df["speed_autocorr"] = df["speed"].rolling(10).corr(df["speed"].shift(1))

    # 8️⃣ 로그 변환으로 극단 강화
    df["log_jerk"] = np.sign(df["jerk"]) * np.log1p(np.abs(df["jerk"]))
    df["log_speed"] = np.log1p(df["speed"])

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
