import pandas as pd
import numpy as np

def indicators_generation(df_chunk: pd.DataFrame) -> pd.DataFrame:
    df = df_chunk.copy()
    
    # 0. 시간 안전장치 (dt가 너무 작으면 속도가 폭발함)
    # 최소 단위를 1ms(0.001)로 보정하고 아주 작은 epsilon을 더합니다.
    dt = df["deltatime"].clip(lower=0.001) + 1e-6
    
    # 1. 기본 물리량 계산
    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["dist"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    
    # speed, acc, jerk 계산
    df["speed"] = df["dist"] / dt
    df["acc"] = df["speed"].diff()
    df["jerk"] = df["acc"].diff()
    df["jerk_diff"] = df["jerk"].diff()

    # 2. 방향 및 회전 관련
    df["angle"] = np.arctan2(df["dy"], df["dx"])
    df["turn"] = (df["angle"].diff() + np.pi) % (2 * np.pi) - np.pi
    df["turn_acc"] = df["turn"].diff()
    
    # 3. 보간법 탐지용 통계 피처
    df["jerk_std"] = df["jerk"].rolling(window=5, min_periods=1).std()
    df["speed_var"] = df["speed"].rolling(window=3, center=True, min_periods=1).std()
    
    # 직선성 (분모 0방지 위해 clip)
    rolling_dist_sum = df["dist"].rolling(window=5).sum()
    line_dist = np.sqrt((df["x"] - df["x"].shift(4))**2 + (df["y"] - df["y"].shift(4))**2).clip(lower=1e-6)
    df["straightness"] = (rolling_dist_sum / line_dist).fillna(1.0)

    # -------------------------------------------------------
    # 4. [핵심 추가] 수치 압축 및 안정화 (Log Scaling)
    # -------------------------------------------------------
    # 값이 수만 단위로 튀는 피처들을 로그로 압축합니다. (ln(1+|x|) 방식)
    log_features = ["speed", "acc", "jerk", "jerk_diff", "jerk_std", "speed_var"]
    for col in log_features:
        if col in df.columns:
            # 부호 유지하며 로그 압축: 10,000 -> 약 9.2
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    # 직선성 피처는 튀는 값(Outlier)이 많으므로 상한선 설정
    df["straightness"] = df["straightness"].clip(1, 10)

    # 5. NaN/inf 최종 정리
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df