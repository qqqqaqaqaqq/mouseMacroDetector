import pandas as pd
import numpy as np

def make_gauss(data: pd.DataFrame, chunk_size: int, chunk_stride: int, offset: int) -> np.array:
    data_np = data.values[offset:] 
    chunks = []
    eps = 1e-9

    for i in range(0, len(data_np) - chunk_size + 1, chunk_stride):
        window = data_np[i : i + chunk_size] # (chunk_size, num_features)
        
        # 1. 기존 통계량 (Moment)
        m = np.mean(window, axis=0)
        s = np.std(window, axis=0)
        diff = window - m
        sk = np.mean(diff**3, axis=0) / (s**3 + eps)
        kt = np.mean(diff**4, axis=0) / (s**4 + eps) - 3
        
        # 2. [추가] 선형성 및 연속성 지표 (매크로 고립 핵심)
        # 윈도우 내 가속도 변화율의 평균 (매크로는 0에 수렴)
        diff_1 = np.diff(window, axis=0)
        roughness = np.mean(np.abs(diff_1), axis=0)
        
        # 3. [추가] 자기상관 (Autocorrelation lag-1)
        # 매크로는 앞뒤 데이터가 너무 비슷해서 1에 가깝고, 유저는 낮음
        # 단순 구현을 위해 (t)와 (t-1)의 상관계수 개념 도입
        if chunk_size > 1:
            ac = np.mean(window[1:] * window[:-1], axis=0) / (np.var(window, axis=0) + eps)
        else:
            ac = np.zeros(window.shape[1])

        # 모든 지표 결합
        chunks.append(np.concatenate([m, s, sk, kt, roughness, ac]))

    return np.array(chunks)