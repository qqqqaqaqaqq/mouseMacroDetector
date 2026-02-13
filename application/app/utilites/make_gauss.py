import numpy as np
import sys
import pandas as pd
from numpy.lib.stride_tricks import as_strided

def make_gauss(data: pd.DataFrame, chunk_size: int, chunk_stride: int, offset: int, train_mode:bool=True) -> np.array:
    # 1. 데이터 준비 및 메모리 뷰 생성 (복사 비용 0)
    data_np = data.values[offset:].astype(np.float64)
    n_samples, n_features = data_np.shape
    eps = 1e-9

    num_chunks = (n_samples - chunk_size) // chunk_stride + 1
    if num_chunks <= 0: return np.array([])
    
    # Sliding Window 생성 (as_strided 사용으로 메모리 효율 극대화)
    itemsize = data_np.itemsize
    chunks = as_strided(
        data_np,
        shape=(num_chunks, chunk_size, n_features),
        strides=(chunk_stride * n_features * itemsize, n_features * itemsize, itemsize)
    )

    # 2. 벡터화 가능한 연산은 루프 밖에서 한 번에 처리 (가장 빠름)
    m = np.mean(chunks, axis=1, keepdims=True)
    diff = chunks - m
    s = np.std(chunks, axis=1, ddof=0)
    s_safe = s + eps

    sk = np.mean(diff**3, axis=1) / (s_safe**3)  # 왜도
    roughness = np.mean(np.abs(np.diff(chunks, axis=1)), axis=1)  # 거칠기
    theo_entropy = 0.5 * np.log2(2 * np.pi * np.e * (s_safe**2) + eps)  # 이론적 엔트로피

    # 3. 실측 엔트로피 계산 (진행바 유지를 위해 루프 사용)
    actual_entropy = np.zeros((num_chunks, n_features))
    total_steps = num_chunks

    for idx in range(total_steps):
        window = chunks[idx]
        
        # 컬럼별 엔트로피 계산 (이 부분은 histogram 특성상 루프가 필요함)
        for col in range(n_features):
            counts, _ = np.histogram(window[:, col], bins=10)
            p = counts / (counts.sum() + eps)
            p = p[p > 0]
            actual_entropy[idx, col] = -np.sum(p * np.log2(p))

        # --- 요청하신 진행바 로직 그대로 유지 ---
        if train_mode:
            if (idx + 1) % max(1, (total_steps // 50)) == 0 or (idx + 1) == total_steps:
                progress = (idx + 1) / total_steps
                bar = '■' * int(20 * progress) + '□' * (20 - int(20 * progress))
                sys.stdout.write(f'\r진행중: [{bar}] {progress*100:>5.1f}% ({idx+1}/{total_steps})')
                sys.stdout.flush()

    # 4. 엔트로피 갭 계산 및 최종 병합
    entropy_gap = theo_entropy - actual_entropy
    
    # 기존 input_feature 순서대로 병합: [sk, actual_entropy, entropy_gap, roughness]
    result = np.concatenate([sk, entropy_gap, roughness], axis=1)

    if train_mode:
        sys.stdout.write('\n')
        sys.stdout.flush()

    return result