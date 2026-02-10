import torch

def inference_score(outputs, batch, device):
    # 0. 입력에 nan이 있는지 체크 (디버깅용)
    if torch.isnan(batch).any() or torch.isnan(outputs).any():
        return torch.tensor(0.0).to(device)

    abs_err = torch.abs(outputs - batch)
    
    # 1. Z-Score 계산
    feature_mean = torch.mean(batch, dim=(0, 1))
    feature_std = torch.std(batch, dim=(0, 1))
    
    # 표준편차가 0인 경우(변동성 없음) z_score를 0으로 강제 고정
    # 0으로 나누는 것을 방지하는 가장 안전한 방법
    safe_std = torch.where(feature_std == 0, torch.ones_like(feature_std), feature_std)
    z_score = torch.abs((batch - feature_mean) / safe_std)
    z_score = torch.where(feature_std.expand_as(z_score) == 0, torch.zeros_like(z_score), z_score)
    
    # 2. 가중치 적용
    weight = torch.where(z_score > 2.0, 0.1, 1.0)
    weight = torch.where(z_score < 0.2, 5.0, weight)
    
    weighted_err = abs_err * weight
    
    # 3. 최종 결과 nan 체크
    result = torch.mean(weighted_err, dim=(1, 2))
    return torch.where(torch.isnan(result), torch.tensor(0.0).to(self.device), result)

def Loss_Calculation(outputs, batch):
    abs_err = torch.abs(outputs - batch)
    
    # 1. 배치 내 각 피처별 평균과 표준편차 계산 (dim 0, 1 기준)
    # feature_std shape: (5,)
    feature_mean = torch.mean(batch, dim=(0, 1))
    feature_std = torch.std(batch, dim=(0, 1)) + 1e-6 # 0 나누기 방지
    
    # 2. Z-Score 계산 (현재 값이 평균에서 몇 표준편차만큼 떨어져 있는지)
    # z_score: (batch, seq, feature)
    z_score = torch.abs((batch - feature_mean) / feature_std)
    
    # 3. [유저 특징] 2시그마(상위 약 5%)를 벗어나는 극단치에 가중치 부여
    # 일반적인 유저의 '튀는 동작'을 더 강하게 학습
    weight = torch.where(z_score > 2.0, 15.0, 1.0)
    
    # 4. [매크로 특징] 반대로 Z-Score가 너무 낮은(변동성이 없는) 구간 패널티
    # 평균에 너무 딱 붙어 있는(0.2시그마 미만) 정적 구간
    static_mask = z_score < 0.2
    weight = torch.where(static_mask, 10.0, weight)
    
    weighted_err = abs_err * weight
    
    # 샘플별 평균 에러 반환
    return torch.mean(weighted_err, dim=(1, 2))

def MSE_Loss(outputs, batch):
    return torch.mean((outputs - batch)**2, dim=(1, 2)) 

def MAE_Loss(outputs, batch):
    return torch.mean(torch.abs(outputs - batch), dim=(1, 2))

def RMSE_Loss(outputs, batch):
    return torch.sqrt(torch.mean((outputs - batch)**2, dim=(1, 2)) + 1e-6)

def Huber_Loss_Score(outputs, batch, delta=1.0):
    errors = torch.abs(outputs - batch)
    # Huber 로직: 오차가 delta보다 작으면 0.5 * error^2, 크면 delta * (error - 0.5 * delta)
    quad = torch.min(errors, torch.tensor(delta).to(errors.device))
    lin = errors - quad
    
    # 샘플별 평균 계산 (batch_size 크기의 벡터 반환)
    sample_huber = (0.5 * quad**2 + delta * lin).mean(dim=(1, 2))
    return sample_huber

def TopK_Loss_Score(outputs, batch, k=0.1):
    # 1️⃣ 제곱 오차로 극단 강조
    errors = (outputs - batch) ** 2      # (B, T, F)

    B = errors.size(0)
    errors = errors.view(B, -1)          # (B, T*F)

    # 2️⃣ k 안전장치
    k_val = max(1, int(errors.size(1) * k))

    # 3️⃣ 상위 k% 선택
    topk_errors, _ = torch.topk(errors, k_val, dim=1)

    # 4️⃣ 샘플별 평균 반환 (B,)
    return topk_errors.mean(dim=1)