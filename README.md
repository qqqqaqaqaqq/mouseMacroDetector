# 🤖 Model Architecture: Transformer Macro Autoencoder

Transformer 기반의 오토인코더(Autoencoder) 구조
- 정상 패턴: 모델이 높은 정확도로 복원하여 재구성 오차 0에 수렴합니다.
- 이상 패턴(매크로): 모델이 학습하지 못한 패턴이므로 복원 능력이 떨어져 재구성 오차가 높게 발생합ㄴ다.

Detection Logic
- Normal Patterns: The model reconstructs these with high precision, causing the reconstruction error to converge to zero.
- Anomalous Patterns (Macro): Since these are patterns the model has not encountered during training, the reconstruction capability decreases, resulting in a high reconstruction error.

![Architecture Diagram](./public/Architecture.png)

# 정식 1.0.0 버전 출시 전까지 기능 개선 및 안정화를 위해 빈번한 업데이트가 진행될 예정입니다.
# Frequent updates are expected for feature enhancement and stabilization until the official v1.0.0 release.

# 🚀 Macro Detector Update (Ver 0.0.5)

#### 📊 주요 변경 사항

데이터 저장 변경 점
postgres 지원 삭제 -> json 온리

학습 변경 점
학습 데이터셋 변경 -> 지표에 대한 가우스 정규 분포 학습으로 변경, Chunk_size 제공
손실 계산 MAE 로 변경, 편차 계산 RobustScaler
Domain 특화로 변경 -> fps용, 홈페이지 매크로 마우스 탐지용 등 모델 분화 (데이터 특징에 의해 종합 판단은 불가라 판단)

추론 변경 점
실시간 탐지 -> 마우스 간섭 문제로 삭제
가우시안 차트 제공 -> 데이터가 가우시안 차트와 비슷할 수록 human 멀어질수록 macro

---

### 📦 Libray 지원

```bash
pip install git+https://github.com/qqqqaqaqaqq/mouseMacroLibrary.git

---