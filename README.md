# 🤖 Model Architecture: Transformer Macro Autoencoder

Transformer 기반의 오토인코더(Autoencoder) 구조
- 정상 패턴: 모델이 높은 정확도로 복원하여 재구성 오차 0에 수렴합니다.
- 이상 패턴(매크로): 모델이 학습하지 못한 패턴이므로 복원 능력이 떨어져 재구성 오차가 높게 발생합ㄴ다.

- Feature Embedding : 5차원의 입력 피처(x, y, dist 등)를 d_model(64차원)의 고차원 벡터로 확장하여 복잡한 상관관계를 학습할 준비를 합니다.
- Positional Encoding : Transformer는 RNN과 달리 순서 정보가 없으므로, 시퀀스 내 각 위치 정보($1^{st}, 2^{nd}, ...$)를 나타내는 벡터를 더해줍니다.
- Transformer Encoder : Multi-Head Self-Attention 메커니즘을 통해 시퀀스 전체를 동시에 훑으며, 과거의 움직임이 현재에 미치는 영향을 파악합니다.
- Linear Decoder : 인코더가 뽑아낸 추상적인 특징들을 다시 원래의 5개 피처 차원으로 복원합니다.

Detection Logic
- Normal Patterns: The model reconstructs these with high precision, causing the reconstruction error to converge to zero.
- Anomalous Patterns (Macro): Since these are patterns the model has not encountered during training, the reconstruction capability decreases, resulting in a high reconstruction error.

- Feature Embedding: Expands the 5-dimensional input features (e.g., $x, y, dist$) into a high-dimensional vector of $d_{model}$ (64 dimensions) to prepare the model for learning complex correlations.
- Positional Encoding: Since Transformers do not inherently process sequential order like RNNs, this adds vectors that represent the positional information ($1^{st}, 2^{nd}, \dots$) within the sequence.
- Transformer Encoder: Utilizes the Multi-Head Self-Attention mechanism to scan the entire sequence simultaneously, capturing how past movements influence the present state.
- Linear Decoder: Reconstructs the abstract features extracted by the encoder back into the original 5-feature dimensions.

![Architecture Diagram](./public/Architecture.png)

# 정식 1.0.0 버전 출시 전까지 기능 개선 및 안정화를 위해 빈번한 업데이트가 진행될 예정입니다.
# Frequent updates are expected for feature enhancement and stabilization until the official v1.0.0 release.

# 🚀 Macro Detector Update (Ver 0.0.5)

### 🇰🇷 한글 업데이트 내역 (Korean)

#### 📊 주요 변경 사항
* **데이터셋 확장**: 실제 게임 환경 기반의 마우스 좌표 데이터를 대폭 추가하여 실전 탐지 성능 강화.
* **추론 시각화 지원**: JSON 데이터 추론 시, 탐지 결과를 시각적으로 확인 가능한 **가우스 정규분포 차트** 출력을 지원합니다.
  ![Architecture Diagram](./public/chart.png)
* **알고리즘 최적화**: 
    * 새로운 Feature 추출 로직 및 모델 파라미터 조정 (세부 사항은 `config.json` 참고).

#### 🏗️ 시스템 아키텍처 및 UI
* **프로젝트 구조 분리**: 유지보수를 위해 `application`, `backend`, `frontend` 레이어로 모듈화.
* **보안 섹션 UI/UX**: 보안 대시보드 전용 UI를 추가하여 데이터 모니터링 편의성 증대.

---

### 🇺🇸 English Update Notes

#### 📊 Key Updates
* **Dataset Expansion**: Integrated extensive mouse coordinate data from real-world gaming environments for enhanced robustness.
* **Inference Plotting**: Supports **Gaussian normal distribution charts** to visualize inference results during analysis.
* **Algorithmic Refinement**: 
    * **Loss Calculation**: Switched from `MSE` to **`MAE`** (Mean Absolute Error) for better handling of outliers.
    * New feature extraction logic and tuned model parameters (See `config.json`).

#### 🏗️ Architecture & UI
* **Modular Structure**: Decoupled into `application`, `backend`, and `frontend` layers for better scalability.
* **Security UI/UX**: Introduced a dedicated security dashboard and components for professional monitoring.

---

### 📦 Installation & Upgrade (설치 및 업데이트)

```bash
pip install git+https://github.com/qqqqaqaqaqq/mouseMacroLibrary.git

---

## 📂 Data Management
* **Database Support:** Efficient data handling using **PostgreSQL** and **JSON** formats.

## 🛠 Installation
* To install the required dependencies, run the following command:
  ```bash
  pip install -r requirements.txt

## 사용 설명서 (Manual)
Manual.pptx

## 영상
[![실행 영상](https://img.youtube.com/vi/iwi31PxQc3I/0.jpg)](https://youtu.be/iwi31PxQc3I)