import torch
import joblib
import numpy as np

from collections import deque

from app.models.TransformerMacroDetector import TransformerMacroAutoencoder

from app.services.indicators import indicators_generation

import pandas as pd
import app.core.globals as g_vars
from multiprocessing import Queue

from app.utilites.points_to_features import points_to_features

from multiprocessing import Event

def inferece_plot_main(chart_queue: Queue, features, threshold, stop_event=None):
    import sys
    from app.services.RealTimeMonitor import RealTimeMonitor
    from PyQt6.QtCore import QTimer
    
    if stop_event is None:
        from multiprocessing import Event
        stop_event = Event()

    # 초기 threshold는 기본값으로 시작하지만, 
    # RealTimeMonitor 내부에서 동적으로 업데이트될 예정입니다.
    monitor = RealTimeMonitor(features, threshold)
    
    def update():
        if stop_event.is_set():
            timer.stop()
            monitor.app.quit()
            return

        try:
            while not chart_queue.empty():
                data = chart_queue.get_nowait()
                
                # 데이터 구조가 (tensor_np, error, current_threshold)라고 가정
                if len(data) == 3:
                    monitor.update_view(data[0], data[1], data[2])
                else:
                    # 이전 버전 호환용 (threshold가 없을 경우 고정값 사용)
                    monitor.update_view(data[0], data[1], threshold)
                    
        except (EOFError, BrokenPipeError, ConnectionResetError):
            timer.stop()
            monitor.app.quit()
        except Exception:
            pass
                
    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(16) # 약 60 FPS
    
    sys.exit(monitor.app.exec())
    
class MacroDetector:
    def __init__(self, model_path: str, seq_len=g_vars.SEQ_LEN, threshold=0.8, device=None, chart_Show=True, stop_event=None):
        self.seq_len = seq_len
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        allowance_time = 0.2

        self.base_threshold = g_vars.threshold * 1.1
        self.smooth_error_buf = deque(maxlen=5)
        self.macro_strike_count = 0 
        self.strike_limit = int(allowance_time / g_vars.tolerance)

        self.stop_event = stop_event
        if not self.stop_event:
            self.stop_event = Event()

        self.chart_Show = chart_Show
        self.plot_proc = None

        # ===== 모델 초기화 =====
        self.model = TransformerMacroAutoencoder(
            input_size=len(g_vars.FEATURES),
            d_model=g_vars.d_model,
            nhead=4,
            num_layers=g_vars.num_layers,
            dim_feedforward=128,
            dropout=g_vars.dropout
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        self.error_buffer = deque(maxlen=200) # 최근 에러 보관
        self.base_threshold = threshold # 최소 기준점

        self.scaler = joblib.load(g_vars.scaler_path)

        # ===== 좌표 buffer =====
        self.buffer = deque(maxlen=seq_len * 3)
        self.prev_speed = 0.0

    def start_plot_process(self):
        if not self.chart_Show:
            return

        if self.plot_proc and self.plot_proc.is_alive():
            return

        from multiprocessing import Process
        self.plot_proc = Process(
            target=inferece_plot_main,
            args=(
                g_vars.CHART_DATA,
                g_vars.FEATURES,
                self.threshold,
                self.stop_event
            ),
            daemon=False
        )
        self.plot_proc.start()

    def push(self, data:dict):
        self.buffer.append((data.get('x'), data.get('y'), data.get('timestamp'), data.get('deltatime')))

        if len(self.buffer) < self.seq_len * 3:
            return None
        
        return self._infer()

    def _infer(self):
        # 1. 데이터 수집 및 DataFrame 생성
        xs = [p[0] for p in self.buffer]
        ys = [p[1] for p in self.buffer]
        ts = [p[2] for p in self.buffer] 
        deltatime = [p[3] for p in self.buffer] 

        df = pd.DataFrame({"timestamp": ts, "x": xs, "y": ys, "deltatime" : deltatime})
        df = indicators_generation(df)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df[g_vars.FEATURES].copy()
        
        
        # settings 객체에 저장된 CLIP_BOUNDS를 사용하여 데이터 왜곡 방지
        if g_vars.CLIP_BOUNDS:
            for col in g_vars.FEATURES:
                if col in g_vars.CLIP_BOUNDS:
                    bounds = g_vars.CLIP_BOUNDS[col]
                    df[col] = df[col].clip(lower=bounds['min'], upper=bounds['max'])

        if len(df) < self.seq_len:
            return None
        
        # 2. 모델 입력을 위한 Feature 가공 및 스케일링
        X_infer, _ = points_to_features(df_chunk=df, seq_len=self.seq_len, stride=g_vars.STRIDE)

        if X_infer is None or X_infer.size == 0 or len(X_infer.shape) < 3:
            return None

        n_infer, seq_len, n_features = X_infer.shape
        X_infer_reshaped = X_infer.reshape(-1, n_features)
        
        # 학습된 스케일러 적용 (반드시 최신 파일이 로드된 상태여야 함)
        X_infer_scaled = self.scaler.transform(X_infer_reshaped)
        X_infer = X_infer_scaled.reshape(n_infer, seq_len, n_features)

        # 최신 시퀀스 추출 및 텐서 변환
        X_tensor = torch.tensor(X_infer[-1], dtype=torch.float32).unsqueeze(0).to(self.device)
      
        try:
            if hasattr(self.scaler, 'feature_names_in_'):
                print(f"DEBUG | Scaler features: {self.scaler.feature_names_in_}")
            else:
                print(f"DEBUG | Scaler is older version, cannot print feature names.")
            
            # 스케일링 된 후의 첫 번째 포인트 값 확인
            scaled_val = X_tensor[0, 0, :].cpu().numpy()
            print(f"DEBUG | Scaled Sample (First 5): {scaled_val[:5]}")
        except Exception as e:
            print(f"DEBUG | Logging error: {e}")
  

        # 3. 모델 추론 (Reconstruction)
        with torch.no_grad():
            output = self.model(X_tensor)
            # 학습 때 Est.Thresh를 뽑았던 방식과 동일하게 MAE 사용
            recon_error = torch.abs(output - X_tensor).mean().item()
            self.error_buffer.append(recon_error)
        
        # 4. 에러 평활화 (Smoothing)
        if not hasattr(self, 'smooth_error_buf'):
            from collections import deque
            self.smooth_error_buf = deque(maxlen=15)
        
        self.smooth_error_buf.append(recon_error)
        recon_error = np.mean(self.smooth_error_buf)
        
        # 5. 동적 임계치 계산 (MAD 기반)
        if len(self.error_buffer) >= 100:
            errors_np = np.array(self.error_buffer)
            median = np.median(errors_np)
            mad = np.median(np.abs(errors_np - median))
            dynamic_thresh = median + (5 * mad * 1.4826)
            current_threshold = max(self.base_threshold, dynamic_thresh)
        else:
            current_threshold = self.base_threshold

        # 6. Strike System (연속 판정 로직)
        is_anomaly = recon_error > current_threshold
        effective_strike_limit = max(5, int(0.05 / g_vars.tolerance)) 

        if not hasattr(self, 'macro_strike_count'):
            self.macro_strike_count = 0

        if is_anomaly:
            self.macro_strike_count += 2 # 이상치일 때 더 빠르게 증가
        else:
            # 즉시 0이 아니라 1씩 감소 (최소 0)
            self.macro_strike_count = max(0, self.macro_strike_count - 1)
 
        # 7. 최종 판정
        final_is_human = self.macro_strike_count < effective_strike_limit

        # 8. 모니터링 데이터 큐 전송
        if g_vars.CHART_DATA is not None:
            try:
                # 시각화를 위해 원본 텐서, 평활화된 에러, 현재 기준선을 보냄
                g_vars.CHART_DATA.put_nowait((X_tensor.cpu().numpy(), recon_error, current_threshold))
            except:
                pass

        return {
            "is_human": final_is_human, 
            "prob": recon_error, 
            "threshold": current_threshold,
            "strike": self.macro_strike_count,
            "limit": effective_strike_limit
        }