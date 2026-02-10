import torch
import joblib
import numpy as np
import pandas as pd
from collections import deque
from multiprocessing import Queue, Event
import sys

from sklearn.preprocessing import RobustScaler
import app.core.globals as g_vars
from app.models.TransformerMacroDetector import TransformerMacroAutoencoder
from app.core.indicators import indicators_generation

from app.utilites.make_gauss import make_gauss
from app.utilites.make_sequence import make_seq

from app.utilites.loss_caculation import Loss_Calculation, inference_score

def inferece_plot_main(chart_queue: Queue, features, threshold, chart_view, process_lock, stop_event=None):
    from app.utilites.plot_monitor import RealTimeMonitor
    from PyQt6.QtCore import QTimer
    

    exit_code = 1
    if stop_event is None:
        stop_event = Event()

    try:
        monitor = RealTimeMonitor(features, threshold)
        
        def update():
            if stop_event.is_set():
                with process_lock:
                    chart_view.value = False            
                monitor.app.quit()
                return

            try:
                while not chart_queue.empty():
                    data = chart_queue.get_nowait()
                    
                    if isinstance(data, str):
                        if data == "NEW_SESSION":
                            monitor.update_view(data, None, None)
                        continue                

                    # 데이터 구조에 따른 업데이트 (tensor, error, threshold)
                    if len(data) == 3:
                        monitor.update_view(data[0], data[1], data[2])
                    else:
                        monitor.update_view(data[0], data[1], threshold)
                        
            except (EOFError, BrokenPipeError, ConnectionResetError):
                monitor.app.quit()
            except Exception:
                pass
                    
        # 3. 타이머 설정 (약 60FPS)
        timer = QTimer()
        timer.timeout.connect(update)
        timer.start(16)

        exit_code = monitor.app.exec()

    except Exception as e:
        print(f"❌ 차트 프로세스 에러: {e}")
    finally:
        with process_lock:
            chart_view.value = False
        print(f"✅ 차트 리소스 반납 완료 {chart_view.value}")

        sys.exit(exit_code)
    

class MacroDetector:
    def __init__(self, model_path: str, seq_len=g_vars.SEQ_LEN, threshold=None, device=None, chart_Show=True, stop_event=None):
        self.seq_len = seq_len
        self.base_threshold = threshold if threshold is not None else g_vars.threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.allowable_add_data = g_vars.SEQ_LEN + 5

        self.buffer = deque(maxlen=self.allowable_add_data)

        self.stop_event = stop_event or Event()
        self.chart_Show = chart_Show
        self.plot_proc = None

        # ===== 모델 초기화 =====
        self.model = TransformerMacroAutoencoder(
            input_size=len(g_vars.FEATURES),
            d_model=g_vars.d_model,
            nhead=g_vars.n_head,
            num_layers=g_vars.num_layers,
            dim_feedforward=g_vars.dim_feedforward,
            dropout=g_vars.dropout
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.scaler:RobustScaler = joblib.load(g_vars.scaler_path)

    def push(self, data: dict):
        self.buffer.append((data.get('x'), data.get('y'), data.get('timestamp'), data.get('deltatime')))
        
        # 최소 seq_len + 1은 채워져야 분석 시작
        if len(self.buffer) < self.allowable_add_data:
            return None
        
        return self._infer()

    def start_plot_process(self):
        """실시간 차트 프로세스를 시작합니다."""
        if not self.chart_Show or (self.plot_proc and self.plot_proc.is_alive()):
            return

        from multiprocessing import Process

        self.plot_proc = Process(
            target=inferece_plot_main, 
            kwargs={
                "chart_queue" : g_vars.CHART_DATA, 
                "features" : g_vars.FEATURES, 
                "threshold" : self.base_threshold, 
                "stop_event" : self.stop_event, 
                "chart_view" : g_vars.INFERENCE_CHART_VIEW,
                "process_lock" : g_vars.PROCESS_LOCK
            },
            daemon=False
        )
        self.plot_proc.start()




    def _infer(self):
        df = pd.DataFrame(list(self.buffer), columns=["x", "y", "timestamp", "deltatime"])
        df = indicators_generation(df)

        df_filter_chunk = df[g_vars.FEATURES].copy()

        chunks_scaled_array = self.scaler.transform(df_filter_chunk)
        
        chunks_scaled_df = pd.DataFrame(chunks_scaled_array, columns=g_vars.FEATURES)

        chunks_scaled = chunks_scaled_df.values

        final_input:np.array = make_seq(data=chunks_scaled, seq_len=g_vars.SEQ_LEN, stride=1)

        last_seq = torch.tensor(final_input[-1], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(last_seq)
            avg_error = inference_score(outputs=output, batch=last_seq, device=self.device).item()

            # 임계치 판정 logic
            is_human = avg_error <= self.base_threshold
            
            if not is_human:
                if hasattr(self, 'log_queue'):
                    print(f"🚨 [DETECTION] Error: {avg_error:.4f}")

        if g_vars.CHART_DATA is not None:
            try:
                # 시각화 프로세스에 현재 시퀀스의 특징값(평균)과 오차 정보를 보냄
                current_features_mean = chunks_scaled[-1] # 가장 최신 시점의 스케일링된 지표들
                g_vars.CHART_DATA.put_nowait((current_features_mean, avg_error, self.base_threshold))
            except Exception: 
                pass

        return {
            "is_human": is_human,
            "macro_probability": "🚨 MACRO" if not is_human else "🙂 HUMAN",
            "prob_value": avg_error, # score 대신 error 값 전달
            "raw_error": round(avg_error, 5),
            "threshold": self.base_threshold
        }