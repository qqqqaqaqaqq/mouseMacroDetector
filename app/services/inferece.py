from pynput.mouse import Controller

import time
import app.core.globals as g_vars
from datetime import datetime
from multiprocessing import Queue

from app.services.macro_dectector import MacroDetector
from multiprocessing import Event

def main(stop_event=None, log_queue:Queue=None, chart_Show=True):
    if stop_event is None:
        stop_event = Event()

    detector = MacroDetector(
        model_path=g_vars.save_path,
        seq_len=g_vars.SEQ_LEN,
        threshold=g_vars.threshold,
        chart_Show=chart_Show,
        stop_event=stop_event
    )

    detector.start_plot_process()
    
    if log_queue:
        log_queue.put("🟢 Macro Detector Running")
    else:
        print("🟢 Macro Detector Running")

    mouse_controller = Controller()

    pre_x = None
    pre_y = None

    # 간격
    tolerance = g_vars.tolerance

    # 초기값 설정
    start_time = time.perf_counter()
    end_time = time.perf_counter()

    error_start_time = None
    while not stop_event.is_set():
        try:
            # --- 보호 모드 탈출 성공 시 시간 계산 ---
            if error_start_time is not None:
                total_error_duration = time.perf_counter() - error_start_time
                print(f"✅ 보호 모드 해제 (지속 시간: {total_error_duration:.2f}초)")

            error_start_time = None # 시간 초기화            
            if end_time - start_time < tolerance:
                
                end_time = time.perf_counter()
                continue

            x, y = mouse_controller.position

            if pre_x is None or pre_y is None:
                pre_x, pre_y = x, y
                start_time = end_time = time.perf_counter()
                continue

            if x == pre_x and y == pre_y:
                start_time = end_time = time.perf_counter()
                continue

            delta = end_time - start_time

            data = {
                'timestamp': datetime.now().isoformat(),
                'x': int(x),
                'y': int(y),
                'deltatime': delta
            }

            pre_x, pre_y = x, y

            result = detector.push(data)

            if result:
                if result["is_human"]:
                    if log_queue:
                        log_queue.put(f"🙂 HUMAN | prob={result['prob']:.3f}")
                    else:
                        print(f"🙂 HUMAN | prob={result['prob']:.3f}")
                else:
                    if log_queue:
                        log_queue.put(f"🚨 MACRO | prob={result['prob']:.3f}") 
                    else:
                        print(f"🚨 MACRO | prob={result['prob']:.3f}") 
        except Exception as e:
                # 에러가 처음 발생한 시점 기록
                if error_start_time is None:
                    error_start_time = time.perf_counter()
                    print(f"🚨 보호 모드 진입 (원인: {e})")

                current_error_duration = time.perf_counter() - error_start_time
                print(f"🟢 보호 모드 작동 중... ({current_error_duration:.1f}초 경과)", end="\r")

                time.sleep(1)
                
                start_time = time.perf_counter()
                end_time = time.perf_counter()
                continue

    if log_queue:
        log_queue.put("🛑 Macro Detector Stopped")
    else:
        print("🛑 Macro Detector Stopped")

    stop_event.set()    