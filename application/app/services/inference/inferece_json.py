import time
import app.core.globals as g_vars
from datetime import datetime
from multiprocessing import Queue
from app.services.inference.macro_dectector import MacroDetector
from multiprocessing import Event
from queue import Empty
from tkinter import filedialog, messagebox
import os
import json

def main(stop_event=None, log_queue:Queue=None, chart_Show=True):
    use_existing = False
    if g_vars.init_model_path and g_vars.init_scale_path:
        if os.path.exists(g_vars.init_model_path) and os.path.exists(g_vars.init_scale_path):
            model_name = os.path.basename(g_vars.init_model_path)
            msg = f"이전에 사용한 모델을 다시 사용하시겠습니까?\n\n모델: {model_name}"
            use_existing = messagebox.askyesno("경로 재사용", msg)
        else:
            if log_queue: log_queue.put("⚠️ 이전 모델 파일이 경로에 없습니다. 새로 선택합니다.")

    # 2. '아니오'를 눌렀거나 기존 경로가 없는 경우에만 새로 선택
    if not use_existing:
        initial_dir = g_vars.scaler_path
        
        # (1) 모델 선택
        new_model_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="[1/2] 학습된 모델(.pt) 파일을 선택하세요",
            filetypes=(("PyTorch 모델", "*.pt"), ("모든 파일", "*.*"))
        )
        if not new_model_path:
            if log_queue: log_queue.put("❌ 모델 선택이 취소되었습니다.")
            return
        g_vars.init_model_path = new_model_path

        # (2) 스케일러 선택
        new_scale_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="[2/2] 해당 모델의 스케일러(.pkl) 파일을 선택하세요",
            filetypes=(("스케일러 파일", "*.pkl"), ("모든 파일", "*.*"))
        )
        if not new_scale_path:
            if log_queue: log_queue.put("❌ 스케일러 선택이 취소되었습니다.")
            return
        g_vars.init_scale_path = new_scale_path

    # 3. 최종 경로 확정 로그 (이 부분을 g_vars 사용으로 수정!)
    if log_queue:
        # local variable 대신 g_vars 값을 참조하여 에러 방지
        m_name = os.path.basename(g_vars.init_model_path)
        s_name = os.path.basename(g_vars.init_scale_path)
        log_queue.put(f"📂 로드 완료:\n- 모델: {m_name}\n- 스케일러: {s_name}")

    while True:
        if stop_event is None:
            stop_event = Event()

        # Detector 초기화
        detector = MacroDetector(
            model_path=g_vars.init_model_path,
            seq_len=g_vars.SEQ_LEN,
            threshold=g_vars.threshold,
            chart_Show=chart_Show,
            stop_event=stop_event,
            scale_path=g_vars.init_scale_path
        )

        if g_vars.INFERENCE_CHART_VIEW.value == False:
            with g_vars.PROCESS_LOCK:
                g_vars.INFERENCE_CHART_VIEW.value = True

            if log_queue:
                log_queue.put(f"✅ 차트 활성화 상태, 비교 분석 모드로 진해됩니다.")
            else:
                print(f"✅ 차트 활성화 상태, 비교 분석 모드로 진해됩니다.")
            detector.start_plot_process()

        if log_queue : log_queue.put(f"weight_threshold : {g_vars.weight_threshold}")
        else:
            print(f"weight_threshold : {g_vars.weight_threshold}")
            
        user_data:list[dict]

        file_pahh = filedialog.askopenfilename(title="Json 파일을 선택해 주세요", filetypes=(("json 파일", "*.json"), ("모든 파일", "*.*")))
        if not os.path.exists(file_pahh):
            return [] 

        try:
            with open(file_pahh, "r", encoding="utf-8") as f:
                data = json.load(f)
        
            user_data = data
        except Exception as e:
            print(e)
            user_data = []

        timeinterval = 7

        if g_vars.INFERENCE_CHART_VIEW.value == False:
            while timeinterval != 0:
                timeinterval -= 1
                if log_queue:
                    log_queue.put(f"inference 시작까지 count : {timeinterval}")
                else:
                    print(f"inference 시작까지 count : {timeinterval}")

                time.sleep(1)

        if log_queue:
            log_queue.put("🟢 Macro Detector Running")
        else:
            print("🟢 Macro Detector Running")

        g_vars.CHART_DATA.put_nowait("NEW_SESSION")

        all_raw_e = []
        try:
            for step in user_data:
                if stop_event.is_set():
                    if log_queue:
                        log_queue.put("🛑 Detector 중지")
                    else:
                        print("🛑 Detector 중지")
                    break
                data = {
                    'timestamp': datetime.fromisoformat(step.get("timestamp")),
                    'x': step.get("x"),
                    'y': step.get("y"),
                    'deltatime': step.get("deltatime")  
                }
                result = detector.push(data)

                if result:
                    # 확률 수치(float)를 가져옵니다.
                    m_prob = result.get('prob_value', 0.0) 
                    m_str = result.get('macro_probability', "0%")
                    raw_e = result.get('raw_error', 0.0)

                    if result["is_human"]:
                        log_msg = f"{m_str} (err: {raw_e:.4f})"
                    else:
                        # 매크로 판정 시 사이렌 이모지와 함께 확률 강조
                        log_msg = f"{m_str} (err: {raw_e:.4f}) 🚨"

                    all_raw_e.append(raw_e)

                    # 출력 대상 선택 (Queue 혹은 Print)
                    if log_queue:
                        log_queue.put(log_msg)
                    else:
                        print(log_msg)

            if all_raw_e:
                avg_raw_e = sum(all_raw_e) / len(all_raw_e)
                final_msg = f"📊 전체 구간 평균 에러: {avg_raw_e:.6f}"
                if log_queue:
                    log_queue.put(final_msg)
                else:
                    print(final_msg)

            if log_queue:
                log_queue.put(f"종료 : ctrl + shift + q")
            else:
                print("종료 : ctrl + shift + q")
        finally:
            detector.buffer.clear()
            try:
                while True:
                    g_vars.CHART_DATA.get_nowait()
            except Empty:
                pass

        user_input = "n"
        user_check = True
        for retry in range(5):
            user_input = input("추가 진행 하시겠습니까? (y/n): ").strip()
            if user_input not in ['y', 'n']:
                print(f"잘못 입력하셨습니다. 재시도 {retry} / 5")
            else:
                user_check = True
                break
            
            user_check = False
        
        if user_check == False:
            return
        
        if user_input == "n":
            return