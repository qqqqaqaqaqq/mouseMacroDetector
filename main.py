import sys
import os
import time

# 1. 가장 먼저 보이는 메시지
print("🚀 프로그램 실행 중... 잠시만 기다려 주세요.")

# 2. 가장 무거운 torch 로딩 시각화
print("📦 라이브러리 로드 중 (PyTorch)...", end="\r")
import torch
print("📦 라이브러리 로드 완료 (PyTorch)   ")

print("⚙️ 시스템 환경 설정 중...", end="\r")
import multiprocessing 
import app.core.globals as g_vars
import ctypes
from app.core.settings import settings
print("⚙️ 시스템 환경 설정 완료         ")

if __name__ == "__main__":
    multiprocessing.freeze_support() 

    print("Welcome")
    user_input = input("inference Mode? (y/n): ").lower()
    
    if user_input not in ['y', 'n']:
        print(f"❌ 잘못된 입력입니다 ('{user_input}').")
        for i in range(3, 0, -1):
            print(f"⚠️ {i}초 후 프로그램이 종료됩니다...", end="\r")
            time.sleep(1)
        print("\nBye! 👋")
        sys.exit() # 프로그램 종료

    inference_Mode = user_input == 'y'

    if inference_Mode:
        import keyboard
        import threading
        import app.services.inferece as inference
        from multiprocessing import Event
        from app.utilites.resource_monitoring import ResourceMonitor
        
        user_input2 = input("chart Show? (y/n): ").lower()
        

        if user_input2 not in ['y', 'n']:
            print(f"❌ 잘못된 입력입니다 ('{user_input2}').")
            for i in range(3, 0, -1):
                print(f"⚠️ {i}초 후 프로그램이 종료됩니다...", end="\r")
                time.sleep(1)
            print("\nBye! 👋")
            sys.exit() # 프로그램 종료

        chart_Show = user_input2 == 'y'

        ctypes.windll.kernel32.SetConsoleTitleW("Inference Mode (Quit: CTRL+SHIFT+Q)")
        g_vars.init_manager()

        stop_move_event = Event()

        def trigger_stop_event():
            stop_move_event.set()
            print("\n🛑 STOP SIGNAL RECEIVED (CTRL+SHIFT+Q)")

        keyboard.add_hotkey('ctrl+shift+q', trigger_stop_event)

        def console_resource_logger(stop_ev, monitor):
            # 초기 타이틀 설정
            base_title = "Inference Mode (Quit: CTRL+SHIFT+Q)"
            
            while not stop_ev.is_set():
                stats = monitor.get_stats()
                
                # 타이틀에 들어갈 문자열 구성
                new_title = f"{base_title} | CPU: {stats['cpu']} | RAM: {stats['ram']} | GPU: {stats['gpu']}"
                
                # 실시간으로 윈도우 타이틀 변경
                ctypes.windll.kernel32.SetConsoleTitleW(new_title)
                
                time.sleep(1) # 1초 간격 갱신
            
            # 종료 시 타이틀 복구
            ctypes.windll.kernel32.SetConsoleTitleW("Inference Stopped")
            
        # 모니터 객체 생성 및 스레드 시작
        monitor = ResourceMonitor()
        res_thread = threading.Thread(
            target=console_resource_logger, 
            args=(stop_move_event, monitor), 
            daemon=True # 프로그램 종료 시 자동 종료
        )
        res_thread.start()

        # 인퍼런스 실행 (이 함수가 종료될 때까지 대기함)
        inference.main(
            stop_event=stop_move_event,
            chart_Show=chart_Show
        )
    else:
        from app.gui.main_window import VantageUI
        from app.db.session import init_db
        from PyQt6.QtWidgets import QApplication

        if settings.Recorder == "postgres":
            print("실행")
            init_db()

        g_vars.init_manager()

        app = QApplication(sys.argv)
        window = VantageUI()
        window.show()
        sys.exit(app.exec())