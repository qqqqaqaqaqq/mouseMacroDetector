import sys
import os

def get_resource_path(relative_path):
    """ 개발 환경과 PyInstaller 환경 모두에서 작동하는 경로 변환 함수 """
    if getattr(sys, 'frozen', False):
        # 빌드 후: _internal 폴더 (또는 _MEIPASS) 기준
        base_path = sys._MEIPASS
    else:
        # 개발 환경: 현재 프로젝트의 '루트' 폴더를 기준으로 설정
        # 만약 이 코드가 /app/gui/ 안에 있다면 .parent.parent를 이용해 루트로 이동
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # 만약 개발 환경에서 'app/gui/style.css'라고 불렀는데 
        # 실제 경로가 /app/gui/app/gui/style.css가 된다면?
        # 아래처럼 현재 경로에 'app'이 포함되어 있는지 체크해서 중복을 방지합니다.
        if "app" in base_path:
             # 'app' 폴더 상위인 루트 경로를 base_path로 잡음
             base_path = base_path.split("app")[0]

    return os.path.normpath(os.path.join(base_path, relative_path))