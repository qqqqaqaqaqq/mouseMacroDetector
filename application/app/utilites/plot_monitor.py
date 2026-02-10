import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import numpy as np

class RealTimeMonitor:
    def __init__(self, features, threshold, max_points=300):
        self.app = QApplication.instance() or QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="Adaptive Y-Axis Multi-Session Monitor")
        self.win.resize(1200, 900)
        self.win.setBackground('#121212')

        self.features = features
        self.max_points = max_points
        
        self.plots = []
        self.active_curves = []
        self.colors = ['#00BFFF', '#FFD700', '#ADFF2F', '#FF69B4', '#FFA500', '#00FA9A', '#FF4500']
        self.current_color_idx = -1
        
        self.error_history = []
        self._setup_layouts()
        self._setup_status_plot(threshold)
        self.start_new_session()

    def _setup_layouts(self):
        self.win.addLabel("<b>[ Feature Time-Series: Adaptive Y-Axis ]</b>", colspan=3, color='#FFFFFF')
        self.win.nextRow()
        
        for i, f in enumerate(self.features):
            p = self.win.addPlot(title=f"{f}")
            p.showGrid(x=True, y=True, alpha=0.1)
            
            # X축은 고정
            p.setXRange(0, self.max_points)
            
            # [핵심 수정] Y축 자동 범위 설정
            # enableAutoRange(y=True)는 들어오는 데이터의 min/max를 자동으로 추적합니다.
            p.enableAutoRange(axis='y', enable=True)
            p.setAutoVisible(y=True) # 현재 보이는 데이터 범위 내에서만 스케일 조정
            
            self.plots.append(p)
            if (i + 1) % 3 == 0: 
                self.win.nextRow()
        self.win.nextRow()

    def _setup_status_plot(self, threshold):
        self.status_plot = self.win.addPlot(title="Anomaly Score", colspan=3)
        self.status_plot.setFixedHeight(180)
        self.status_plot.setXRange(0, self.max_points)
        self.status_plot.enableAutoRange(axis='y', enable=True) # 에러 축도 가변적으로
        self.status_plot.showGrid(x=True, y=True, alpha=0.2)
        
        self.thresh_line = pg.InfiniteLine(pos=threshold, angle=0, pen=pg.mkPen('#FF4444', width=2, style=Qt.PenStyle.DashLine))
        self.status_plot.addItem(self.thresh_line)
        self.error_curve = None

    def start_new_session(self):
        self.current_color_idx = (self.current_color_idx + 1) % len(self.colors)
        color = self.colors[self.current_color_idx]
        
        self.current_session_data = [[] for _ in range(len(self.features))]
        self.active_curves = []
        
        for i, p in enumerate(self.plots):
            # 새 세션 선 추가
            curve = p.plot(pen=pg.mkPen(color, width=1.5))
            self.active_curves.append(curve)
            
        self.error_history = []
        self.error_curve = self.status_plot.plot(pen=pg.mkPen(color, width=2))

    def update_view(self, current_features, avg_error, current_threshold):
        if current_features is None: return
        
        if isinstance(current_features, str) and current_features == "NEW_SESSION":
            self.start_new_session()
            return

        try:
            # 1. 상단 피처 그래프 업데이트
            for i in range(len(self.features)):
                val = current_features[i]
                self.current_session_data[i].append(val)
                
                if len(self.current_session_data[i]) > self.max_points:
                    self.current_session_data[i].pop(0)
                
                self.active_curves[i].setData(self.current_session_data[i])

            # 2. 하단 에러 그래프 업데이트
            if avg_error is not None:
                self.error_history.append(avg_error)
                if len(self.error_history) > self.max_points:
                    self.error_history.pop(0)
                
                self.error_curve.setData(self.error_history)
                self.thresh_line.setValue(current_threshold)
                
                if avg_error > current_threshold:
                    self.status_plot.getViewBox().setBackgroundColor((100, 0, 0, 40))
                else:
                    self.status_plot.getViewBox().setBackgroundColor((18, 18, 18, 255))
                    
        except Exception as e:
            print(f"Update Error: {e}")