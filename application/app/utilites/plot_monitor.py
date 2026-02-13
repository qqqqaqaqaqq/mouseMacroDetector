import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import numpy as np

class RealTimeMonitor:
    def __init__(self, features, threshold):
        self.app = QApplication.instance() or QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="Extreme Values - Multi Session Monitor")
        self.win.resize(1600, 1000)
        self.win.setBackground('#121212')

        self.original_features = features 
        self.num_features = len(self.original_features) 
        
        self.stat_types = ['skewed', 'entropy_gap', 'roughness']
        self.all_stat_names = [f"{f}_{s}" for s in self.stat_types for f in self.original_features]

        self.plots = []
        # 핵심 변경: 현재 세션의 커브들만 따로 관리
        self.current_active_curves = [] 
        self.colors = ['#00BFFF', '#FFD700', '#ADFF2F', '#FF69B4', '#FFA500', '#00FA9A', '#FF4500']
        self.current_color_idx = -1
        self.x_range = np.linspace(-10, 10, 300)

        self._setup_layouts(threshold)
        self.start_new_session()

    def _setup_layouts(self, threshold):
        cols_per_row = min(self.num_features, 8) 
        sections = [
            ("SECTION 1 : skewed", "#00BFFF"),
            ("SECTION 2 : entropy_gap", "#ADFF2F"),
            ("SECTION 3 : roughness", "#FF69B4")
        ]

        for s_idx, (title, color) in enumerate(sections):
            self.win.addLabel(f"<b><span style='color: {color}; font-size: 11pt;'>[ {title} ]</span></b>", colspan=cols_per_row)
            self.win.nextRow()
            for f_idx in range(self.num_features):
                total_idx = (s_idx * self.num_features) + f_idx
                p = self.win.addPlot(title=f"<span style='color: #DDDDDD; font-size: 8pt;'>{self.all_stat_names[total_idx]}</span>")
                p.setMinimumWidth(180) 
                self._apply_plot_style(p)
                self.plots.append(p)
                if (f_idx + 1) % cols_per_row == 0: self.win.nextRow()
            if self.num_features % cols_per_row != 0: self.win.nextRow()

        self.status_plot = self.win.addPlot(title="<b>Reconstruction Error Trace</b>", colspan=cols_per_row)
        self.status_plot.setFixedHeight(180)
        self.status_plot.setXRange(0, 300)
        self.status_plot.showGrid(x=True, y=True, alpha=0.2)
        self.error_history = []
        self.thresh_line = pg.InfiniteLine(pos=threshold, angle=0, pen=pg.mkPen('#FF4444', width=2, style=Qt.PenStyle.DashLine))
        self.status_plot.addItem(self.thresh_line)

    def _apply_plot_style(self, p):
        p.showGrid(x=True, y=True, alpha=0.15)
        p.setXRange(-8, 8) 
        p.enableAutoRange(axis='y', enable=True)
        p.getAxis('left').setStyle(tickFont=pg.Qt.QtGui.QFont('Arial', 7))
        p.getAxis('bottom').setStyle(tickFont=pg.Qt.QtGui.QFont('Arial', 7))

    def start_new_session(self):
        """새 세션 시작 시 기존 커브는 유지하고 새로운 커브 세트를 생성"""
        self.current_color_idx = (self.current_color_idx + 1) % len(self.colors)
        color = self.colors[self.current_color_idx]
        
        # 이전 세션의 커브들이 흐릿하게 보이게 하고 싶다면 여기서 투명도 조절 가능 (선택 사항)
        # 예: for c in self.current_active_curves: c.setAlpha(0.3, False)

        # 1. 각 Plot마다 새 세션을 위한 새로운 Curve 객체를 추가함
        self.current_active_curves = [p.plot(pen=pg.mkPen(color, width=2)) for p in self.plots]
        
        # 2. 에러 차트에도 새 세션용 선 추가
        self.error_history = []
        self.current_error_curve = self.status_plot.plot(pen=pg.mkPen(color, width=2))

    def update_view(self, current_features, avg_error, current_threshold):
        if current_features is None: return
        
        # "NEW_SESSION" 문자열이 들어오면 새로운 선 세트를 생성함
        if isinstance(current_features, str) and "NEW_SESSION" in current_features:
            self.start_new_session()
            return

        try:
            # Sigma 참조 (항상 Std 섹션 데이터 사용)
            stds_for_shape = current_features[self.num_features : 2 * self.num_features]

            for i in range(len(self.plots)):
                mu = current_features[i]
                feature_idx = i % self.num_features
                sigma = np.clip(stds_for_shape[feature_idx], 0.15, 4.0)
                
                pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((self.x_range - mu) / sigma)**2)
                
                # [핵심] 현재 활성화된 세션의 커브만 업데이트 (나머지는 그대로 멈춤)
                self.current_active_curves[i].setData(self.x_range, pdf)
                
                # Y축 자동 조절
                max_val = max(pdf)
                self.plots[i].setYRange(0, max_val * 1.1, padding=0)

            if avg_error is not None:
                self.error_history.append(avg_error)
                if len(self.error_history) > 300: self.error_history.pop(0)
                self.current_error_curve.setData(self.error_history)
                self.thresh_line.setValue(current_threshold)
                
                bg_color = (180, 0, 0, 50) if avg_error > current_threshold else (18, 18, 18, 255)
                self.status_plot.getViewBox().setBackgroundColor(bg_color)
        except Exception:
            pass