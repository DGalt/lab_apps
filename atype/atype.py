from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
import neurphys.read_pv as rpv
import pyqtgraph as pg
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import itertools
from collections import OrderedDict


class ATypeAnalysis(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        desktop = QtWidgets.QDesktopWidget()
        width = desktop.screenGeometry().width()
        ratio = width / 1920
        self.resize(1400*ratio, 800*ratio)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,
                                       QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        layout = QtWidgets.QHBoxLayout(self)
        self.parent_dir = None
        self.df = None
        self.i_vals = []
        self.g_vals = []

        left_col = QtWidgets.QVBoxLayout()

        # ek
        self.ek_layout = QtWidgets.QHBoxLayout()
        self.ek_label = QtWidgets.QLabel('Ek (mV): ')
        self.ek_val = QtWidgets.QLineEdit('-108')
        self.ek_val.setSizePolicy(sizePolicy)
        self.ek_val.setFixedWidth(100)
        self.ek_layout.addWidget(self.ek_label)
        self.ek_layout.addWidget(self.ek_val)

        # baseline sweep
        self.bsl_sweep_layout = QtWidgets.QHBoxLayout()
        self.bsl_sweep_label = QtWidgets.QLabel('Baseline sweep: ')
        self.bsl_sweep_val = QtWidgets.QLineEdit('Sweep0006')
        self.bsl_sweep_val.setSizePolicy(sizePolicy)
        self.bsl_sweep_val.setFixedWidth(100)
        self.bsl_sweep_layout.addWidget(self.bsl_sweep_label)
        self.bsl_sweep_layout.addWidget(self.bsl_sweep_val)

        # holding
        self.holding_layout = QtWidgets.QHBoxLayout()
        self.holding_label = QtWidgets.QLabel('Holding (mV): ')
        self.holding_val = QtWidgets.QLineEdit('-80')
        self.holding_val.setSizePolicy(sizePolicy)
        self.holding_val.setFixedWidth(100)
        self.holding_layout.addWidget(self.holding_label)
        self.holding_layout.addWidget(self.holding_val)

        # start
        self.start_layout = QtWidgets.QHBoxLayout()
        self.start_label = QtWidgets.QLabel('Step start (s): ')
        self.start_val = QtWidgets.QLineEdit('1.5')
        self.start_val.setSizePolicy(sizePolicy)
        self.start_val.setFixedWidth(100)
        self.start_layout.addWidget(self.start_label)
        self.start_layout.addWidget(self.start_val)

        # offset
        self.offset_layout = QtWidgets.QHBoxLayout()
        self.offset_label = QtWidgets.QLabel('Start offset (s): ')
        self.offset_val = QtWidgets.QLineEdit('0.01')
        self.offset_val.setSizePolicy(sizePolicy)
        self.offset_val.setFixedWidth(100)
        self.offset_layout.addWidget(self.offset_label)
        self.offset_layout.addWidget(self.offset_val)

        # first step
        self.first_step_layout = QtWidgets.QHBoxLayout()
        self.first_step_label = QtWidgets.QLabel('First step (mV): ')
        self.first_step_val = QtWidgets.QLineEdit('-10')
        self.first_step_val.setSizePolicy(sizePolicy)
        self.first_step_val.setFixedWidth(100)
        self.first_step_layout.addWidget(self.first_step_label)
        self.first_step_layout.addWidget(self.first_step_val)

        # step delta
        self.delta_layout = QtWidgets.QHBoxLayout()
        self.delta_label = QtWidgets.QLabel('Step delta (mV): ')
        self.delta_val = QtWidgets.QLineEdit('10')
        self.delta_val.setSizePolicy(sizePolicy)
        self.delta_val.setFixedWidth(100)
        self.delta_layout.addWidget(self.delta_label)
        self.delta_layout.addWidget(self.delta_val)

        # number of steps
        self.num_steps_layout = QtWidgets.QHBoxLayout()
        self.num_steps_label = QtWidgets.QLabel('Number of steps: ')
        self.num_steps_val = QtWidgets.QLineEdit('5')
        self.num_steps_val.setSizePolicy(sizePolicy)
        self.num_steps_val.setFixedWidth(100)
        self.num_steps_layout.addWidget(self.num_steps_label)
        self.num_steps_layout.addWidget(self.num_steps_val)

        # fit stop
        self.stop_layout = QtWidgets.QHBoxLayout()
        self.stop_label = QtWidgets.QLabel('Fit stop (s): ')
        self.stop_val = QtWidgets.QLineEdit('1.99')
        self.stop_val.setSizePolicy(sizePolicy)
        self.stop_val.setFixedWidth(100)
        self.stop_layout.addWidget(self.stop_label)
        self.stop_layout.addWidget(self.stop_val)

        topSpacer = QtWidgets.QSpacerItem(20, 40,
                                          QtGui.QSizePolicy.Minimum,
                                          QtGui.QSizePolicy.Maximum)
        bottomSpacer = QtWidgets.QSpacerItem(20, 40,
                                             QtGui.QSizePolicy.Minimum,
                                             QtGui.QSizePolicy.Expanding)

        self.run_button = QtWidgets.QPushButton("Run new analysis")
        self.run_button.clicked.connect(self.run_new_analysis)
        self.rerun_button = QtWidgets.QPushButton("Re-run analysis")
        self.rerun_button.clicked.connect(self.run_analysis)
        self.copy_button = QtWidgets.QPushButton("Copy output")
        self.copy_button.clicked.connect(self.copy_output)

        left_col.addItem(topSpacer)
        left_col.addLayout(self.ek_layout)
        left_col.addLayout(self.holding_layout)
        left_col.addLayout(self.start_layout)
        left_col.addLayout(self.offset_layout)
        left_col.addLayout(self.first_step_layout)
        left_col.addLayout(self.delta_layout)
        left_col.addLayout(self.num_steps_layout)
        left_col.addLayout(self.stop_layout)
        left_col.addWidget(self.run_button)
        left_col.addWidget(self.rerun_button)
        left_col.addWidget(self.copy_button)
        left_col.addItem(bottomSpacer)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.table = QtWidgets.QTableWidget()
        self.table.setFixedWidth(300)
        self.headers = ['Steps', 'I (pA)', 'g', 'tau (ms)']
        self.table.setColumnCount(len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)
        header = self.table.horizontalHeader()
        [header.setResizeMode(i, QtWidgets.QHeaderView.Stretch) for i in range(len(self.headers))]

        layout.addLayout(left_col)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.table)

    def load_data(self):
        folder = QtWidgets.QFileDialog().getExistingDirectory(self,
                                                              "Select data folder",
                                                              self.parent_dir)

        self.parent_dir = os.path.dirname(folder)
        self.df = rpv.import_folder(folder)['voltage recording']

        if self.df is None:
            self.gen_error_mbox('Folder does not contain voltage recording data')

    def initialize_parameters(self):
        self.bsl_sweep = self.bsl_sweep_val
        try:
            self.ek = float(self.ek_val.text())
            self.holding = float(self.holding_val.text())
            self.start = float(self.start_val.text())
            self.offset = float(self.offset_val.text())
            self.first_step = float(self.first_step_val.text())
            self.delta = float(self.delta_val.text())
            self.num_steps = int(self.num_steps_val.text())
            self.stop = float(self.stop_val.text())
            self.steps = [self.holding + self.first_step +
                          self.delta * i for i in range(self.num_steps)]
            return True
        except TypeError:
            message = """A parameter value is invalid. Check that all parameters
            besides baseline sweep are numeric only"""
            self.gen_error_mbox(message)
            return False

    def analyze_peaks(self):
        start = self.start + self.offset
        stop = start + 0.5
        sub = self.df.loc[self.bsl_sweep.text()]
        mask = (sub.time >= start) & (sub.time <= stop)
        bsl = sub.loc[mask, 'primary'].mean()

        sweeps = self.df.index.levels[0]
        peaks = []
        peak_times = []
        plot = self.plot_widget.addPlot(0, 0)
        for step, sweep in zip(self.steps, sweeps):
            sub = self.df.loc[sweep]
            mask = (sub.time >= start) & (sub.time <= stop)
            peak_ix = sub.loc[mask, 'primary'].idxmax()
            peak = sub.loc[peak_ix, 'primary']
            peak_time = sub.loc[peak_ix, 'time']
            i = peak - bsl
            g = i / (step - self.ek)

            peak_times.append(peak_time)
            peaks.append(peak)
            self.i_vals.append(i)
            self.g_vals.append(g)
            plot.plot(sub.time, sub.primary, pen='b')

        plot.plot(peak_times, peaks, pen=None, symbol='o',
                  symbolPen='r', symbolBrush='r')

    def fit_transient(self):
        sweep = self.df.loc['Sweep0001'].copy()
        start = self.start + self.offset
        mask = (sweep.time >= start) & (sweep.time <= self.stop)
        peak_ix = sweep.loc[mask, 'primary'].idxmax()
        peak_time = sweep.loc[peak_ix, 'time']

        mask = (sweep.time >= peak_time) & (sweep.time <= self.stop)
        sub = sweep.loc[mask]

        x_zeroed = sub.time.values - sub.time.iloc[0]
        # guess = np.array([1, 1, 1, 1, 0])
        guess = np.array([1, 1, 0])

        # def exp_decay(x, a, b, c, d, e):
            # return a*np.exp(-x/b) + c*np.exp(-x/d) + e
        def exp_decay(x, a, b, c):
            return a*np.exp(-x/b) + c

        popt, pcov = curve_fit(exp_decay, x_zeroed*1e3, sub.primary, guess)

        # amp1 = popt[0]
        # tau1 = popt[1]
        # amp2 = popt[2]
        # tau2 = popt[3]

        # tau = ((tau1*amp1)+(tau2*amp2))/(amp1+amp2)
        self.tau = popt[1]
        fit = exp_decay(x_zeroed*1e3, *popt)
        plot = self.plot_widget.addPlot(1, 0)
        plot.plot(sweep.time, sweep.primary, pen='b')
        plot.plot(sub.time.values, fit, pen='r')

    def write_table(self):
        self.table.setRowCount(self.num_steps)
        for i in range(self.num_steps):
            vals = [self.steps[i], self.i_vals[i], self.g_vals[i]]
            for j, val in enumerate(vals):
                item = QtWidgets.QTableWidgetItem('%0.3f' % val)
                self.table.setItem(i, j, item)

        item = QtWidgets.QTableWidgetItem('%0.3f' % self.tau)
        self.table.setItem(0, 3, item)

    def run_analysis(self):
        initialized = self.initialize_parameters()
        if initialized and self.df is not None:
            self.plot_widget.clear()
            # self.table.clear()
            self.analyze_peaks()
            self.fit_transient()
            self.write_table()

    def run_new_analysis(self):
        self.load_data()
        self.run_analysis()

    def copy_output(self):
        if any(self.i_vals):
            step_df = pd.DataFrame({'Steps': self.steps,
                                    'I (pA)': self.i_vals,
                                    'g': self.g_vals})
            step_df['tau (ms)'] = self.tau
            step_df.loc[1:, 'tau (ms)'] = np.nan
            step_df[['Steps', 'I (pA)', 'g', 'tau (ms)']].to_clipboard(index=False)

    def gen_error_mbox(self, message):
        msg = QtWidgets.QMessageBox()
        msg.Critical
        msg.setText(message)
        msg.setWindowTitle('Error')
        msg.exec_()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = ATypeAnalysis()
    ex.show()
    sys.exit(app.exec_())


