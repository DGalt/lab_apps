from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
import neurphys.read_pv as rpv
import neurphys.utilities as util
import pyqtgraph as pg
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapz
import itertools
from collections import OrderedDict

class bAPAnalysis(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        desktop = QtWidgets.QDesktopWidget()
        # width = desktop.screenGeometry().width()
        dpi = desktop.logicalDpiX()
        # ratio = width / 1920
        self.ratio = dpi / 96
        self.resize(1400*self.ratio, 800*self.ratio)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.df_list = []
        self.avg_df = None
        self.r_prof = 'Prof 1'
        self.g_prof = 'Prof 2'

        layout = QtWidgets.QHBoxLayout(self)
        left_col = QtWidgets.QVBoxLayout()

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setMaximumWidth(220*self.ratio)

        label_width = 120 * self.ratio

        self.stim_layout = QtWidgets.QHBoxLayout()
        self.stim_label = QtWidgets.QLabel('Stim start: ')
        self.stim_label.setFixedWidth(label_width)
        self.stim_val = QtWidgets.QLineEdit('0.5')
        self.stim_val.setFixedWidth(100*self.ratio)  
        self.stim_layout.addWidget(self.stim_label)
        self.stim_layout.addWidget(self.stim_val)

        self.g0_start_layout = QtWidgets.QHBoxLayout()
        self.g0_start_label = QtWidgets.QLabel('g0 start: ')
        self.g0_start_label.setFixedWidth(label_width)
        self.g0_start_val = QtWidgets.QLineEdit('0.35')
        self.g0_start_val.setFixedWidth(100*self.ratio)
        self.g0_start_layout.addWidget(self.g0_start_label)
        self.g0_start_layout.addWidget(self.g0_start_val)

        self.g0_stop_layout = QtWidgets.QHBoxLayout()
        self.g0_stop_label = QtWidgets.QLabel('g0 start: ')
        self.g0_stop_label.setFixedWidth(label_width)
        self.g0_stop_val = QtWidgets.QLineEdit('0.48')
        self.g0_stop_val.setFixedWidth(100*self.ratio)
        self.g0_stop_layout.addWidget(self.g0_stop_label)
        self.g0_stop_layout.addWidget(self.g0_stop_val)

        self.tb4_stop_layout = QtWidgets.QHBoxLayout()
        self.tb4_stop_label = QtWidgets.QLabel('Delta t before peak: ')
        self.tb4_stop_label.setFixedWidth(label_width)
        self.tb4_stop_val = QtWidgets.QLineEdit('0.1')
        self.tb4_stop_val.setFixedWidth(100*self.ratio)
        self.tb4_stop_layout.addWidget(self.tb4_stop_label)
        self.tb4_stop_layout.addWidget(self.tb4_stop_val)

        self.fit_stop_layout = QtWidgets.QHBoxLayout()
        self.fit_stop_label = QtWidgets.QLabel('Fit stop: ')
        self.fit_stop_label.setFixedWidth(label_width)
        self.fit_stop_val = QtWidgets.QLineEdit('')
        self.fit_stop_val.setFixedWidth(100*self.ratio)
        self.fit_stop_layout.addWidget(self.fit_stop_label)
        self.fit_stop_layout.addWidget(self.fit_stop_val)

        self.load_btn = QtWidgets.QPushButton('Load folder(s)')
        self.load_btn.clicked.connect(self.load_folders)
        self.clear_btn = QtWidgets.QPushButton('Clear folder(s)')
        self.clear_btn.clicked.connect(self.clear_folders)
        self.run_btn = QtWidgets.QPushButton('Run analysis')
        self.run_btn.clicked.connect(self.run_analysis)

        left_col.addWidget(self.list_widget)
        left_col.addLayout(self.stim_layout)
        left_col.addLayout(self.g0_start_layout)
        left_col.addLayout(self.g0_stop_layout)
        left_col.addLayout(self.tb4_stop_layout)
        left_col.addLayout(self.fit_stop_layout)
        left_col.addWidget(self.load_btn)
        left_col.addWidget(self.clear_btn)
        left_col.addWidget(self.run_btn)

        self.plot_widget = pg.GraphicsLayoutWidget(self)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(1)
        self.table.setRowCount(7)
        self.table.setHorizontalHeaderLabels(['Values'])
        self.table.setVerticalHeaderLabels(['Peak', 'Total Area', 'Average Area', 'a', 'b', 'c', 'd'])
        self.table.horizontalHeader().setResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.setFixedWidth(200*self.ratio)

        layout.addLayout(left_col)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.table)

    def load_folders(self):
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.DirectoryOnly)
        dialog.setViewMode(QtWidgets.QFileDialog.Detail)
        dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)

        for view in dialog.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
            if isinstance(view.model(), QtWidgets.QFileSystemModel):
                view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        if dialog.exec_():
            folders = dialog.selectedFiles()
            dir_path = dialog.directory().path()
            if dir_path in folders:
                folders.remove(dir_path)

            for folder in folders:
                df = rpv.import_folder(folder)['linescan']
                if df is None:
                    self.gen_error_mbox('Folder %s does not contain necessary data' % folder)
                    folders.remove(folder)
                else:
                    self.df_list.append(df)

            if any(folders):
                self.update_list_widget(folders)

    def update_list_widget(self, folders):
        for full_path in folders:
            folder = os.path.split(full_path)[-1] 
            item = QtWidgets.QListWidgetItem()
            item.setText(folder)
            item.setToolTip(full_path)
            self.list_widget.addItem(item)

    def get_avg_df(self):
        if len(self.df_list) == 1:
            self.avg_df = self.df_list[0].copy()
            self.avg_df['gr'] = self.get_gr_col(self.avg_df)

        elif len(self.df_list) > 1:
            norm_df_list = []
            for df in self.df_list:
                df['gr'] = self.get_gr_col(df)
                norm_df_list.append(df)

            self.avg_df = pd.concat(norm_df_list).groupby(level=1).mean()

    def get_gr_col(self, df):
        df['gnorm'] = df['Prof 2'] / df['Prof 1']
        try:
            start = float(self.g0_start_val.text())
        except ValueError:
            self.gen_error_mbox('g0 start value must be a number >= 0')

        try:
            stop = float(self.g0_stop_val.text())
        except ValueError:
            self.gen_error_mbox('g0 stop value must be a number >= 0')

        mask = (df['Prof 2 Time'] >= start) & (df['Prof 2 Time'] <= stop)
        g0 = df.loc[mask, 'gnorm'].mean()

        return util.simple_smoothing((df['gnorm']-g0).values, 9)

    def clear_folders(self):
        self.list_widget.clear()
        self.df_list = []
        self.avg_df = None
        self.data_dict = None

    def gen_subset(self):
        try:
            start = float(self.stim_val.text())
        except ValueError:
            self.gen_error_mbox('Stim start must be a number >= 0')
            return None

        if self.fit_stop_val.text() == '':
            stop = self.avg_df['Prof 2 Time'].iloc[-1]
        else:
            try:
                stop = float(self.fit_stop_val.text())
            except ValueError:
                self.gen_error_mbox('Fit stop must be a number >= 0')
                return None

        try:
            tb4peak = float(self.tb4_stop_val.text())
        except ValueError:
            self.gen_error_mbox('Time before peak must be a number >= 0')
            return None

        peak_ix = self.avg_df[(self.avg_df['Prof 2 Time'] >= start) &
                              (self.avg_df['Prof 2 Time'] <= stop)].gr.idxmax()

        if isinstance(peak_ix, tuple):
            peak_ix = peak_ix[-1]

        new_start = self.avg_df.loc[peak_ix, 'Prof 2 Time'] - tb4peak

        return self.avg_df[(self.avg_df['Prof 2 Time'] >= new_start) &
                           (self.avg_df['Prof 2 Time'] <= stop)]

    def gen_fit(self, subset):
        def eq(x, a, b, c, d):
            return a*(1-np.exp(-x*b))*(np.exp(-x*c)) + d

        guess = [1, 1e-3, 1, 0]
        x = subset['Prof 2 Time'] - subset['Prof 2 Time'].iloc[0]
        popt, pcov = curve_fit(eq, x, subset['gr'], p0=guess, maxfev=5000)

        fit = eq(x, *popt)

        return fit, popt

    def run_analysis(self):
        self.plot_widget.clear()
        self.clear_table()
        if len(self.df_list) > 0:
            self.get_avg_df()
        else:
            return
        subset = self.gen_subset()
        if subset is not None:
            fit, popt = self.gen_fit(subset)
            plot = self.plot_widget.addPlot()
            plot.plot(self.avg_df['Prof 2 Time'], self.avg_df['gr'], pen='b')
            x = subset['Prof 2 Time'].values
            y = subset['gr'].values
            plot.plot(x, y, pen='r')
            plot.plot(x, fit.values, pen='g')
            dx = subset['Prof 2 Time'].iloc[1] - subset['Prof 2 Time'].iloc[0]
            total_area = np.trapz(subset.gr)
            avg_area = np.trapz(subset.gr, dx=dx)
            self.data_dict = OrderedDict([('Peak', subset.gr.max()),
                                          ('Total Area', total_area),
                                          ('Average Area', avg_area),
                                          ('a', popt[0]),
                                          ('b', popt[1]),
                                          ('c', popt[2]),
                                          ('d', popt[3])])
            for i, key in enumerate(self.data_dict.keys()):
                item = QtWidgets.QTableWidgetItem('%0.4f' % self.data_dict[key])
                self.table.setItem(i, 0, item)

    def clear_table(self):
        for i in range(7):
            item = QtWidgets.QTableWidgetItem('')
            self.table.setItem(i, 0, item)

    def gen_error_mbox(self, message):
        msg = QtWidgets.QMessageBox()
        msg.Critical
        msg.setText(message)
        msg.setWindowTitle('Error')
        msg.exec_()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = bAPAnalysis()
    ex.show()
    sys.exit(app.exec_())
