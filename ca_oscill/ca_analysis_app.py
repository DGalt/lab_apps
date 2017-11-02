from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
import neurphys.read_pv as rpv
import neurphys.utilities as util
import neurphys.pacemaking as pace
import pyqtgraph as pg
import numpy as np
import pandas as pd
import itertools
from collections import OrderedDict


class CaAnalysis(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        desktop = QtWidgets.QDesktopWidget()
        width = desktop.screenGeometry().width()
        ratio = width / 1920
        self.resize(1400*ratio, 800*ratio)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        self.vm = None
        self.ls = None
        self.fmax_vm = None
        self.fmax_ls = None
        self.kd = 120
        self.background = 0
        self.dye_rf = 22
        self.obs_rf = 18
        self.smooth_by = 9
        self.prof = 'Prof 2'
        self.prof_t = 'Prof 2 Time'
        self.mph = None
        self.mpd = None
        self.parent_dir = ''
        self.output_df = None

        self.layout = QtWidgets.QHBoxLayout(self)

        self.leftCol = QtWidgets.QVBoxLayout()
        #Kd
        self.kdLayout = QtWidgets.QHBoxLayout()
        self.kdLabel = QtWidgets.QLabel("Kd: ")
        self.kdVal = QtWidgets.QLineEdit("120")
        self.kdVal.setSizePolicy(sizePolicy)
        self.kdVal.setFixedWidth(100)
        self.kdLayout.addWidget(self.kdLabel)
        self.kdLayout.addWidget(self.kdVal)
        #Background
        self.bkgLayout = QtWidgets.QHBoxLayout()
        self.bkgLabel = QtWidgets.QLabel("Background: ")
        self.bkgVal = QtWidgets.QLineEdit("0")
        self.bkgVal.setSizePolicy(sizePolicy)
        self.bkgVal.setFixedWidth(100)
        self.bkgLayout.addWidget(self.bkgLabel)
        self.bkgLayout.addWidget(self.bkgVal)
        #Dye Rf
        self.drfLayout = QtWidgets.QHBoxLayout()
        self.drfLabel = QtGui.QLabel("Dye Rf: ")
        self.drfVal = QtGui.QLineEdit("22")
        self.drfVal.setSizePolicy(sizePolicy)
        self.drfVal.setFixedWidth(100)
        self.drfLayout.addWidget(self.drfLabel)
        self.drfLayout.addWidget(self.drfVal)
        #Observed Rf
        self.orfLayout = QtWidgets.QHBoxLayout()
        self.orfLabel = QtWidgets.QLabel("Observed Rf: ")
        self.orfVal = QtWidgets.QLineEdit("18")
        self.orfVal.setSizePolicy(sizePolicy)
        self.orfVal.setFixedWidth(100)
        self.orfLayout.addWidget(self.orfLabel)
        self.orfLayout.addWidget(self.orfVal)
        #smoothing
        self.smthLayout = QtWidgets.QHBoxLayout()
        self.smthLabel = QtWidgets.QLabel("Smooth by: ")
        self.smthVal = QtWidgets.QLineEdit("9")
        self.smthVal.setSizePolicy(sizePolicy)
        self.smthVal.setFixedWidth(100)
        self.smthLayout.addWidget(self.smthLabel)
        self.smthLayout.addWidget(self.smthVal)
        #prof label
        self.profLayout = QtGui.QHBoxLayout()
        self.profLabel = QtGui.QLabel("Linescan Profile: ")
        self.profVal = QtGui.QLineEdit("Prof 2")
        self.profVal.setSizePolicy(sizePolicy)
        self.profVal.setFixedWidth(100)
        self.profLayout.addWidget(self.profLabel)
        self.profLayout.addWidget(self.profVal)
        ### detect_peaks parameters
        self.autoCheckbox = QtWidgets.QCheckBox()
        self.autoCheckbox.setText("Automatically determine vals:")
        self.autoCheckbox.setChecked(True)
        self.autoCheckbox.stateChanged.connect(self.change_state)
        #mph
        self.mphLayout = QtWidgets.QHBoxLayout()
        self.mphLabel = QtWidgets.QLabel("Min. Peak Height:")
        self.mphVal = QtWidgets.QLineEdit()
        self.mphVal.setSizePolicy(sizePolicy)
        self.mphVal.setFixedWidth(100)
        self.mphVal.setEnabled(False)
        self.mphLayout.addWidget(self.mphLabel)
        self.mphLayout.addWidget(self.mphVal)
        #mpd
        self.mpdLayout = QtWidgets.QHBoxLayout()
        self.mpdLabel = QtWidgets.QLabel("Min. Peak Dist:")
        self.mpdVal = QtWidgets.QLineEdit()
        self.mpdVal.setSizePolicy(sizePolicy)
        self.mpdVal.setFixedWidth(100)
        self.mpdVal.setEnabled(False)
        self.mpdLayout.addWidget(self.mpdLabel)
        self.mpdLayout.addWidget(self.mpdVal)

        topSpacer = QtWidgets.QSpacerItem(20, 40,
                                      QtGui.QSizePolicy.Minimum,
                                      QtGui.QSizePolicy.Maximum)
        bottomSpacer = QtWidgets.QSpacerItem(20, 40,
                                        QtGui.QSizePolicy.Minimum,
                                        QtGui.QSizePolicy.Expanding)

        self.runButton = QtWidgets.QPushButton("Run new analysis")
        self.runButton.clicked.connect(self.run_new_analysis)
        self.rerunButton = QtWidgets.QPushButton("Re-run analysis")
        self.rerunButton.clicked.connect(self.run_analysis)
        self.copy_button = QtWidgets.QPushButton("Copy output")
        self.copy_button.clicked.connect(self.copy_output)

        self.leftCol.addItem(topSpacer)
        self.leftCol.addLayout(self.kdLayout)
        self.leftCol.addLayout(self.bkgLayout)
        self.leftCol.addLayout(self.drfLayout)
        self.leftCol.addLayout(self.orfLayout)
        self.leftCol.addLayout(self.smthLayout)
        self.leftCol.addLayout(self.profLayout)
        self.leftCol.addWidget(self.autoCheckbox)
        self.leftCol.addLayout(self.mphLayout)
        self.leftCol.addLayout(self.mpdLayout)
        self.leftCol.addWidget(self.runButton)
        self.leftCol.addWidget(self.rerunButton)
        self.leftCol.addWidget(self.copy_button)
        self.leftCol.addItem(bottomSpacer)

        self.plotWidget = pg.GraphicsLayoutWidget(self)
        self.table = QtWidgets.QTableWidget()
        self.headers = ['Average Area', 'Total Area', 'Peak', 'Baseline', 'Average']
        self.table.setColumnCount(len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)
        header = self.table.horizontalHeader()
        [header.setResizeMode(i, QtWidgets.QHeaderView.Stretch) for i in range(len(self.headers))]

        self.tab_widget = QtWidgets.QTabWidget(self)
        self.tab_widget.addTab(self.plotWidget, 'Plot')
        self.tab_widget.addTab(self.table, 'Output')

        self.layout.addLayout(self.leftCol)
        self.layout.addWidget(self.tab_widget)

    def change_state(self):
        if self.autoCheckbox.isChecked():
            self.mphVal.setText('')
            self.mphVal.setEnabled(False)
            self.mpdVal.setText('')
            self.mpdVal.setEnabled(False)
        else:
            self.mphVal.setEnabled(True)
            self.mpdVal.setEnabled(True)

    def load_data(self):
        folder = QtWidgets.QFileDialog().getExistingDirectory(self,
                                                          "Select folder containing oscillation data",
                                                          self.parent_dir)
        self.parent_dir = os.path.dirname(folder)

        data_dict = rpv.import_folder(folder)
        if data_dict['voltage recording'] is None or data_dict['linescan'] is None:
            QtWidgets.QMessageBox.about(self, "Error", "Folder does not contain necessary data")
            self.vm = None
            self.ls = None
            return
        else:
            self.vm = data_dict['voltage recording']
            self.ls = data_dict['linescan']

        folder = QtWidgets.QFileDialog().getExistingDirectory(self,
                                                          "Select folder containing fmax data",
                                                          self.parent_dir)
        data_dict = rpv.import_folder(folder)
        if data_dict['voltage recording'] is None or data_dict['linescan'] is None:
            QtWidgets.QMessageBox.about(self, "Error", "Folder does not contain necessary data")
            self.fmax_vm = None
            self.fmax_ls = None
            return
        else:
            self.fmax_vm = data_dict['voltage recording']
            self.fmax_ls = data_dict['linescan']

    def calc_fmax(self):
        self.fmax_ls['bkg_sub'] = self.fmax_ls[self.prof] - self.background
        sampling = 1 / (self.fmax_vm.time[1] - self.fmax_vm.time[0])
        ix = np.nanargmax(np.gradient(util.simple_smoothing(self.fmax_vm.secondary.values, 200)))
        end = self.fmax_vm.time.iloc[int(ix-sampling*0.05)]
        start = end - 0.5

        mask = (self.fmax_ls[self.prof_t] >= start) & (self.fmax_ls[self.prof_t] <= end)
        f0 = self.fmax_ls.loc[mask, 'bkg_sub'].mean()
        fmax = f0 * (self.dye_rf / self.obs_rf)

        top = self.plotWidget.addPlot(0, 0)
        top.plot(self.fmax_vm.time, self.fmax_vm.primary, pen='b')
        middle = self.plotWidget.addPlot(1, 0)
        middle.plot(self.fmax_vm.time, self.fmax_vm.secondary, pen='b')
        middle.setXLink(top)
        bottom = self.plotWidget.addPlot(2, 0)
        bottom.plot(self.fmax_ls[self.prof_t], self.fmax_ls['bkg_sub'], pen='b')
        bottom.plot(self.fmax_ls[self.prof_t][mask], self.fmax_ls['bkg_sub'][mask], pen='r')
        bottom.setXLink(top)

        return fmax

    def calc_ca(self, fmax):
        self.ls['bkg_sub'] = self.ls[self.prof] - self.background
        self.ls['ca_conc'] = (self.kd * ((1-self.ls['bkg_sub'] / fmax) /
                                         (self.ls['Prof 2'] / fmax - (1/self.dye_rf))))

        ls_sampling = 1 / (self.ls[self.prof_t][1] - self.ls[self.prof_t][0])
        self.ls['ca_smth'] = util.simple_smoothing(self.ls['ca_conc'].values, self.smooth_by)

        if self.autoCheckbox.isChecked():
            self.mph = self.ls['ca_smth'].max()/2
            self.mpd = ls_sampling*0.25

            self.mphVal.setText(str(self.mph))
            self.mpdVal.setText(str(self.mpd))

        ixs = pace.detect_peaks(self.ls['ca_smth'], mph=self.mph, mpd=self.mpd)

        top = self.plotWidget.addPlot(0, 1)
        top.plot(self.vm.time, self.vm.primary, pen='b')
        middle = self.plotWidget.addPlot(1, 1)
        middle.plot(self.ls[self.prof_t], self.ls['ca_smth'], pen='b')
        middle.plot(self.ls[self.prof_t][ixs], self.ls['ca_smth'][ixs],
                    pen=None, symbolBrush=pg.mkColor('r'),
                    symbolPen=pg.mkPen('r'), symbol="d")
        middle.setXLink(top)

        colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        bottom = self.plotWidget.addPlot(2, 1)
        output_dict = OrderedDict([[header, []] for header in self.headers])
        for i, ix in enumerate(ixs[1:-1]):
            tr_ix1 = self.ls['ca_smth'].iloc[ixs[i]:ix].idxmin()[-1]
            tr_ix2 = self.ls['ca_smth'].iloc[ix:ixs[i+2]].idxmin()[-1]
            sub = self.ls.iloc[tr_ix1:tr_ix2].copy()

            avg_area = ((np.trapz(sub['ca_smth'], sub[self.prof_t])) /
                   (sub['Prof 2 Time'][-1] - sub['Prof 2 Time'][0]))
            total_area = np.trapz(sub['ca_smth'])
            peak = sub['ca_smth'].max()
            peak_ix = sub['ca_smth'].idxmax()
            bsl1 = sub.loc[:peak_ix, 'ca_smth'].iloc[:100].mean()
            bsl2 = sub.loc[peak_ix:, 'ca_smth'].iloc[-100:].mean()
            baseline = (bsl1+bsl2)/2 
            avg = sub['ca_smth'].mean()

            output_dict['Average Area'].append(avg_area)
            output_dict['Total Area'].append(total_area)
            output_dict['Peak'].append(peak)
            output_dict['Baseline'].append(baseline)
            output_dict['Average'].append(avg)
            self.table.insertRow(i)
            for j, metric in enumerate([avg_area, total_area, peak, baseline, avg]):
                item = QtWidgets.QTableWidgetItem("%0.3f" % metric)
                self.table.setItem(i, j, item)

            bottom.plot(sub[self.prof_t], sub['ca_smth'], pen=next(colors))
        bottom.setXLink(top)
        self.output_df = pd.DataFrame(output_dict)

    def run_analysis(self):
        self.plotWidget.clear()
        self.output_df = None
        self.table.setRowCount(0)
        self.kd = float(self.kdVal.text())
        self.background = float(self.bkgVal.text())
        self.dye_rf = float(self.drfVal.text())
        self.obs_rf = float(self.orfVal.text())
        self.smooth_by = int(self.smthVal.text())
        self.prof = self.profVal.text()
        self.prof_t = self.profVal.text() + ' Time'

        if self.vm is None or self.fmax_vm is None:
            return

        if self.autoCheckbox.isChecked():
            self.mph = None
            self.mpd = None
        else:
            try:
                self.mph = float(self.mphVal.text())
                self.mpd = float(self.mpdVal.text())
            except ValueError:
                QtWidgets.QMessageBox.about(self, "Error",
                                            "Value for Min. Peak Height or Min Peak Dist is invalid")
                return

        fmax = self.calc_fmax()
        self.calc_ca(fmax)

    def run_new_analysis(self):
        self.load_data()
        self.run_analysis()

    def copy_output(self):
        if self.output_df is not None:
            self.output_df.to_clipboard(index=False)
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = CaAnalysis()
    ex.show()
    sys.exit(app.exec_())
