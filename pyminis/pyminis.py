from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
import neurphys.read_abf as abf
import neurphys.read_pv as rpv
import neurphys.pacemaking as pace
import neurphys.utilities as util
import pyqtgraph as pg
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import time
import warnings
import logging
import traceback
warnings.filterwarnings("ignore")


class MiniAnalysis(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        desktop = QtWidgets.QDesktopWidget()
        dpi = desktop.logicalDpiX()
        #width = desktop.screenGeometry().width()
        self.setWindowTitle("PyMinis")
        self.ratio = dpi / 96
        self.resize(1400*self.ratio, 800*self.ratio)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.checked = None
        self.counter = 0
        self.data_col = 'primary'
        self.decay_fit = None
        self.decay_x = None
        self.detect_start = 0.02
        self.detect_stop = None
        self.detection_plot = None
        self.df = None
        self.end_fit = 0.3
        self.event_bsl_window = 40
        self.fit_a1 = None
        self.fit_a2 = None
        self.fit_a3 = None
        self.fit_c = None
        self.fit_plot = None
        self.fit_start_ix = None
        self.fit_tau1 = None
        self.fit_tau2 = None
        self.fit_tau3 = None
        self.fit_vals = None
        self.heights = None
        self.indexes = []
        self.item = None
        self.mpd = 0.01
        self.parent_dir = ''
        self.peak_time = None
        self.peak_time_delta = 0.02
        self.points_plot = None
        self.poly_subset = None
        self.poly_order = 1
        self.rise_fit = None
        self.rms_start = 0
        self.rms_stop = 0.1
        self.rms_multiple = 1
        self.sampling = None
        self.smth_by = 10
        self.stim_start = 0
        self.sub_trans = True
        self.sweep = None
        self.sweep_median = None
        self.tolerance = 20
        self.time = 0
        self.user_a1 = None
        self.user_a2 = None
        self.user_a3 = None
        self.user_c = None
        self.user_tau1 = None
        self.user_tau2 = None
        self.user_tau3 = None

        self.menubar = self.menuBar()
        self.setup_file_menu()
        self.setup_copy_menu()

        central_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(central_widget)

        # LEFT COLUMN
        self.left_col = QtWidgets.QVBoxLayout()
        self.size_policy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,
                                             QtGui.QSizePolicy.Fixed)
        self.size_policy.setHorizontalStretch(0)
        self.size_policy.setVerticalStretch(0)
        self.label_width = 160 * self.ratio
        self.edit_width = 45 * self.ratio
        self.hspacer = QtWidgets.QSpacerItem(10*self.ratio, 40
                                            , QtWidgets.QSizePolicy.Minimum
                                            , QtWidgets.QSizePolicy.Maximum)
        self.vspacer = QtWidgets.QSpacerItem(100, 0
                                            , QtWidgets.QSizePolicy.Minimum
                                            , QtWidgets.QSizePolicy.Expanding)

        # tree widget
        self.tree_widget = QtWidgets.QTreeWidget(self)
        self.tree_widget.headerItem().setText(0, '')
        self.tree_widget.itemChanged.connect(self.update_checked)
        self.tree_widget.setMaximumWidth(220*self.ratio)

        # tab widget
        self.tab_widget = QtGui.QTabWidget(self)
        self.tab_widget.setMinimumSize(QtCore.QSize(230*self.ratio,
                                                    400*self.ratio))
        self.tab_widget.setMaximumWidth(230*self.ratio)
        self.tab_widget.setMaximumHeight(500*self.ratio)

        self.tab_widget.addTab(self.create_fit_tab(), "Fit transient")
        self.tab_widget.addTab(self.create_events_tab(), "Detect events")

        # buttons
        buttons_layout = QtWidgets.QHBoxLayout()
        self.plot_btn = QtWidgets.QPushButton('Plot sweep')
        self.plot_btn.clicked.connect(self.plot_sweep_basic)
        self.clear_btn = QtWidgets.QPushButton('Clear plot')
        self.clear_btn.clicked.connect(self.clear_all)
        self.auto_btn = QtWidgets.QPushButton('Auto-range')
        self.auto_btn.clicked.connect(self.auto_plots)
        buttons_layout.addWidget(self.plot_btn)
        buttons_layout.addWidget(self.clear_btn)
        buttons_layout.addWidget(self.auto_btn)

        self.left_col.addWidget(self.tree_widget)
        self.left_col.addWidget(self.tab_widget)
        self.left_col.addLayout(buttons_layout)

        self.table = QtWidgets.QTableWidget()
        self.table.setFixedWidth(150*self.ratio)
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(['Amplitude (pA)'])
        header = self.table.horizontalHeader()
        header.setResizeMode(0, QtGui.QHeaderView.Stretch)

        self.plot_widget = pg.GraphicsLayoutWidget(self)

        self.layout.addLayout(self.left_col)
        self.layout.addWidget(self.plot_widget)
        self.layout.addWidget(self.table)

        self.setCentralWidget(central_widget)

    def setup_file_menu(self):
        file_menu = self.menubar.addMenu('File')
        load_abf_action = QtGui.QAction('Load Axon File', self)
        load_abf_action.triggered.connect(self.load_abf)

        load_pv_action = QtGui.QAction('Load PV folder', self)
        load_pv_action.triggered.connect(self.load_pv)

        file_menu.addAction(load_abf_action)
        file_menu.addAction(load_pv_action)

    def setup_copy_menu(self):
        copy_menu = self.menubar.addMenu('Copy Data')

        copy_calc_vals = QtGui.QAction('Copy calculated values', self)
        copy_calc_vals.triggered.connect(self.copy_calc_vals)
        copy_fit = QtGui.QAction('Copy fit', self)
        copy_fit.triggered.connect(self.copy_fit)
        copy_sub = QtGui.QAction('Copy subtraction', self)
        copy_sub.triggered.connect(self.copy_sub)

        copy_menu.addAction(copy_calc_vals)
        copy_menu.addAction(copy_fit)
        copy_menu.addAction(copy_sub)

    def create_fit_tab(self):
        self.fit_tab = QtWidgets.QWidget()
        self.transient_layout = QtWidgets.QVBoxLayout(self.fit_tab)

        self.transient_checkbox = QtGui.QCheckBox()
        self.transient_checkbox.setText('Fit and subtract transient')
        self.transient_checkbox.setChecked(True)
        self.transient_checkbox.stateChanged.connect(self.change_transient)

        self.stim_layout = QtWidgets.QHBoxLayout()
        self.stim_label = QtWidgets.QLabel('Stim start (s):')
        self.stim_label.setFixedWidth(self.label_width)
        self.stim_txt = QtWidgets.QLineEdit('0')
        self.stim_txt.setFixedWidth(self.edit_width)
        self.stim_txt.setSizePolicy(self.size_policy)
        self.stim_txt.editingFinished.connect(self.update_stim_time)
        self.stim_layout.addWidget(self.stim_label)
        self.stim_layout.addWidget(self.stim_txt)
        #self.stim_layout.addItem(self.hspacer)

        self.peak_layout = QtWidgets.QHBoxLayout()
        self.peak_label = QtWidgets.QLabel('Approx. t to peak (s from stim):')
        self.peak_label.setFixedWidth(self.label_width)
        self.peak_txt = QtWidgets.QLineEdit('0.02')
        self.peak_txt.setFixedWidth(self.edit_width)
        self.peak_txt.setSizePolicy(self.size_policy)
        self.peak_txt.editingFinished.connect(self.update_peak_time)
        self.peak_layout.addWidget(self.peak_label)
        self.peak_layout.addWidget(self.peak_txt)
        #self.peak_layout.addItem(self.hspacer)

        self.end_fit_layout = QtWidgets.QHBoxLayout()
        self.end_fit_label = QtWidgets.QLabel('End fit at (s):')
        self.end_fit_label.setFixedWidth(self.label_width)
        self.end_fit_txt = QtWidgets.QLineEdit('0.3')
        self.end_fit_txt.setFixedWidth(self.edit_width)
        self.end_fit_txt.setSizePolicy(self.size_policy)
        self.end_fit_txt.editingFinished.connect(self.update_end_fit_time)
        self.end_fit_layout.addWidget(self.end_fit_label)
        self.end_fit_layout.addWidget(self.end_fit_txt)

        self.decay_param_title = QtWidgets.QLabel('Decay fit paramaters')
        self.decay_param_title.setFixedHeight(30)
        self.decay_param_title.setAlignment(QtCore.Qt.AlignCenter)
        self.decay_param_title.setToolTip('a1*e^(-x*tau1)+a2*e^(-x*tau2)+a3*e^(-x*tau3)+c')

        # exponential #1
        self.a1_layout = QtWidgets.QHBoxLayout()
        self.a1_label = QtWidgets.QLabel('a1:')
        self.a1_label.setFixedWidth(self.label_width)
        self.a1_layout.addWidget(self.a1_label)

        self.a1_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.fit_tab)
        self.a1_slider.valueChanged.connect(self.change_a1_param)
        self.a1_slider.setRange(-1000, 3000)

        self.tau1_layout = QtWidgets.QHBoxLayout()
        self.tau1_label = QtWidgets.QLabel('tau1:')
        self.tau1_label.setFixedWidth(self.label_width)
        self.tau1_layout.addWidget(self.tau1_label)

        self.tau1_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.fit_tab)
        self.tau1_slider.valueChanged.connect(self.change_tau1_param)
        self.tau1_slider.setRange(-1000, 3000)

        #exponential #2
        self.a2_layout = QtWidgets.QHBoxLayout()
        self.a2_label = QtWidgets.QLabel('a2:')
        self.a2_label.setFixedWidth(self.label_width)
        self.a2_layout.addWidget(self.a2_label)

        self.a2_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.fit_tab)
        self.a2_slider.valueChanged.connect(self.change_a2_param)
        self.a2_slider.setRange(-1000, 3000)

        self.tau2_layout = QtWidgets.QHBoxLayout()
        self.tau2_label = QtWidgets.QLabel('tau2:')
        self.tau2_label.setFixedWidth(self.label_width)
        self.tau2_layout.addWidget(self.tau2_label)

        self.tau2_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.fit_tab)
        self.tau2_slider.valueChanged.connect(self.change_tau2_param)
        self.tau2_slider.setRange(-1000, 3000)

        #exponential #3
        self.a3_layout = QtWidgets.QHBoxLayout()
        self.a3_label = QtWidgets.QLabel('a3:')
        self.a3_label.setFixedWidth(self.label_width)
        self.a3_layout.addWidget(self.a3_label)

        #self.a3_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.fit_tab)
        #self.a3_slider.valueChanged.connect(self.change_a3_param)
        #self.a3_slider.setRange(-1000, 3000)

        self.tau3_layout = QtWidgets.QHBoxLayout()
        self.tau3_label = QtWidgets.QLabel('tau_3:')
        self.tau3_label.setFixedWidth(self.label_width)
        self.tau3_layout.addWidget(self.tau3_label)

        #self.tau3_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.fit_tab)
        #self.tau3_slider.valueChanged.connect(self.change_tau3_param)
        #self.tau3_slider.setRange(-1000, 3000)

        self.c_layout = QtWidgets.QHBoxLayout()
        self.c_label = QtWidgets.QLabel('c:')
        self.c_label.setFixedWidth(self.label_width)
        self.c_layout.addWidget(self.c_label)

        self.c_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.fit_tab)
        self.c_slider.valueChanged.connect(self.change_c_param)
        self.c_slider.setRange(-1000, 3000)

        button_layout = QtWidgets.QHBoxLayout()
        self.fit_button = QtWidgets.QPushButton('Fit and Plot')
        self.fit_button.clicked.connect(self.fit_and_plot)
        button_layout.addItem(self.hspacer)
        button_layout.addWidget(self.fit_button)
        button_layout.addItem(self.hspacer)

        self.transient_layout.addWidget(self.transient_checkbox)
        self.transient_layout.addLayout(self.stim_layout)
        self.transient_layout.addLayout(self.peak_layout)
        self.transient_layout.addLayout(self.end_fit_layout)
        self.transient_layout.addWidget(self.decay_param_title)
        self.transient_layout.addLayout(self.a1_layout)
        self.transient_layout.addWidget(self.a1_slider)
        self.transient_layout.addLayout(self.tau1_layout)
        self.transient_layout.addWidget(self.tau1_slider)
        self.transient_layout.addLayout(self.a2_layout)
        self.transient_layout.addWidget(self.a2_slider)
        self.transient_layout.addLayout(self.tau2_layout)
        self.transient_layout.addWidget(self.tau2_slider)
        #self.transient_layout.addLayout(self.a3_layout)
        #self.transient_layout.addWidget(self.a3_slider)
        #self.transient_layout.addLayout(self.tau3_layout)
        #self.transient_layout.addWidget(self.tau3_slider)
        self.transient_layout.addLayout(self.c_layout)
        self.transient_layout.addWidget(self.c_slider)
        self.transient_layout.addLayout(button_layout)
        self.transient_layout.addItem(self.vspacer)

        return self.fit_tab

    def create_events_tab(self):
        self.event_tab = QtWidgets.QWidget()
        self.event_param_layout = QtWidgets.QVBoxLayout(self.event_tab)
        self.event_param_title = QtWidgets.QLabel('Event detection paramaters')
        self.event_param_title.setAlignment(QtCore.Qt.AlignCenter)
        self.event_param_title.setFixedHeight(30)

        self.mpd_layout = QtWidgets.QHBoxLayout()
        self.mpd_label = QtWidgets.QLabel('Min. peak distance (s):')
        self.mpd_label.setFixedWidth(self.label_width)
        self.mpd_txt = QtWidgets.QLineEdit('0.01')
        self.mpd_txt.setFixedWidth(self.edit_width)
        self.mpd_txt.setSizePolicy(self.size_policy)
        self.mpd_txt.editingFinished.connect(self.update_mpd_val)
        self.mpd_layout.addWidget(self.mpd_label)
        self.mpd_layout.addWidget(self.mpd_txt)
        #self.mpd_layout.addItem(self.hspacer)

        self.rms_layout = QtWidgets.QHBoxLayout()
        self.rms_label = QtWidgets.QLabel('RMS multiplier:')
        self.rms_label.setFixedWidth(self.label_width)
        self.rms_val_txt = QtWidgets.QLineEdit('1')
        self.rms_val_txt.setFixedWidth(self.edit_width)
        self.rms_val_txt.setSizePolicy(self.size_policy)
        self.rms_val_txt.editingFinished.connect(self.update_rms_multiple)
        self.rms_layout.addWidget(self.rms_label)
        self.rms_layout.addWidget(self.rms_val_txt)
        #self.rms_layout.addItem(self.hspacer)

        self.rms_start_layout = QtWidgets.QHBoxLayout()
        self.rms_start_label = QtWidgets.QLabel('RMS region start (s):')
        self.rms_start_label.setFixedWidth(self.label_width)
        self.rms_start_txt = QtWidgets.QLineEdit('0.0')
        self.rms_start_txt.setFixedWidth(self.edit_width)
        self.rms_start_txt.setSizePolicy(self.size_policy)
        self.rms_start_txt.editingFinished.connect(self.update_rms_start)
        self.rms_start_layout.addWidget(self.rms_start_label)
        self.rms_start_layout.addWidget(self.rms_start_txt)
        #self.rms_start_layout.addItem(self.hspacer)

        self.rms_stop_layout = QtWidgets.QHBoxLayout()
        self.rms_stop_label = QtWidgets.QLabel('RMS region stop (s):')
        self.rms_stop_label.setFixedWidth(self.label_width)
        self.rms_stop_txt = QtWidgets.QLineEdit('0.1')
        self.rms_stop_txt.setFixedWidth(self.edit_width)
        self.rms_stop_txt.setSizePolicy(self.size_policy)
        self.rms_stop_txt.editingFinished.connect(self.update_rms_start)
        self.rms_stop_layout.addWidget(self.rms_stop_label)
        self.rms_stop_layout.addWidget(self.rms_stop_txt)
        #self.rms_stop_layout.addItem(self.hspacer)

        self.start_layout = QtWidgets.QHBoxLayout()
        self.start_label = QtWidgets.QLabel('Detection start (s from peak):')
        self.start_label.setFixedWidth(self.label_width)
        self.start_txt = QtWidgets.QLineEdit('0.02')
        self.start_txt.setFixedWidth(self.edit_width)
        self.start_txt.setSizePolicy(self.size_policy)
        self.start_txt.editingFinished.connect(self.update_start)
        self.start_layout.addWidget(self.start_label)
        self.start_layout.addWidget(self.start_txt)
        #self.start_layout.addItem(self.hspacer)

        self.stop_layout = QtWidgets.QHBoxLayout()
        self.stop_label = QtWidgets.QLabel('Detection stop (s):')
        self.stop_label.setFixedWidth(self.label_width)
        self.stop_txt = QtWidgets.QLineEdit('')
        self.stop_txt.setFixedWidth(self.edit_width)
        self.stop_txt.setSizePolicy(self.size_policy)
        self.stop_txt.editingFinished.connect(self.update_stop)
        self.stop_layout.addWidget(self.stop_label)
        self.stop_layout.addWidget(self.stop_txt)
        #self.stop_layout.addItem(self.hspacer)

        self.event_bsl_layout = QtWidgets.QHBoxLayout()
        self.event_bsl_label = QtWidgets.QLabel('Event baseline (# points):')
        self.event_bsl_label.setFixedWidth(self.label_width)
        self.event_bsl_txt = QtWidgets.QLineEdit('40')
        self.event_bsl_txt.setFixedWidth(self.edit_width)
        self.event_bsl_txt.setSizePolicy(self.size_policy)
        self.event_bsl_txt.editingFinished.connect(self.update_event_bsl)
        self.event_bsl_layout.addWidget(self.event_bsl_label)
        self.event_bsl_layout.addWidget(self.event_bsl_txt)
        #self.polyfit_layout.addItem(self.hspacer)

        self.smth_layout = QtWidgets.QHBoxLayout()
        self.smth_label = QtWidgets.QLabel('Smooth by (# points):')
        self.smth_label.setFixedWidth(self.label_width)
        self.smth_txt = QtWidgets.QLineEdit('10')
        self.smth_txt.setFixedWidth(self.edit_width)
        self.smth_txt.setSizePolicy(self.size_policy)
        self.smth_txt.editingFinished.connect(self.update_smth)
        self.smth_layout.addWidget(self.smth_label)
        self.smth_layout.addWidget(self.smth_txt)
        #self.smth_layout.addItem(self.hspacer)

        self.tolerance_layout = QtWidgets.QHBoxLayout()
        self.tolerance_label = QtWidgets.QLabel('Selection tolerance: ')
        self.tolerance_label.setFixedWidth(self.label_width)
        self.tolerance_txt = QtWidgets.QLineEdit('20')
        self.tolerance_txt.setFixedWidth(self.edit_width)
        self.tolerance_txt.setSizePolicy(self.size_policy)
        self.tolerance_txt.editingFinished.connect(self.update_tolerance)
        self.tolerance_layout.addWidget(self.tolerance_label)
        self.tolerance_layout.addWidget(self.tolerance_txt)
        #self.tolerance_layout.addItem(self.hspacer)

        buttons_layout = QtWidgets.QHBoxLayout()
        self.detect_btn = QtWidgets.QPushButton('Run detection')
        self.detect_btn.clicked.connect(self.run_detection)
        self.num_btn = QtWidgets.QPushButton('Calc vals')
        self.num_btn.clicked.connect(self.calc_vals)
        buttons_layout.addWidget(self.detect_btn)
        buttons_layout.addWidget(self.num_btn)

        self.event_param_layout.addWidget(self.event_param_title)
        self.event_param_layout.addLayout(self.mpd_layout)
        self.event_param_layout.addLayout(self.rms_layout)
        self.event_param_layout.addLayout(self.rms_start_layout)
        self.event_param_layout.addLayout(self.rms_stop_layout)
        self.event_param_layout.addLayout(self.start_layout)
        self.event_param_layout.addLayout(self.stop_layout)
        self.event_param_layout.addLayout(self.event_bsl_layout)
        self.event_param_layout.addLayout(self.smth_layout)
        self.event_param_layout.addLayout(self.tolerance_layout)
        self.event_param_layout.addLayout(buttons_layout)
        self.event_param_layout.addItem(self.vspacer)

        return self.event_tab

    def load_abf(self):
        abf_file = QtWidgets.QFileDialog.getOpenFileName(self
                                                    , 'Select Axon (.abf) file'
                                                    , self.parent_dir)[0]
        self.parent_dir = os.path.dirname(abf_file)
        if os.path.splitext(abf_file)[-1] == '.abf':
            self.df = abf.read_abf(abf_file)
            self.update_tree(abf_file)
        elif any(abf_file):
            self.gen_error_mbox('Invalid file')

    def load_pv(self):
        folder = QtGui.QFileDialog().getExistingDirectory(self,
                                                          "Select PV data folder",
                                                          self.parent_dir)
        self.parent_dir = os.path.dirname(folder)
        data_dict = rpv.import_folder(folder)
        if data_dict['voltage recording'] is None:
            message = 'Folder does not contain necessary data'
            self.gen_error_mbox(message)
        else:
            self.df = data_dict['voltage recording']
            self.update_tree(folder)

    def copy_calc_vals(self):
        if self.heights is not None:
            df = pd.DataFrame({'Amplitude (pA)': self.heights})
            df.to_clipboard(index=False)

    def copy_fit(self):
        if self.sweep is not None and 'fit' in self.sweep.columns:
            self.sweep[['time', 'fit']].to_clipboard(index=False)

    def copy_sub(self):
        if self.sweep is not None and 'subtraction' in self.sweep.columns:
            self.sweep[['time', 'subtraction']].to_clipboard(index=False)

    def update_tree(self, path):
        self.tree_widget.clear()
        self.tree_widget.headerItem().setText(0, os.path.split(path)[-1])
        self.tree_widget.headerItem().setToolTip(0, path)
        sweeps = self.df.index.levels[0]

        for sweep in sweeps:
            sweep_item = QtWidgets.QTreeWidgetItem(self.tree_widget)
            sweep_item.setFlags(QtCore.Qt.ItemIsEnabled |
                                QtCore.Qt.ItemIsUserCheckable)
            sweep_item.setText(0, sweep)
            sweep_item.setCheckState(0, QtCore.Qt.Unchecked)

        top = self.tree_widget.topLevelItem(0)
        top.setCheckState(0, QtCore.Qt.Checked)

    def update_checked(self, item):
        if item.checkState(0) == QtCore.Qt.Checked:
            if self.checked is None:
                self.checked = item
                self.sweep = self.df.loc[item.text(0)]
                self.sampling = 1/(self.sweep.time.iloc[1]-
                                   self.sweep.time.iloc[0])
            else:
                self.checked.setCheckState(0, QtCore.Qt.Unchecked)
                self.checked = item
                self.sweep = self.df.loc[item.text(0)]
                self.sampling = 1/(self.sweep.time.iloc[1]-
                                   self.sweep.time.iloc[0])
        else:
            self.checked = None
            self.sweep = None

    def update_stim_time(self):
        new_val = self.stim_txt.text()
        try:
            new_val = float(new_val)
            if new_val < 0:
                message = 'Stim start must >= 0'
                self.gen_error_mbox(message)
            else:
                self.stim_start = new_val
        except:
            message = 'Stim start must either >= 0'
            self.gen_error_mbox(message)

    def update_peak_time(self):
        new_val = self.peak_txt.text()
        try:
            new_val = float(new_val)
            if new_val < 0:
                message = 'Time must >= 0'
                self.gen_error_mbox(message)
            else:
                self.peak_time_delta = new_val
        except:
            message = 'Time must either >= 0'
            self.gen_error_mbox(message)

    def update_end_fit_time(self):
        new_val = self.end_fit_txt.text()
        try:
            new_val = float(new_val)
            if new_val < 0:
                message = 'Time must >= 0'
                self.gen_error_mbox(message)
            else:
                self.end_fit = new_val
        except:
            message = 'Time must either >= 0'
            self.gen_error_mbox(message)

    def update_mpd_val(self):
        try:
            new_val = float(self.mpd_txt.text())
            if new_val <= 0:
                message = 'Min. peak distance must be > 0'
                self.gen_error_mbox(message)
            else:
                self.mpd = new_val
        except ValueError:
            message = 'Min. peak distance must be > 0'
            self.gen_error_mbox(message)

    def update_rms_multiple(self):
        try:
            new_val = float(self.rms_val_txt.text())
            if new_val <= 0:
                message = 'RMS multiplier must be > 0'
                self.gen_error_mbox(message)
            else:
                self.rms_multiple = new_val
        except ValueError:
            message = 'RMS multiplier must be > 0'
            self.gen_error_mbox(message)

    def update_rms_start(self):
        try:
            new_val = float(self.rms_start_txt.text())
            if new_val < 0:
                message = 'RMS region start time must be >=0'
                self.gen_error_mbox(message)
            else:
                self.rms_start = new_val
        except ValueError:
            message = 'RMS region start time must be >=0'
            self.gen_error_mbox(message)

    def update_rms_stop(self):
        try:
            new_val = float(self.rms_stop_txt.text())
            if new_val < 0:
                message = 'RMS region stop time must be >=0'
                self.gen_error_mbox(message)
            else:
                self.rms_stop = new_val
        except ValueError:
            message = 'RMS region stop time must be >=0'
            self.gen_error_mbox(message)

    def update_start(self):
        new_val = self.start_txt.text()
        try:
            new_val = float(new_val)
            if new_val < 0:
                message = '''Start time must either be a number >= 0 or
                            left empty'''
                self.gen_error_mbox(message)
            else:
                self.detect_start = new_val
        except:
            if any(new_val):
                message = '''Start time must either be a number >= 0 or
                            left empty'''
                self.gen_error_mbox(message)
            else:
                self.detect_start = None

    def update_stop(self):
        new_val = self.stop_txt.text()
        try:
            new_val = float(new_val)
            if new_val < 0:
                message = '''Stop time must either be a number >= 0 or
                            left empty'''
                self.gen_error_mbox(message)
            else:
                self.detect_stop = new_val
        except:
            if any(new_val):
                message = '''Stop time must either be a number >= 0 or
                            left empty'''
                self.gen_error_mbox(message)
            else:
                self.detect_stop = None

    def update_smth(self):
        try:
            new_val = int(self.smth_txt.text())
            if new_val < 1:
                message = 'Smooth by must be integer >= 1'
                self.gen_error_mbox(message)
            else:
                self.smth_by = new_val
        except ValueError:
            message = 'Smooth by must be integer >= 1'
            self.gen_error_mbox(message)

    def update_tolerance(self):
        try:
            new_val = int(self.tolerance_txt.text())
            if new_val < 1:
                message = 'Selection tolerance must be integer >= 1'
                self.gen_error_mbox(message)
            else:
                self.tolerance = new_val
        except ValueError:
            message = 'Selection tolerance must be integer >= 1'
            self.gen_error_mbox(message)

    def update_event_bsl(self):
        try:
            new_val = int(self.event_bsl_txt.text())
            if new_val < 1:
                message = 'Event baseline window must an an integer >= 1'
                self.gen_error_mbox(message)
            else:
                self.event_bsl_window = new_val
        except ValueError:
            message = 'Event baseline window must an an integer >= 1'
            self.gen_error_mbox(message)

    def fit_and_plot(self):
        self.clear_all()
        self.gen_fit()
        self.plot_fit()

    def set_fit_params(self, popt):
        self.fit_a1 = popt[0]
        self.fit_tau1 = popt[1]
        self.fit_a2 = popt[2]
        self.fit_tau2 = popt[3]
        #self.fit_a3 = popt[4]
        #self.fit_tau3 = popt[5]
        self.fit_c = popt[-1]

        self.user_a1 = self.fit_a1
        self.user_a2 = self.fit_a2
        #self.user_a3 = self.fit_a3
        self.user_tau1 = self.fit_tau1
        self.user_tau2 = self.fit_tau2
        #self.user_tau3 = self.fit_tau3
        self.user_c = self.fit_c

        self.a1_label.setText('a1: %s'%self.fit_a1)
        self.a2_label.setText('a2: %s'%self.fit_a2)
        #self.a3_label.setText('a3: %s'%self.fit_a3)
        self.tau1_label.setText('tau_1: %s'%self.fit_tau1)
        self.tau2_label.setText('tau2: %s'%self.fit_tau2)
        #self.tau3_label.setText('tau3: %s'%self.fit_tau3)
        self.c_label.setText('c: %s'%self.fit_c)

        self.slide_a1_step = self.fit_a1 / 1000
        self.slide_a2_step = self.fit_a2 / 1000
        #self.slide_a3_step = self.fit_a3 / 1000
        self.slide_tau1_step = self.fit_tau1 / 1000
        self.slide_tau2_step = self.fit_tau2 / 1000
        #self.slide_tau3_step = self.fit_tau3 / 1000
        self.slide_c_step = self.fit_c / 1000

        self.a1_slider.setSliderPosition(0)
        self.a2_slider.setSliderPosition(0)
        #self.a3_slider.setSliderPosition(0)
        self.tau1_slider.setSliderPosition(0)
        self.tau2_slider.setSliderPosition(0)
        #self.tau3_slider.setSliderPosition(0)
        self.c_slider.setSliderPosition(0)

    def change_transient(self):
        if self.transient_checkbox.isChecked():
            self.start_label.setText('Detection start (s from peak):')
            self.start_txt.setText('0.02')
            self.detect_start = 0.02
            self.stim_txt.setEnabled(True)
            self.peak_txt.setEnabled(True)
            self.sub_trans = True
        else:
            self.start_label.setText('Detection start (s):')
            self.start_txt.setText('')
            self.detect_start = None
            self.stim_txt.setEnabled(False)
            self.peak_txt.setEnabled(False)
            self.sub_trans = False

    def change_a1_param(self, val):
        if self.user_a1 is not None:
            self.user_a1 = self.fit_a1 + val*self.slide_a1_step
            self.a1_label.setText('a1: %s'%self.user_a1)
            self.update_decay_fit()

    def change_a2_param(self, val):
        if self.user_a2 is not None:
            self.user_a2 = self.fit_a2 + val*self.slide_a2_step
            self.a2_label.setText('a2: %s'%self.user_a2)
            self.update_decay_fit()

    def change_a3_param(self, val):
        if self.user_a3 is not None:
            self.user_a3 = self.fit_a3 + val*self.slide_a3_step
            self.a3_label.setText('a3: %s'%self.user_a3)
            self.update_decay_fit()

    def change_tau1_param(self, val):
        if self.user_tau1 is not None:
            self.user_tau1 = self.fit_tau1 + val*self.slide_tau1_step
            self.tau1_label.setText('tau1: %s'%self.user_tau1)
            self.update_decay_fit()

    def change_tau2_param(self, val):
        if self.user_tau2 is not None:
            self.user_tau2 = self.fit_tau2 + val*self.slide_tau2_step
            self.tau2_label.setText('tau2: %s'%self.user_tau2)
            self.update_decay_fit()

    def change_tau3_param(self, val):
        if self.user_tau3 is not None:
            self.user_tau3 = self.fit_tau3 + val*self.slide_tau3_step
            self.tau3_label.setText('tau3: %s'%self.user_tau3)
            self.update_decay_fit()

    def change_c_param(self, val):
        if self.user_c is not None:
            self.user_c = self.fit_c + val*self.slide_c_step
            self.c_label.setText('c: %s'%self.user_c)
            self.update_decay_fit()

    def plot_sweep_basic(self):
        if self.sweep is not None:
            plot = self.plot_widget.addPlot(self.counter, 0, enableMenu=False)
            plot.plot(self.sweep.time, self.sweep.primary, pen='b')
            self.counter += 1

            return plot

    def clear_all(self):
        self.plot_widget.clear()
        self.table.setRowCount(0)

        self.counter = 0
        self.data_col = 'primary'
        self.detection_plot = None
        self.fit_plot = None
        self.heights = None
        self.indexes = []
        #self.peak_time = None
        self.plots = []
        self.points = {}
        self.points_plot = None
        self.polyfit = None

    def auto_plots(self):
        if self.counter > 0:
            for i in range(self.counter):
                self.plot_widget.getItem(i, 0).autoRange()

    def gen_fit(self):
        mask = ((self.sweep.time >= self.stim_start) &
                    (self.sweep.time <= self.stim_start + self.peak_time_delta))
        self.peak_ix = self.sweep.loc[mask, 'primary'].idxmin()
        self.peak_time = self.sweep.loc[self.peak_ix, 'time']

        self.fit_transient()
        self.update_sweep_fit()

    def update_decay_fit(self):
        #def fit_eq(x, a, b, c, d, e, f, g):
            #return a*(np.exp(-x/b)) + c*(np.exp(-x/d)) + e*(np.exp(-x/f))+ g
        def fit_eq(x, a, b, c, d, e):
            return a*(np.exp(-x/b)) + c*(np.exp(-x/d)) + e

        self.fit_vals = fit_eq(self.fit_x
                               , self.user_a1, self.user_tau1
                               , self.user_a2, self.user_tau2
                               #, self.user_a3, self.user_tau3
                               , self.user_c)
        self.update_sweep_fit()
        self.fit_plot.setData(self.sweep.time, self.sweep.fit)

    def update_sweep_fit(self):
        first20 = self.sweep.loc[:20, self.data_col].mean()
        front_fill = util.simple_smoothing(self.sweep.loc[:self.peak_ix, self.data_col].values, 20)
        front_fill[np.isnan(front_fill)] = first20

        back_fill = np.full(len(self.sweep)-(len(front_fill) + len(self.fit_vals)),
                             self.fit_vals[-1])
        self.sweep['fit'] = np.concatenate((front_fill, self.fit_vals, back_fill))

    def gen_subtraction(self):
        if 'fit' not in self.sweep.columns:
            self.gen_fit()
        self.sweep['subtraction'] = self.sweep.primary - self.sweep.fit
        self.data_col = 'subtraction'

    def plot_fit(self):
        plot1 = self.plot_sweep_basic()
        self.fit_plot = plot1.plot(self.sweep.time, self.sweep.fit,
                                   pen=pg.mkPen('r', width=1.5*self.ratio))

        return plot1

    def gen_subset(self):
        if self.detect_start is None:
            start = self.sweep.time.iloc[0]
        elif self.sub_trans:
            start = self.peak_time + self.detect_start
        else:
            start = self.detect_start

        if self.detect_stop is None:
            stop = self.sweep.time.iloc[-1]
        else:
            stop = self.detect_stop

        mask = ((self.sweep.time >= start) & (self.sweep.time <= stop))

        return self.sweep.loc[mask]

    def get_event_ixs(self, subset):
        mpd_points = int(self.mpd * self.sampling)
        smthd = subset['smthd'].values
        ixs = pace.detect_peaks(smthd, mpd=mpd_points, valley=True)
        self.indexes = subset.index.values[ixs].tolist()
        self.check_height()

    def plot_detected_events(self, subtraction=True, xlink=None):
        self.detection_plot = self.plot_widget.addPlot(self.counter
                                                       , 0
                                                       , enableMenu=False)
        x = self.sweep.time.values
        y = self.sweep[self.data_col]
        self.detection_plot.plot(x, y, pen='b', name='data')
        self.detection_plot.scene().sigMouseClicked.connect(self.plot_clicked)
        if xlink is not None:
            self.detection_plot.setXLink(xlink)

        x = self.sweep.loc[self.indexes, 'time'].values
        y = self.sweep.loc[self.indexes, self.data_col].values
        self.points_plot = self.detection_plot.plot(x, y, pen=None
                                                    , symbol='o'
                                                    , symbolPen='r'
                                                    , symbolBrush='r'
                                                    , symbolSize=7*self.ratio)

        self.points_plot.sigPointsClicked.connect(self.point_clicked)

    def run_detection(self):
        self.clear_all()
        if self.sub_trans and self.sweep is not None:
            self.gen_subtraction()
            xlink_plot = self.plot_fit()
            self.sweep['smthd'] = util.simple_smoothing(self.sweep['subtraction'].values,
                                                        self.smth_by)
            #self.sweep_median = self.sweep[self.data_col].median()
            self.data_col = 'smthd'
            subset = self.gen_subset()
            self.get_event_ixs(subset)
            self.plot_detected_events(xlink=xlink_plot)

        elif self.sweep is not None:
            self.sweep['smthd'] = util.simple_smoothing(self.sweep.primary.values,
                                                        self.smth_by)
            #self.sweep_median = self.sweep[self.data_col].median()
            self.data_col = 'smthd'
            subset = self.gen_subset()
            self.get_event_ixs(subset)
            self.plot_detected_events(subtraction=False)

    def calc_vals(self):
        self.table.setRowCount(0)
        if any(self.indexes):
            self.indexes.sort()
            self.heights = self.get_heights()

            for i, height in enumerate(self.heights):
                self.table.insertRow(i)
                item = QtGui.QTableWidgetItem("%0.3f" % height)
                self.table.setItem(i, 0, item)

    def add_point(self, index):
        if index not in self.indexes:
            self.indexes.append(index)
            self.update_points_plot()

    def remove_point(self, index):
        if index in self.indexes:
            self.indexes.remove(index)
            self.update_points_plot()

    def update_points_plot(self):
        x = self.sweep.loc[self.indexes, 'time'].values
        y = self.sweep.loc[self.indexes, self.data_col].values
        self.points_plot.setData(x, y
                                 , pen=None
                                 , symbol='o'
                                 , symbolPen='r'
                                 , symbolBrush='r'
                                 , symbolSize=7*self.ratio)

    def find_nearest_peak(self, index, y_pos=None):
        if index < self.tolerance:
            ix1 = 0
        else:
            ix1 = index-self.tolerance
        if index+self.tolerance > len(self.sweep):
            ix2 = len(self.sweep)
        else:
            ix2 = index+self.tolerance

        ix = self.sweep[self.data_col].iloc[ix1:ix2].idxmin()


        cur_range = (self.sweep[self.data_col].iloc[ix1:ix2].max() -
                     self.sweep[self.data_col].iloc[ix1:ix2].min())
        peak = self.sweep.loc[ix, self.data_col]
        diff = abs(peak - y_pos)
        if diff < cur_range/5:
            self.add_point(ix)

    def plot_clicked(self, event):
        if event.button()==2:
            items = self.plot_widget.scene().items(event.scenePos())
            for item in items:
                if isinstance(item, pg.ViewBox):
                    pos = item.mapSceneToView(event.scenePos())
                    index = int(round(pos.x()*self.sampling))
                    if 0 <= index < len(self.sweep):
                        self.find_nearest_peak(index, pos.y())

    def point_clicked(self, item, points):
        tdiff = time.time() - self.time
        if tdiff < 1:
            point_x = points[0].pos()[0]
            index = int(round(point_x*self.sampling))
            self.remove_point(index)

        self.time = time.time()

    def gen_error_mbox(self, message):
        msg = QtWidgets.QMessageBox()
        msg.Critical
        msg.setText(message)
        msg.setWindowTitle("Error")
        msg.exec_()

    def fit_transient(self):
        #diff = (self.peak_time - self.stim_start) * 0.2
        fit_sub = self.sweep.loc[(self.sweep.time >= self.peak_time) &
                                 (self.sweep.time <= self.end_fit)]
        self.fit_start_ix = fit_sub.index.values[0]

        guess = np.array([-1, 1, -1, 10, fit_sub.primary.max()])
        self.fit_x = (fit_sub.time - fit_sub.time.values[0]).values
        self.fit_x *= 1e3

        #def fit_eq(x, a, b, c, d, e, f, g):
            #return a*(np.exp(-x/b)) + c*(np.exp(-x/d)) + e*(np.exp(-x/f))+ g
        def fit_eq(x, a, b, c, d, e):
            return a*(np.exp(-x/b)) + c*(np.exp(-x/d)) + e

        try:
            popt, pcov = curve_fit(fit_eq, self.fit_x, fit_sub.primary,
                                   guess, maxfev=10000)
        except RuntimeError:
            message = '''Fit Failed\nCheck input parameters.\nIf correct try increasing (or decreasing) end fit time'''
            self.gen_error_mbox(message)

        #x_full_zeroed = peak_sub.time - peak_sub.time.values[0]
        self.fit_vals = fit_eq(self.fit_x, *popt)
        self.set_fit_params(popt)

    def calc_rms(self, vals):
        vals = np.atleast_1d(vals).astype('float64')
        mean = np.nanmean(vals)
        ss = np.nansum(np.power((vals-mean), 2))
        rms = np.sqrt(ss/len(vals))

        return rms

    def gen_polyfit(self):
        x = self.sweep.time
        y = self.sweep[self.data_col].copy()
        y.fillna(y.mean(), inplace=True)
        coeffs = np.polyfit(x, y, self.poly_order)
        fit = np.poly1d(coeffs)
        self.sweep['polyfit'] = fit(self.sweep.time)
        self.poly_subset = self.gen_subset()['polyfit'].values

    def get_heights(self):
        delta = self.event_bsl_window
        heights = []
        for ix in self.indexes:
            if ix-delta < 0:
                ix1 = 0
            else:
                ix1 = ix-delta

            if ix+delta >= len(self.sweep[self.data_col]):
                ix2 = len(self.sweep[self.data_col]) - 1
            else:
                ix2 = ix+delta

            #max1 = np.nanmax(self.sweep.loc[ix1:ix, self.data_col].values)
            #max2 = np.nanmax(self.sweep.loc[ix+1:ix2+1, self.data_col].values)
            #bsl = np.mean([max1, max2])
            bsl = np.nanmax(self.sweep.loc[ix1:ix2+1, self.data_col].values)
            height = bsl - self.sweep.loc[ix, self.data_col]
            heights.append(height)

        return np.array(heights)

    def check_height(self):
        #values = np.atleast_1d(values).astype('float64')
        #indexes = np.atleast_1d(indexes).astype('int')
        try:
            mask = ((self.sweep.time >= self.rms_start) &
                   (self.sweep.time <= self.rms_stop))
            rms_region = self.sweep.loc[mask, self.data_col]
            rms = self.calc_rms(rms_region)
        except:
            message = 'No data points in RMS region. Check start and stop times'
            self.gen_error_mbox(message)
            return

        #diff = values[indexes] - self.poly_subset[indexes]
        indexes = np.atleast_1d(self.indexes).astype('int')
        heights = self.get_heights()
        #diff = values[indexes] - self.sweep_median
        #new_ixs = indexes[-1*diff > rms*self.rms_multiple]
        self.indexes = indexes[heights > rms*self.rms_multiple].tolist()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename='./error.log',
                        filemode='w')
    def log_uncaught_exceptions(ex_cls, ex, tb):
        logging.debug(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
        logging.debug(''.join(traceback.format_tb(tb)))
        logging.debug('{0}: {1}\n'.format(ex_cls, ex))

    sys.excepthook = log_uncaught_exceptions
    app = QtWidgets.QApplication(sys.argv)
    ex = MiniAnalysis()
    ex.show()
    sys.exit(app.exec_())
