"""An interactive PyQT QC data frame."""

import logging

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtProperty, Qt, QVariant, QAbstractTableModel, QModelIndex, \
    QObject, QPoint, pyqtSignal, pyqtSlot, QCoreApplication, QSettings
from PyQt5.QtGui import QColor, QPalette, QShowEvent
from PyQt5.QtWidgets import QMenu, QAction
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import pandas as pd
import numpy as np
from pyqtgraph import colormap, ColorMap

from ibllib.misc import qt

_logger = logging.getLogger(__name__)


class DataFrameTableModel(QAbstractTableModel):
    def __init__(self, parent: QObject = ..., dataFrame: pd.DataFrame | None = None):
        super().__init__(parent)
        self._dataframe = pd.DataFrame() if dataFrame is None else dataFrame

    def setDataFrame(self, dataFrame: pd.DataFrame):
        self.beginResetModel()
        self._dataframe = dataFrame.copy()
        self.endResetModel()

    def dataFrame(self) -> pd.DataFrame:
        return self._dataframe

    dataFrame = pyqtProperty(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...):
        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            if orientation == Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return QVariant()

    def rowCount(self, parent: QModelIndex = ...):
        if isinstance(parent, QModelIndex) and parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent: QModelIndex = ...):
        if isinstance(parent, QModelIndex) and parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index: QModelIndex, role: int = ...) -> QVariant:
        if not index.isValid():
            return QVariant()
        if role == Qt.DisplayRole:
            val = self._dataframe.iloc[index.row(), index.column()]
            if isinstance(val, np.generic):
                return val.item()
            return QVariant(str(val))
        return QVariant()

    def sort(self, column: int, order: Qt.SortOrder = ...):
        if self.columnCount() == 0:
            return
        column = self._dataframe.columns[column]
        self.layoutAboutToBeChanged.emit()
        self._dataframe.sort_values(by=column, ascending=not order, inplace=True)
        self.layoutChanged.emit()


class ColoredDataFrameTableModel(DataFrameTableModel):
    colormapChanged = pyqtSignal(str)
    alphaChanged = pyqtSignal(int)
    _normData = pd.DataFrame
    _background: np.ndarray
    _cmap: ColorMap
    _alpha: int
    _foreground: np.ndarray

    def __init__(self, parent: QObject = ..., dataFrame: pd.DataFrame | None = None,
                 colormap: str = 'plasma', alpha: int = 255):
        super().__init__(parent=parent, dataFrame=dataFrame)
        self.modelReset.connect(self._normalizeData)
        self.dataChanged.connect(self._normalizeData)
        self.colormapChanged.connect(self._defineColors)
        self.setColormap(colormap)
        self.setAlpha(alpha)

    @pyqtSlot(str)
    def setColormap(self, name: str):
        for source in [None, 'matplotlib', 'colorcet']:
            if name in colormap.listMaps(source):
                self._cmap = colormap.get(name, source)
                self.colormapChanged.emit(name)
                return
        _logger.warning(f'No such colormap: "{name}"')

    def getColormap(self) -> str:
        return self._cmap.name

    colormap = pyqtProperty(str, fget=getColormap, fset=setColormap)

    @pyqtSlot(int)
    def setAlpha(self, alpha: int = 255):
        _, self._alpha, _ = sorted([0, alpha, 255])
        self.alphaChanged.emit(self._alpha)
        self.layoutChanged.emit()

    def getAlpha(self) -> int:
        return self._alpha

    alpha = pyqtProperty(int, fget=getAlpha, fset=setAlpha)

    def _normalizeData(self):
        df = self._dataframe.copy()
        if df.empty:
            self._background = df
            return

        # coerce non-bool / non-numeric values to numeric
        cols = df.select_dtypes(exclude=['bool', 'number']).columns
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

        # normalize numeric values, avoiding inf values and division by zero
        cols = df.select_dtypes(include=['number']).columns
        df[cols].replace([np.inf, -np.inf], np.nan)
        m = df[cols].nunique() <= 1  # boolean mask for columns with only 1 unique value
        df[cols[m]] = df[cols[m]].where(df[cols[m]].isna(), other=0.0)
        cols = cols[~m]
        df[cols] = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())

        # convert boolean values
        cols = df.select_dtypes(include=['bool']).columns
        df[cols] = df[cols].astype(float)

        # store as property & call _setRgba()
        self._normData = df
        self._defineColors()

    def _defineColors(self):
        if self._normData.empty:
            self._background = np.ndarray([])
            self._foreground = np.ndarray([])
        else:
            m = np.isfinite(self._normData)  # binary mask for finite values
            self._background = np.ones((*self._normData.shape, 3), dtype=int) * 255
            self._background[m] = self._cmap.mapToByte(self._normData.values[m])[:, :3]
            self._foreground = 255 - (self._background * np.array([[[0.21, 0.72, 0.07]]])).sum(axis=2).astype(int)
        self.layoutChanged.emit()

    def data(self, index, role=...):
        if not index.isValid():
            return QVariant()
        if role == Qt.BackgroundRole:
            row = self._dataframe.index[index.row()]
            val = self._background[row][index.column()]
            return QColor.fromRgb(*val, self._alpha)
        if role == Qt.ForegroundRole:
            row = self._dataframe.index[index.row()]
            val = self._foreground[row][index.column()] * self._alpha
            return QColor('black') if val < 32512 else QColor('white')
        return super().data(index, role)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100, wheel=None):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(
            self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        FigureCanvasQTAgg.updateGeometry(self)
        if wheel:
            self.ax, self.ax2 = fig.subplots(
                2, 1, gridspec_kw={"height_ratios": [2, 1]}, sharex=True
            )
        else:
            self.ax = fig.add_subplot(111)
        self.draw()


class PlotWindow(QtWidgets.QWidget):
    def __init__(self, parent=None, wheel=None):
        QtWidgets.QWidget.__init__(self, parent=None)
        self.canvas = PlotCanvas(wheel=wheel)
        self.vbl = QtWidgets.QVBoxLayout()  # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
        self.vbl.addWidget(NavigationToolbar2QT(self.canvas, self))


class GraphWindow(QtWidgets.QWidget):
    _pinnedColumns = []

    def __init__(self, parent=None, wheel=None):
        QtWidgets.QWidget.__init__(self, parent=parent)

        # Store layout changes to QSettings
        self.settings = QSettings()

        self.columnPinned = pyqtSignal(int, bool)

        self.pushButtonLoad = QtWidgets.QPushButton("Select File", self)
        self.pushButtonLoad.clicked.connect(self.loadFile)

        # define table model & view
        self.tableModel = ColoredDataFrameTableModel(self)
        self.tableView = QtWidgets.QTableView(self)
        self.tableView.setModel(self.tableModel)
        self.tableView.setSortingEnabled(True)
        self.tableView.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.tableView.horizontalHeader().setSectionsMovable(True)
        self.tableView.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableView.horizontalHeader().customContextMenuRequested.connect(self.contextMenu)
        self.tableView.doubleClicked.connect(self.tv_double_clicked)

        # define colors for highlighted cells
        p = self.tableView.palette()
        p.setColor(QPalette.Highlight, Qt.black)
        p.setColor(QPalette.HighlightedText, Qt.white)
        self.tableView.setPalette(p)

        # QAction for pinning columns
        self.pinAction = QAction('Pin column', self)
        self.pinAction.setCheckable(True)
        self.pinAction.toggled.connect(self.pinColumn)

        # Filter columns by name
        self.lineEditFilter = QtWidgets.QLineEdit(self)
        self.lineEditFilter.setPlaceholderText('Filter columns')
        self.lineEditFilter.textChanged.connect(self.changeFilter)

        # colormap picker
        self.comboboxColormap = QtWidgets.QComboBox(self)
        colormaps = {self.tableModel.colormap, 'inferno', 'magma', 'plasma'}
        self.comboboxColormap.addItems(sorted(list(colormaps)))
        self.comboboxColormap.setCurrentText(self.tableModel.colormap)
        self.comboboxColormap.currentTextChanged.connect(self.tableModel.setColormap)

        # slider for alpha values
        self.sliderAlpha = QtWidgets.QSlider(Qt.Horizontal, self)
        self.sliderAlpha.setMinimum(0)
        self.sliderAlpha.setMaximum(255)
        self.sliderAlpha.setValue(self.tableModel.alpha)
        self.sliderAlpha.valueChanged.connect(self.tableModel.setAlpha)

        # Horizontal layout
        hLayout = QtWidgets.QHBoxLayout()
        hLayout.addWidget(self.lineEditFilter)
        hLayout.addWidget(QtWidgets.QLabel('Colormap', self))
        hLayout.addWidget(self.comboboxColormap)
        hLayout.addWidget(QtWidgets.QLabel('Alpha', self))
        hLayout.addWidget(self.sliderAlpha)
        hLayout.addStretch(1)
        hLayout.addWidget(self.pushButtonLoad)

        # Vertical layout
        vLayout = QtWidgets.QVBoxLayout(self)
        vLayout.addLayout(hLayout)
        vLayout.addWidget(self.tableView)

        self.wplot = PlotWindow(wheel=wheel)
        self.wplot.show()
        self.tableModel.dataChanged.connect(self.wplot.canvas.draw)

        self.wheel = wheel

    def showEvent(self, a0: QShowEvent) -> None:
        super().showEvent(a0)
        self.activateWindow()

    def contextMenu(self, pos: QPoint):
        idx = self.sender().logicalIndexAt(pos)
        action = self.pinAction
        action.setData(idx)
        action.setChecked(idx in self._pinnedColumns)
        menu = QMenu(self)
        menu.addAction(action)
        menu.exec(self.sender().mapToGlobal(pos))

    @pyqtSlot(bool)
    @pyqtSlot(bool, int)
    def pinColumn(self, pin: bool, idx: int | None = None):
        idx = idx if idx is not None else self.sender().data()
        if not pin and idx in self._pinnedColumns:
            self._pinnedColumns.remove(idx)
        if pin and idx not in self._pinnedColumns:
            self._pinnedColumns.append(idx)
        self.changeFilter(self.lineEditFilter.text())

    def changeFilter(self, string: str):
        headers = [self.tableModel.headerData(x, Qt.Horizontal, Qt.DisplayRole)
                   for x in range(self.tableModel.columnCount())]
        for idx, column in enumerate(headers):
            self.tableView.setColumnHidden(idx, string.lower() not in column.lower()
                                           and idx not in self._pinnedColumns)

    def loadFile(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open File", "", "CSV Files (*.csv)"
        )
        if len(fileName) == 0:
            return
        df = pd.read_csv(fileName)
        self.updateDataframe(df)

    def updateDataframe(self, dataFrame: pd.DataFrame):
        self.tableModel.setDataFrame(dataFrame)

    def tv_double_clicked(self, index: QModelIndex):
        data = self.tableModel.dataFrame.iloc[index.row()]
        t0 = data["intervals_0"]
        t1 = data["intervals_1"]
        dt = t1 - t0
        if self.wheel:
            idx = np.searchsorted(self.wheel["re_ts"], np.array([t0 - dt / 10, t1 + dt / 10]))
            period = self.wheel["re_pos"][idx[0] : idx[1]]
            if period.size == 0:
                _logger.warning("No wheel data during trial #%i", index.row())
            else:
                min_val, max_val = np.min(period), np.max(period)
                self.wplot.canvas.ax2.set_ylim(min_val - 1, max_val + 1)
            self.wplot.canvas.ax2.set_xlim(t0 - dt / 10, t1 + dt / 10)
        self.wplot.canvas.ax.set_xlim(t0 - dt / 10, t1 + dt / 10)
        self.wplot.setWindowTitle(f"Trial {data.get('trial_no', '?')}")
        self.wplot.canvas.draw()


def viewqc(qc=None, title=None, wheel=None):
    QCoreApplication.setOrganizationName('International Brain Laboratory')
    QCoreApplication.setOrganizationDomain('internationalbrainlab.org')
    QCoreApplication.setApplicationName('QC Viewer')
    qt.create_app()
    qcw = GraphWindow(wheel=wheel)
    qcw.setWindowTitle(title)
    if qc is not None:
        qcw.updateDataframe(qc)
    qcw.show()
    return qcw
