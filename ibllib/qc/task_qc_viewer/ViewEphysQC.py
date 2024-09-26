"""An interactive PyQT QC data frame."""

import logging

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtProperty, Qt, QVariant, QAbstractTableModel, QModelIndex, \
    QObject, QPoint, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMenu, QAction, QHeaderView
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import pandas as pd
import numpy as np

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
            val = self._dataframe.iloc[index.row()][index.column()]
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
    _rgba: np.ndarray
    _cmap: ListedColormap

    def __init__(self, parent: QObject = ..., dataFrame: pd.DataFrame | None = None,
                 colorMap: ListedColormap | None = None, alpha: float = 0.5):
        super().__init__(parent=parent, dataFrame=dataFrame)

        self._alpha = alpha
        if colorMap is None:
            self._cmap = plt.get_cmap('plasma')
            self._cmap.set_bad(color='w')
        else:
            self._cmap = colorMap

        self._setRgba()
        self.modelReset.connect(self._setRgba)
        self.dataChanged.connect(self._setRgba)

    def _setRgba(self):
        df = self._dataframe.copy()
        if df.empty:
            self._rgba = df
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

        # store color values to ndarray
        self._rgba = self._cmap(df, self._alpha, True)

    def data(self, index, role=...):
        if not index.isValid():
            return QVariant()
        if role == Qt.BackgroundRole:
            row = self._dataframe.index[index.row()]
            return QColor.fromRgb(*self._rgba[row][index.column()])
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

        self.columnPinned = pyqtSignal(int, bool)

        self.lineEditFilter = QtWidgets.QLineEdit(self)
        self.lineEditFilter.setPlaceholderText('Filter columns by name')
        self.lineEditFilter.textChanged.connect(self.changeFilter)

        self.pushButtonLoad = QtWidgets.QPushButton("Select File", self)
        self.pushButtonLoad.clicked.connect(self.loadFile)

        self.tableModel = ColoredDataFrameTableModel(self)
        self.tableView = QtWidgets.QTableView(self)
        self.tableView.setModel(self.tableModel)
        self.tableView.setSortingEnabled(True)
        self.tableView.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.tableView.horizontalHeader().setSectionsMovable(True)
        self.tableView.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableView.horizontalHeader().customContextMenuRequested.connect(
            self.contextMenu)
        self.tableView.doubleClicked.connect(self.tv_double_clicked)

        self.pinAction = QAction('Pin column', self)
        self.pinAction.setCheckable(True)
        self.pinAction.toggled.connect(self.pinColumn)

        vLayout = QtWidgets.QVBoxLayout(self)
        hLayout = QtWidgets.QHBoxLayout()
        hLayout.addWidget(self.lineEditFilter)
        hLayout.addWidget(self.pushButtonLoad)
        vLayout.addLayout(hLayout)
        vLayout.addWidget(self.tableView)

        self.wplot = PlotWindow(wheel=wheel)
        self.wplot.show()
        self.tableModel.dataChanged.connect(self.wplot.canvas.draw)

        self.wheel = wheel

    def contextMenu(self, pos: QPoint):
        idx = self.sender().logicalIndexAt(pos)
        action = self.pinAction
        action.setData(idx)
        action.setChecked(idx in self._pinnedColumns)
        menu = QMenu(self)
        menu.addAction(action)
        menu.exec(self.mapToParent(pos))

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
    qt.create_app()
    qcw = GraphWindow(wheel=wheel)
    qcw.setWindowTitle(title)
    if qc is not None:
        qcw.updateDataframe(qc)
    qcw.show()
    return qcw
