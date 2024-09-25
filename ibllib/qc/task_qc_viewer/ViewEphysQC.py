"""An interactive PyQT QC data frame."""

import logging

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtProperty, Qt, QVariant, QAbstractTableModel, QModelIndex, QObject
from PyQt5.QtGui import QBrush, QColor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import pandas as pd
import numpy as np

from ibllib.misc import qt

_logger = logging.getLogger(__name__)


class DataFrameTableModel(QAbstractTableModel):
    def __init__(self, parent: QObject = ..., dataFrame: pd.DataFrame = pd.DataFrame()):
        super().__init__(parent)
        self._dataframe = dataFrame

    def setDataFrame(self, dataFrame: pd.DataFrame):
        self.beginResetModel()
        self._dataframe = dataFrame.copy()
        self.endResetModel()

    def dataFrame(self) -> pd.DataFrame:
        return self._dataframe

    dataFrame = pyqtProperty(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...):
        if role == Qt.DisplayRole:
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
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        val = self._dataframe.iloc[row][col]
        if role == Qt.DisplayRole:
            if isinstance(val, np.generic):
                return val.item()
            return QVariant(str(val))
        return QVariant()

    def sort(self, column: int, order: Qt.SortOrder = ...):
        if self.columnCount() == 0:
            return
        self.layoutAboutToBeChanged.emit()
        col_name = self._dataframe.columns.values[column]
        self._dataframe.sort_values(by=col_name, ascending=not order, inplace=True)
        self._dataframe.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()


class ColoredDataFrameTableModel(DataFrameTableModel):
    _colors: pd.DataFrame
    _cmap = plt.get_cmap('plasma')
    _alpha = 0.5

    def __init__(self, parent: QObject = ..., dataFrame: pd.DataFrame = pd.DataFrame()):
        super().__init__(parent=parent, dataFrame=dataFrame)
        self._setColors()
        self.modelReset.connect(self._setColors)
        self.dataChanged.connect(self._setColors)
        self.layoutChanged.connect(self._setColors)

    def _setColors(self):
        df = self._dataframe.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in df.select_dtypes(include=['bool']):
            df[col] = df[col].astype(float)
        for col in df.select_dtypes(exclude=['bool']):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            if df[col].nunique() == 1:
                df[col] = QColor.fromRgb(*self._cmap(0, self._alpha, True))
            else:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                df[col] = [QColor.fromRgb(*x) for x in self._cmap(df[col], self._alpha, True).tolist()]
        self._colors = df

    def data(self, index, role=...):
        if not index.isValid():
            return QVariant()
        if role == Qt.BackgroundRole:
            row = self._dataframe.index[index.row()]
            col = self._dataframe.columns[index.column()]
            val = self._dataframe.iloc[row][col]
            if isinstance(val, (np.bool_, np.number)) and not np.isnan(val):
                return self._colors.iloc[row][col]
            else:
                return QBrush(Qt.white)
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
    def __init__(self, parent=None, wheel=None):
        QtWidgets.QWidget.__init__(self, parent=parent)
        self.lineEditPath = QtWidgets.QLineEdit(self)

        self.pushButtonLoad = QtWidgets.QPushButton("Select File", self)
        self.pushButtonLoad.clicked.connect(self.loadFile)

        self.tableModel = ColoredDataFrameTableModel(self)
        self.tableView = QtWidgets.QTableView(self)
        self.tableView.setModel(self.tableModel)
        self.tableView.setSortingEnabled(True)
        self.tableView.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.tableView.doubleClicked.connect(self.tv_double_clicked)

        vLayout = QtWidgets.QVBoxLayout(self)
        hLayout = QtWidgets.QHBoxLayout()
        hLayout.addWidget(self.lineEditPath)
        hLayout.addWidget(self.pushButtonLoad)
        vLayout.addLayout(hLayout)
        vLayout.addWidget(self.tableView)

        self.wplot = PlotWindow(wheel=wheel)
        self.wplot.show()
        self.tableModel.dataChanged.connect(self.wplot.canvas.draw)

        self.wheel = wheel

    def loadFile(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open File", "", "CSV Files (*.csv)"
        )
        if len(fileName) == 0:
            return
        self.lineEditPath.setText(fileName)
        df = pd.read_csv(fileName)
        self.updateDataframe(df)

    def updateDataframe(self, dataFrame: pd.DataFrame):
        self.tableModel.setDataFrame(dataFrame)

    def tv_double_clicked(self):
        ind = self.tableView.currentIndex()
        data = self.tableModel.dataFrame.loc[ind.row()]
        t0 = data["intervals_0"]
        t1 = data["intervals_1"]
        dt = t1 - t0
        if self.wheel:
            idx = np.searchsorted(self.wheel["re_ts"], np.array([t0 - dt / 10, t1 + dt / 10]))
            period = self.wheel["re_pos"][idx[0] : idx[1]]
            if period.size == 0:
                _logger.warning("No wheel data during trial #%i", ind.row())
            else:
                min_val, max_val = np.min(period), np.max(period)
                self.wplot.canvas.ax2.set_ylim(min_val - 1, max_val + 1)
            self.wplot.canvas.ax2.set_xlim(t0 - dt / 10, t1 + dt / 10)
        self.wplot.canvas.ax.set_xlim(t0 - dt / 10, t1 + dt / 10)

        self.wplot.canvas.draw()


def viewqc(qc=None, title=None, wheel=None):
    qt.create_app()
    qcw = GraphWindow(wheel=wheel)
    qcw.setWindowTitle(title)
    if qc is not None:
        qcw.updateDataframe(qc)
    qcw.show()
    return qcw
