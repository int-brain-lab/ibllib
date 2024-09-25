"""An interactive PyQT QC data frame."""
import logging

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtProperty, Qt, QVariant, QAbstractTableModel, QModelIndex, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import pandas as pd
import numpy as np

from ibllib.misc import qt

_logger = logging.getLogger(__name__)


class DataFrameTableModel(QAbstractTableModel):
    DtypeRole = Qt.UserRole + 1000
    ValueRole = Qt.UserRole + 1001

    def __init__(self, parent=None, dataFrame: pd.DataFrame = pd.DataFrame()):
        super(DataFrameTableModel, self).__init__(parent)
        self._dataframe = dataFrame

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = pyqtProperty(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    @pyqtSlot(int, Qt.Orientation, result=str)
    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return QVariant()

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=Qt.DisplayRole):
        if (not index.isValid() or not (0 <= index.row() < self.rowCount() and
                                        0 <= index.column() < self.columnCount())):
            return QVariant()
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        val = self._dataframe.iloc[row][col]
        if role == Qt.DisplayRole:
            return str(val)
        elif role == DataFrameTableModel.ValueRole:
            return val
        if role == DataFrameTableModel.DtypeRole:
            return dt
        return QVariant()

    def roleNames(self):
        roles = {
            Qt.DisplayRole: b'display',
            DataFrameTableModel.DtypeRole: b'dtype',
            DataFrameTableModel.ValueRole: b'value'
        }
        return roles

    def sort(self, col, order):
        """
        Sort table by given column number.

        :param col: the column number selected (between 0 and self._dataframe.columns.size)
        :param order: the order to be sorted, 0 is descending; 1, ascending
        :return:
        """
        if self._dataframe.empty:
            return
        self.layoutAboutToBeChanged.emit()
        col_name = self._dataframe.columns.values[col]
        # print('sorting by ' + col_name)
        self._dataframe.sort_values(by=col_name, ascending=not order, inplace=True)
        self._dataframe.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()


class PlotCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, wheel=None):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(
            self,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        if wheel:
            self.ax, self.ax2 = fig.subplots(
                2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        else:
            self.ax = fig.add_subplot(111)
        self.draw()


class PlotWindow(QtWidgets.QWidget):
    def __init__(self, parent=None, wheel=None):
        QtWidgets.QWidget.__init__(self, parent=None)
        self.canvas = PlotCanvas(wheel=wheel)
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
        self.vbl.addWidget(NavigationToolbar2QT(self.canvas, self))


class GraphWindow(QtWidgets.QWidget):
    def __init__(self, parent=None, wheel=None):
        QtWidgets.QWidget.__init__(self, parent=parent)
        self.lineEditPath = QtWidgets.QLineEdit(self)

        self.pushButtonLoad = QtWidgets.QPushButton("Select File", self)
        self.pushButtonLoad.clicked.connect(self.loadFile)

        self.tableModel = DataFrameTableModel(self)
        self.tableView = QtWidgets.QTableView(self)
        self.tableView.setModel(self.tableModel)
        self.tableView.setSortingEnabled(True)
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
            self, "Open File", "", "CSV Files (*.csv)")
        self.lineEditPath.setText(fileName)
        df = pd.read_csv(fileName)
        self.updateDataframe(df)

    def updateDataframe(self, dataFrame: pd.DataFrame):
        self.tableModel.setDataFrame(dataFrame)

    def tv_double_clicked(self):
        df = self.tableView.model()._dataframe
        ind = self.tableView.currentIndex()
        start = df.loc[ind.row()]['intervals_0']
        finish = df.loc[ind.row()]['intervals_1']
        dt = finish - start
        if self.wheel:
            idx = np.searchsorted(
                self.wheel['re_ts'], np.array([start - dt / 10, finish + dt / 10]))
            period = self.wheel['re_pos'][idx[0]:idx[1]]
            if period.size == 0:
                _logger.warning('No wheel data during trial #%i', ind.row())
            else:
                min_val, max_val = np.min(period), np.max(period)
                self.wplot.canvas.ax2.set_ylim(min_val - 1, max_val + 1)
            self.wplot.canvas.ax2.set_xlim(start - dt / 10, finish + dt / 10)
        self.wplot.canvas.ax.set_xlim(start - dt / 10, finish + dt / 10)

        self.wplot.canvas.draw()


def viewqc(qc=None, title=None, wheel=None):
    qt.create_app()
    qcw = GraphWindow(wheel=wheel)
    qcw.setWindowTitle(title)
    if qc is not None:
        qcw.updateDataframe(qc)
    qcw.show()
    return qcw
