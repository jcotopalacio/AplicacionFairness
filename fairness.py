import sys
import time

from PyQt5.QtCore import QUrl, QRect
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, \
    QCheckBox, QGridLayout, QTableWidget, QTableWidgetItem, QComboBox, QFrame, QRadioButton, QSpinBox, QScrollArea, \
    QTabWidget, QDoubleSpinBox, QLineEdit, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QCursor
from PyQt5.QtWebEngineWidgets import *
import pandas as pd
from fairlearn.reductions import GridSearch, DemographicParity, ErrorRate
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from raiwidgets import FairnessDashboard
from sklearn.impute import KNNImputer
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import svm, metrics
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
from scipy import stats
import os

dataset = pd.DataFrame()
class_attribute = ''
new_dataset = pd.DataFrame()
name_dataset = ''
dataset_bar_plot = ''
sensitive_attr = ''
result_columns = dict()
X_train, X_test, y_train, y_test, A_train, A_test = pd.DataFrame(), pd.DataFrame(), pd.Series, pd.Series, pd.Series, pd.Series

# Predictions
y_predict_RF = []
y_predict_gbm = []
y_predict_svm = []
y_predict_xgb = []

# Parameters
rf_parameters = []
gbm_parameters = []
svm_parameters = []
xgb_parameters = []

# Unmitigated Classifiers
rf_unmitigated = ''
gbm_unmitigated = ''
svm_unmitigated = ''
xgb_unmitigated = ''

style_title = "font-family: Times New Roman; font-size: 22px; color: 'white'; background: #814D1F; border: 2px solid #814D1F;"
style_param_spin = "border: 2px solid #814D1F; border-radius: 5px; font-size: 14px; padding: 5px; " \
                   "margin-left: 10px; margin-right: 10 px;"
style_combo_box = """
        margin-left: 8px; margin-right: 8px;}
        QComboBox {
            font-size: 14px;
            border: 2px solid #814D1F; 
            border-radius: 8px;       
            padding: 5px; 
        }
        
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 30px;
            border-left: 1px solid #814D1F; 
        } 
        QComboBox::down-arrow
        {
            background-color : #814D1F;
        }      
        QComboBox QAbstractItemView{
            background-color: #C9C9C9;
            border: 2px solid #814D1F;
            border-radius: 8px; 
            selection-background-color:  #814D1F;
                 
        }
        """
widgets = {
    'logo': [],
    'question': [],
    'students': [],
    'car': [],
    'law': [],
    'title_attributes': [],
    'title': [],
    'columns': [],
    'instances': [],
    'graph_class': [],
    'button4': [],
    'button5': [],
    'missing': [],
    'imputation_label': [],
    'imputation_attr': [],
    'list_widget': [],
    'cs1': [],
    'k_neighbors': [],
    'weights': [],
    'button_apply': [],
    'button_rever': [],
    'button_back_f3': [],
    'button_next_f3': [],
    'graphs_label': [],
    'button_corr': [],
    'button_violin': [],
    'button_bar': [],
    'button_cross': [],
    'button_point': []
}


class AttrSelection(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    grid = QGridLayout()
    attributes = []

    def __init__(self):
        super().__init__()
        self.setFixedWidth(700)
        self.setFixedHeight(700)
        self.setWindowTitle("Selección de Atributos")
        self.setStyleSheet("background: #C9C9C9;")
        app_icon = QtGui.QIcon()
        # 'https://pngtree.com/freepng/court-law-is-fair-and-just_4651728.html'>png image from pngtree.com/
        app_icon.addFile("images/logo.png")
        self.setWindowIcon(app_icon)
        self.setLayout(self.grid)

        self.draw_widgets()

    def draw_widgets(self):
        # Title
        graphs_label = QLabel("Selección de atributos (SelectKBest)")
        graphs_label.setAlignment(QtCore.Qt.AlignCenter)
        graphs_label.setWordWrap(True)
        graphs_label.setStyleSheet(
            style_title
        )
        widgets["graphs_label"].append(graphs_label)

        # Feature Selection text
        frame = QFrame()
        box_h = QVBoxLayout()
        style = "font-size: 18px;"

        label = QLabel("Seleccione la función de puntuación y el valor del parámetro k")
        label.setStyleSheet(style)
        box_h.addWidget(label)

        radio_box = QHBoxLayout()
        check_box_1 = QRadioButton("f_classif")
        check_box_2 = QRadioButton("chi2")
        check_box_1.setStyleSheet(style)
        check_box_2.setStyleSheet(style)
        check_box_1.setChecked(True)

        radio_box.addWidget(check_box_1)
        radio_box.addWidget(check_box_2)
        box_h.addLayout(radio_box)

        l1 = QLabel("k = ")
        l1.setStyleSheet(style)
        l1.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        k_value = QSpinBox()
        k_value.setMinimum(1)
        k_value.setMaximum(len(X_train.columns))
        k_value.setStyleSheet(style_param_spin)
        radio_box.addWidget(l1)
        radio_box.addWidget(k_value)

        frame.setLayout(box_h)
        widgets['graph_class'].append(frame)

        # Button Apply
        button_apply = create_mini_buttons("Aplicar")
        button_apply.clicked.connect(lambda: self.apply_attr_sel())
        widgets['button_apply'].append(button_apply)

        # Show results
        area = QFrame()
        area.setStyleSheet(
            "QFrame {border: 2px solid #814D1F;}"
        )

        widgets['cs1'].append(area)

        # Buttons next and back
        accept = create_buttons("Aceptar", 5, 150)
        cancel = create_buttons("Cancelar", 5, 150)

        accept.clicked.connect(lambda: self.apply_all_changes())
        cancel.clicked.connect(self.close)

        widgets['button_bar'].append(accept)
        widgets['button_violin'].append(cancel)

        self.add_widgets()

    def add_widgets(self):
        self.grid.addWidget(widgets['graphs_label'][-1], 0, 0, 1, 3)
        self.grid.addWidget(widgets['graph_class'][-1], 1, 0, 1, 3)
        self.grid.addWidget(widgets['button_apply'][-1], 2, 1, 1, 2)
        self.grid.addWidget(widgets['cs1'][-1], 3, 0, 2, 3)
        self.grid.addWidget(widgets['button_bar'][-1], 5, 0)
        self.grid.addWidget(widgets['button_violin'][-1], 5, 1)

    def apply_attr_sel(self):
        self.clear_area()
        area = widgets['cs1'][-1]
        frame_layout = QVBoxLayout(area)

        radio_inside_frame = widgets['graph_class'][-1].findChildren(QRadioButton)
        score = ''
        for item in radio_inside_frame:
            score = item.text() if item.isChecked() else ''

        spin = widgets['graph_class'][-1].findChildren(QSpinBox)[0].value()

        selection = SelectKBest(score_func=f_classif, k=int(spin)).fit(X_train[:], y_train)
        if score == 'chi2':
            selection = SelectKBest(score_func=chi2, k=int(spin)).fit(X_train[:], y_train)

        attrib = selection.get_support()

        columns = list(X_train)
        attr = [(columns[i], selection.scores_[i]) for i in list(attrib.nonzero()[0])]

        columns = QTableWidget()
        columns.setColumnCount(2)
        columns.setHorizontalHeaderLabels(['Atributos Seleccionados', 'Puntuación'])
        columns.setColumnWidth(0, 400)
        columns.setColumnWidth(1, 209)
        columns.setStyleSheet(
            "font-size: 16px;"
            "color: 'black';"
            "border-style: outset;"
            "border: 2px solid #814D1F;"
        )
        stylesheet = "::section{Background-color: #814D1F;color: white;}"
        columns.horizontalHeader().setStyleSheet(
            stylesheet
        )
        columns.verticalHeader().setFixedWidth(35)
        columns.verticalHeader().setStyleSheet(stylesheet)
        index = 0
        self.attributes.clear()
        for value in attr:
            item = QTableWidgetItem(str(value[0]))
            self.attributes.append(str(value[0]))
            item_2 = QTableWidgetItem(str(round(float(value[1]), 3)))
            item.setTextAlignment(QtCore.Qt.AlignLeft)
            item_2.setTextAlignment(QtCore.Qt.AlignRight)
            columns.insertRow(index)
            columns.setItem(index, 0, item)
            columns.setItem(index, 1, item_2)
            index += 1

        frame_layout.addWidget(columns)

    def clear_area(self):
        if len(widgets['cs1']) != 0:
            widgets['cs1'][-1].hide()
            for i in range(0, len(widgets['cs1'])):
                widgets['cs1'].pop()
            area = QFrame()
            area.setStyleSheet(
                "QFrame {border: 2px solid #814D1F;}"
            )
            widgets['cs1'].append(area)
            self.grid.addWidget(widgets['cs1'][-1], 3, 0, 2, 3)

    def apply_all_changes(self):
        global X_train, X_test
        X_train = X_train.get(self.attributes)
        X_test = X_test.get(self.attributes)
        self.close()


class AnotherWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    grid = QGridLayout()

    def __init__(self, type_graph, data, attr=None):
        super().__init__()
        if type_graph == 'correlograma':
            self.setWindowTitle("Correlograma")
            self.paint_corr(data)
        elif type_graph == 'bar':
            self.setWindowTitle("Gráfico de Barra")
            self.paint_bar(data, attr)
        elif type_graph == 'box':
            self.setWindowTitle("Gráfico de Cajas")
            self.paint_box(data)
        elif type_graph == 'cross':
            self.setWindowTitle("Gráfico de Tabulación Cruzada")
            self.paint_cross(data)
        else:
            self.setWindowTitle("Gráfico de Violín")
            self.paint_violin(data)

        self.setStyleSheet("background: #C9C9C9;")
        app_icon = QtGui.QIcon()
        # <a href='https://pngtree.com/freepng/court-law-is-fair-and-just_4651728.html'>png image from pngtree.com/</a>
        app_icon.addFile("images/logo.png")
        self.setWindowIcon(app_icon)
        self.setLayout(self.grid)

    def paint_corr(self, data):
        fig_size = (20, 20) if name_dataset == 'students' else (10, 10)
        fig = Figure(figsize=fig_size)
        canvas = FigureCanvas(fig)
        canvas.figure.set_facecolor('#C9C9C9')
        canvas.figure.clf()
        # Crear un subplot en la figura
        ax = canvas.figure.add_subplot(111)
        ax.set_facecolor('#C9C9C9')
        corr_matrix = data.corr('pearson')
        # Crear el correlograma usando seaborn
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, mask=mask)

        canvas.figure.tight_layout(pad=0)

        scroll_area = QScrollArea()
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scroll_area.setWidget(canvas)

        widgets['cs1'].append(scroll_area)
        self.grid.addWidget(widgets["cs1"][-1], 0, 0)

        button_save = create_mini_buttons('Salvar el gráfico', 150, 150)
        button_save.clicked.connect(lambda: self.save_plot(canvas))
        widgets['button4'].append(button_save)
        self.grid.addWidget(widgets['button4'][-1], 1, 0)

    def paint_bar(self, data, attr):
        values = data.value_counts()
        index = data.value_counts().index

        graph_class = FigureCanvas(Figure(figsize=(5, 3)))
        graph_class.figure.set_facecolor('#C9C9C9')
        ax = graph_class.figure.subplots()
        bar_container = ax.bar(index, values, color=['#814D1F', '#CC5C53', '#449C2A', '#3282F6', '#8A2094', '#FFFE91'],
                               edgecolor=['#814D1F'])
        ax.set(title=str(attr), ylim=(0, len(dataset)))
        ax.bar_label(bar_container, fmt='{:,.0f}')
        ax.set_xticks(data.unique())
        ax.set_facecolor('#C9C9C9')
        widgets['cs1'].append(graph_class)
        self.grid.addWidget(widgets["cs1"][-1], 0, 0)

        button_save = create_mini_buttons('Salvar el gráfico', 150, 150)
        button_save.clicked.connect(lambda: self.save_plot(graph_class))
        widgets['button4'].append(button_save)
        self.grid.addWidget(widgets['button4'][-1], 1, 0)

    def paint_box(self, data):
        rows = math.ceil(len(data.columns) / 4)
        fig_size = (15, rows * 5)
        graph_class = FigureCanvas(Figure(figsize=fig_size))
        graph_class.figure.set_facecolor('#C9C9C9')
        graph_class.figure.clf()
        i = 1
        for column in data.columns:
            ax = graph_class.figure.add_subplot(rows, 4, i)
            sns.boxplot(data[column], ax=ax)
            ax.set_facecolor('#C9C9C9')
            i += 1

        graph_class.figure.tight_layout(pad=0)

        scroll_area = QScrollArea()
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scroll_area.setStyleSheet("padding: 0px;")
        scroll_area.setWidget(graph_class)

        widgets['cs1'].append(scroll_area)
        self.grid.addWidget(widgets["cs1"][-1], 0, 0)

        button_save = create_mini_buttons('Salvar el gráfico', 150, 150)
        button_save.clicked.connect(lambda: self.save_plot(graph_class))
        widgets['button4'].append(button_save)
        self.grid.addWidget(widgets['button4'][-1], 1, 0)

    def paint_cross(self, data):
        axis_x = []
        column_ = []
        for col in data[0]:
            axis_x.append(dataset_bar_plot[col])
        for col in data[1]:
            column_.append(dataset_bar_plot[col])

        graph_class = FigureCanvas(Figure(figsize=(8, 5)))
        graph_class.figure.set_facecolor('#C9C9C9')
        graph_class.figure.clf()

        ax = graph_class.figure.subplots()
        pd.crosstab(axis_x, column_).plot(kind='bar', ax=ax)
        ax.set_facecolor('#C9C9C9')
        ax.legend(loc=(1.01, .35))
        graph_class.figure.tight_layout(pad=0)

        widgets['cs1'].append(graph_class)
        self.grid.addWidget(widgets["cs1"][-1], 0, 0)

        button_save = create_mini_buttons('Salvar el gráfico', 150, 150)
        button_save.clicked.connect(lambda: self.save_plot(graph_class))
        widgets['button4'].append(button_save)
        self.grid.addWidget(widgets['button4'][-1], 1, 0)

    def paint_violin(self, data):
        graph_class = FigureCanvas(Figure(figsize=(8, 5)))
        graph_class.figure.set_facecolor('#C9C9C9')
        graph_class.figure.clf()

        ax = graph_class.figure.subplots()
        ax.set_facecolor('#C9C9C9')

        if data['y'] != 'None' and data['hue'] != 'None':
            sns.violinplot(x=data['x'], y=data['y'], hue=data['hue'], data=new_dataset, legend='full', ax=ax)
        elif data['y'] != 'None':
            sns.violinplot(x=data['x'], y=data['y'], data=new_dataset, legend='full', ax=ax)
        elif data['hue'] != 'None':
            sns.violinplot(x=data['x'], hue=data['hue'], data=new_dataset, legend='full', ax=ax)
        else:
            sns.violinplot(x=data['x'], data=new_dataset, legend='full', ax=ax)

        widgets['cs1'].append(graph_class)

        self.grid.addWidget(widgets["cs1"][-1], 0, 0)

        button_save = create_mini_buttons('Salvar el gráfico', 150, 150)
        button_save.clicked.connect(lambda: self.save_plot(graph_class))
        widgets['button4'].append(button_save)
        self.grid.addWidget(widgets['button4'][-1], 1, 0)

    def save_plot(self, figure):
        files_types = "JPG (*.jpg);;PNG (*.png)"
        path, extension = QFileDialog.getSaveFileName(self, "Guardar como", os.getcwd(),
                                                      files_types,
                                                      options=QFileDialog.Options())
        figure.figure.savefig(path)


def drop_duplicates():
    dataset.drop_duplicates(inplace=True)


def drop_na_rows(attr):
    global new_dataset
    new_dataset = new_dataset.dropna(subset=[attr])


def clear_widgets():
    """ hide all existing widgets and erase
        them from the global dictionary"""
    for widget in widgets:
        if widgets[widget] != [] and widgets[widget][-1].isWidgetType():
            widgets[widget][-1].hide()
        elif len(widgets[widget]) != 0:
            for wid in range(widgets[widget][-1].count()):
                widgets[widget][-1].itemAt(wid).widget().setHidden(True)
        for i in range(0, len(widgets[widget])):
            widgets[widget].pop()


def create_buttons(answer, l_margin=None, r_margin=None):
    """create identical buttons with custom left & right margins"""
    button = QPushButton(answer)
    button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
    button.setFixedWidth(485)
    button.setStyleSheet(
        "*{margin-left: " + str(l_margin) + "px;" +
        "margin-right: " + str(r_margin) + "px;" +
        '''
        border: 4px solid '#814D1F';
        color: black;
        font-family: 'shanti';
        font-size: 16px;
        border-radius: 25px;
        padding: 15px 0;
        text-align: center;
        margin-top: 10px}
        *:hover{
            background: '#814D1F';
            color: white;
        }
        QToolTip { 
            color: black; 
            background-color: #F0F0F0; 
            border: None; 
            margin-top: 5px;
            border-radius: 1px;}
        QToolTip:hover{
            background: #F0F0F0;
        }
        '''
    )
    return button


def create_mini_buttons(answer, l_margin=None, r_margin=None):
    """create identical buttons with custom left & right margins"""
    button = QPushButton(answer)
    button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

    button.setStyleSheet(
        "*{margin-left: " + str(l_margin) + "px;" +
        "margin-right: " + str(r_margin) + "px;" +
        '''
        border: 2px solid '#814D1F';
        color: black;
        font-family: 'shanti';
        font-size: 14px bold;
        border-radius: 20px;
        padding: 10px;
        margin-top: 15px;
        text-align: center;}
        *:hover{
            background: '#814D1F';
            color: white;
        }
        QToolTip { 
            color: black; 
            background-color: #F0F0F0; 
            border: None; 
            margin-top: 5px;
            border-radius: 1px;}
        QToolTip:hover{
            background: #F0F0F0;
        }
        '''
    )
    return button


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS2
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class MainWindow(QWidget):
    grid = QGridLayout()
    tiempo_inicio = 0

    def __init__(self):
        super().__init__()
        self.tiempo_inicio = time.time()
        self.setWindowTitle("Aplicación de Fairness")
        estilo_tooltip = ''' background: #C9C9C9;}
                            '''
        self.setStyleSheet(estilo_tooltip)

        app_icon = QtGui.QIcon()
        # <a href='https://pngtree.com/freepng/court-law-is-fair-and-just_4651728.html'>png image from pngtree.com/</a>
        app_icon.addFile(resource_path("images/logo.png"))
        self.setWindowIcon(app_icon)
        self.show_frame1()

    def clear_area(self, pos=2):
        if len(widgets['imputation_attr']) != 0:
            widgets['imputation_attr'][-1].hide()
            for i in range(0, len(widgets['imputation_attr'])):
                widgets['imputation_attr'].pop()
            area = QFrame()
            area.setStyleSheet(
                "QFrame {border: 2px solid #814D1F;}"
            )
            widgets['imputation_attr'].append(area)
            self.grid.addWidget(widgets['imputation_attr'][-1], pos, 0, 5 - pos, 4)

    def mitigation(self):

        method = widgets['list_widget'][-1].itemAt(1).widget().currentText()
        grid_size = widgets['list_widget'][-1].itemAt(3).widget().value()

        sweep = ''
        unmitigated = ''
        if method == 'LightGBM' and len(gbm_parameters) != 0:
            unmitigated = gbm_unmitigated
            sweep = GridSearch(HistGradientBoostingClassifier(learning_rate=gbm_parameters[0],
                                                              max_leaf_nodes=gbm_parameters[2],
                                                              min_samples_leaf=gbm_parameters[3],
                                                              max_bins=gbm_parameters[5],
                                                              l2_regularization=gbm_parameters[4],
                                                              max_iter=gbm_parameters[1]),
                               constraints=DemographicParity(),
                               grid_size=grid_size)
        elif method == 'Random Forest' and len(rf_parameters) != 0:
            unmitigated = rf_unmitigated
            sweep = GridSearch(RandomForestClassifier(n_estimators=rf_parameters[0],
                                                      max_depth=rf_parameters[1],
                                                      min_samples_split=rf_parameters[2],
                                                      criterion=rf_parameters[3]),
                               constraints=DemographicParity(),
                               grid_size=grid_size)
        elif method == 'SVM' and len(svm_parameters) != 0:
            unmitigated = svm_unmitigated
            sweep = GridSearch(svm.SVC(C=svm_parameters[0],
                                       kernel=svm_parameters[1],
                                       gamma=svm_parameters[2]),
                               constraints=DemographicParity(),
                               grid_size=grid_size)
        elif method == 'XGBoost' and len(xgb_parameters) != 0:
            unmitigated = xgb_unmitigated
            sweep = GridSearch(XGBClassifier(subsample=xgb_parameters[0],
                                             max_depth=xgb_parameters[1],
                                             colsample_bytree=xgb_parameters[2],
                                             learning_rate=xgb_parameters[3],
                                             n_estimators=xgb_parameters[4]),
                               constraints=DemographicParity(),
                               grid_size=grid_size)

        sweep.fit(X_train, y_train, sensitive_features=A_train)
        predictors = sweep.predictors_

        errors, disparities = [], []

        for m in predictors:
            classifier = lambda x: m.predict(x)

            error = ErrorRate()
            error.load_data(X_train, pd.Series(y_train), sensitive_features=A_train)
            disparity = DemographicParity()
            disparity.load_data(X_train, pd.Series(y_train), sensitive_features=A_train)

            errors.append(error.gamma(classifier)[0])
            disparities.append(disparity.gamma(classifier).max())

        all_results = pd.DataFrame({'predictor': predictors, 'error': errors, 'disparity': disparities})

        all_models_dict = {'census_unmitigated': unmitigated}
        dominant_models_dict = {'census_unmitigated': unmitigated}
        base_name_format = 'census_grid_model_{0}'
        row_id = 0
        for row in all_results.itertuples():
            model_name = base_name_format.format(row_id)
            all_models_dict[model_name] = row.predictor
            errors_for_lower_oreq_disparity = all_results['error'][all_results['disparity'] <= row.disparity]
            if row.error <= errors_for_lower_oreq_disparity.min():
                dominant_models_dict[model_name] = row.predictor
            row_id = row_id + 1

        dashboard_all = dict()
        for name, predictor in all_models_dict.items():
            value = predictor.predict(X_test)
            dashboard_all[name] = value

        dashboard = FairnessDashboard(sensitive_features=A_test,
                                      y_true=y_test,
                                      y_pred=dashboard_all)

        configuration = dashboard.config

        view = QWebEngineView()
        view.load(QUrl(configuration['baseUrl']))

        if len(widgets['cs1']) != 0:
            widgets['cs1'][-1].hide()
            widgets['cs1'].pop()
        widgets['cs1'].append(view)

        label = QLineEdit(str(configuration['baseUrl']))
        label.setStyleSheet(style_param_spin)

        if len(widgets['graphs_label']) != 0:
            widgets['graphs_label'][-1].hide()
            widgets['graphs_label'].pop()
        widgets['graphs_label'].append(label)

        self.grid.addWidget(widgets['graphs_label'][-1], 2, 0, 1, 2)
        self.grid.addWidget(widgets['cs1'][-1], 3, 0, 2, 4)

    def add_widgets_f8(self):
        self.grid.addWidget(widgets['title'][-1], 0, 0, 1, 4)
        self.grid.addLayout(widgets['list_widget'][-1], 1, 0, 1, 4)
        self.grid.addWidget(widgets['button_apply'][-1], 2, 3, 1, 2)
        self.grid.addWidget(widgets['button_back_f3'][-1], 5, 0, 1, 2)
        self.grid.addWidget(widgets['button_next_f3'][-1], 5, 2, 1, 2)

    def frame8(self, ruta, name, name_widget):
        clear_widgets()

        self.setFixedWidth(900)
        self.setFixedHeight(710)

        # Title
        graphs_label = QLabel("Mitigación Fairness")
        graphs_label.setAlignment(QtCore.Qt.AlignCenter)
        graphs_label.setWordWrap(True)
        graphs_label.setStyleSheet(
            style_title
        )
        widgets['title'].append(graphs_label)

        qbox = QHBoxLayout()
        label = QLabel("Seleccione un clasificador")
        label.setStyleSheet("font-size: 16px;")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        combo_box = QComboBox()
        combo_box.addItems(['LightGBM', 'Random Forest', 'SVM', 'XGBoost'])
        combo_box.setStyleSheet(style_combo_box)
        qbox.addWidget(label)
        qbox.addWidget(combo_box)

        label_spin = QLabel("grid_size:")
        label_spin.setStyleSheet("font-size: 16px;")
        label_spin.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_grid_size = QSpinBox()
        spin_grid_size.setStyleSheet(style_param_spin)
        spin_grid_size.setMinimum(1)
        spin_grid_size.setValue(15)
        qbox.addWidget(label_spin)
        qbox.addWidget(spin_grid_size)

        button = create_mini_buttons("Aplicar")

        widgets['button_apply'].append(button)

        widgets['list_widget'].append(qbox)
        button.clicked.connect(lambda: self.mitigation())

        button_back_f3 = create_buttons("Atrás", 20, 60)
        button_next_f3 = create_buttons("Finalizar", 20, 60)

        button_back_f3.clicked.connect(lambda: self.frame7(ruta, name, name_widget))
        button_next_f3.clicked.connect(lambda: self.close())

        widgets['button_back_f3'].append(button_back_f3)
        widgets['button_next_f3'].append(button_next_f3)

        self.add_widgets_f8()

    def fairness(self, text):
        y_predict = []
        if text == 'SVM':
            y_predict = y_predict_svm
        elif text == 'XGBoost':
            y_predict = y_predict_xgb
        elif text == 'LightGBM':
            y_predict = y_predict_gbm
        elif text == 'Random Forest':
            y_predict = y_predict_RF

        if len(y_predict) != 0:
            ax = FairnessDashboard(sensitive_features=A_test,
                                   y_true=y_test,
                                   y_pred={"unmitigated": y_predict})

            configuration = ax.config

            view = QWebEngineView()
            view.load(QUrl(configuration['baseUrl']))

            if len(widgets['cs1']) != 0:
                widgets['cs1'][-1].hide()
                widgets['cs1'].pop()
            widgets['cs1'].append(view)

            label = QLineEdit(str(configuration['baseUrl']))
            label.setStyleSheet(style_param_spin)

            if len(widgets['graphs_label']) != 0:
                widgets['graphs_label'][-1].hide()
                widgets['graphs_label'].pop()
            widgets['graphs_label'].append(label)

            self.grid.addWidget(widgets['graphs_label'][-1], 1, 2, 1, 2)
            self.grid.addWidget(widgets['cs1'][-1], 2, 0, 3, 4)

    def change_fairness(self, classifier):
        text = classifier.currentText()
        self.fairness(text)

    def add_widgets_f7(self):
        self.grid.addWidget(widgets['title'][-1], 0, 0, 1, 4)
        self.grid.addLayout(widgets['list_widget'][-1], 1, 0, 1, 2)
        self.grid.addWidget(widgets['button_back_f3'][-1], 5, 0, 1, 2)
        self.grid.addWidget(widgets['button_next_f3'][-1], 5, 2, 1, 2)

    def frame7(self, ruta, name, name_widget):
        clear_widgets()

        self.setFixedWidth(900)
        self.setFixedHeight(710)

        # Title
        graphs_label = QLabel("Fairness")
        graphs_label.setAlignment(QtCore.Qt.AlignCenter)
        graphs_label.setWordWrap(True)
        graphs_label.setStyleSheet(
            style_title
        )
        widgets['title'].append(graphs_label)

        qbox = QHBoxLayout()
        label = QLabel("Seleccione un clasificador")
        label.setStyleSheet("font-size: 18px;")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        combo_box = QComboBox()
        combo_box.addItems(['LightGBM', 'Random Forest', 'SVM', 'XGBoost'])
        combo_box.setStyleSheet(style_combo_box)
        qbox.addWidget(label)
        qbox.addWidget(combo_box)

        combo_box.activated.connect(lambda: self.change_fairness(combo_box))

        widgets['list_widget'].append(qbox)

        algorithm = ''
        if len(y_predict_gbm) != 0:
            algorithm = 'LightGBM'
        elif len(y_predict_RF) != 0:
            algorithm = 'Random Forest'
            combo_box.setCurrentText('Random Forest')
        elif len(y_predict_svm) != 0:
            algorithm = 'SVM'
            combo_box.setCurrentText('SVM')
        elif len(y_predict_xgb) != 0:
            algorithm = 'XGBoost'
            combo_box.setCurrentText('XGBoost')
        self.fairness(algorithm)

        button_back_f3 = create_buttons("Atrás", 20, 60)
        button_next_f3 = create_buttons("Mitigar", 20, 60)

        button_back_f3.clicked.connect(lambda: self.frame6(ruta, name, name_widget))
        button_next_f3.clicked.connect(lambda: self.frame8(ruta, name, name_widget))

        widgets['button_back_f3'].append(button_back_f3)
        widgets['button_next_f3'].append(button_next_f3)

        self.add_widgets_f7()

    def clear_area_classifiers(self, pos, pos_area, pos_row=2):
        item = widgets['graph_class'][-1].widget(pos).findChildren(QGridLayout)[0].itemAt(pos_area)
        item.widget().hide()
        area = QFrame()
        area.setStyleSheet(
            "QFrame {border: 2px solid #814D1F;}"
        )
        widgets['graph_class'][-1].widget(pos).findChildren(QGridLayout)[0].removeItem(
            widgets['graph_class'][-1].widget(pos)
            .findChildren(QGridLayout)[0].itemAt(pos_area))
        widgets['graph_class'][-1].widget(pos).findChildren(QGridLayout)[0].addWidget(area, pos_row, 0, 2, 4)

    def draw_table_metrics(self, area, y_predict):
        accuracy = QTableWidgetItem('{:.4f}'.format(metrics.accuracy_score(y_true=y_test, y_pred=y_predict)))
        accuracy.setTextAlignment(QtCore.Qt.AlignCenter)
        precision = QTableWidgetItem('{:.4f}'.format(metrics.precision_score(y_true=y_test, y_pred=y_predict)))
        precision.setTextAlignment(QtCore.Qt.AlignCenter)
        recall = QTableWidgetItem('{:.4f}'.format(metrics.recall_score(y_true=y_test, y_pred=y_predict)))
        recall.setTextAlignment(QtCore.Qt.AlignCenter)
        f1 = QTableWidgetItem('{:.4f}'.format(metrics.f1_score(y_true=y_test, y_pred=y_predict)))
        f1.setTextAlignment(QtCore.Qt.AlignCenter)
        kappa = QTableWidgetItem('{:.4f}'.format(metrics.cohen_kappa_score(y_test, y_predict)))
        kappa.setTextAlignment(QtCore.Qt.AlignCenter)
        auc = QTableWidgetItem('{:.4f}'.format(metrics.roc_auc_score(y_true=y_test, y_score=y_predict)))
        auc.setTextAlignment(QtCore.Qt.AlignCenter)
        confusion_matrix = QTableWidgetItem(str(metrics.confusion_matrix(y_true=y_test, y_pred=y_predict)))
        confusion_matrix.setTextAlignment(QtCore.Qt.AlignCenter)

        frame_layout = QVBoxLayout(area)
        columns = QTableWidget()
        columns.setColumnCount(1)
        columns.setRowCount(7)
        columns.setVerticalHeaderLabels(
            ['Accuracy', 'Precision', 'Recall', 'F1', 'Cohen Kappa', 'ROC_AUC', 'Confusion Matrix'])
        columns.setHorizontalHeaderLabels(['Valores'])
        columns.setColumnWidth(0, 320)
        columns.setRowHeight(6, 150)
        columns.setStyleSheet(
            "font-size: 18px;"
            "color: 'black';"
            "border-style: outset;"
            "border: 2px solid #814D1F;"
            "text-align: center;"
        )

        stylesheet = "::section{Background-color: #814D1F;color: white;}"
        columns.horizontalHeader().setStyleSheet(
            stylesheet
        )
        columns.verticalHeader().setFixedWidth(400)
        columns.verticalHeader().setStyleSheet(stylesheet)

        columns.setItem(0, 0, accuracy)
        columns.setItem(1, 0, precision)
        columns.setItem(2, 0, recall)
        columns.setItem(3, 0, f1)
        columns.setItem(4, 0, kappa)
        columns.setItem(5, 0, auc)
        columns.setItem(6, 0, confusion_matrix)

        frame_layout.addWidget(columns)

    def logistic_regression(self, c, penalty, area):
        logistic = LogisticRegression(C=c, penalty=penalty)
        logistic.fit(X_train, y_train)
        y_predict_LR = logistic.predict(X_test)
        self.draw_table_metrics(area, y_predict_LR)

    def random_forest(self, n_estimators, max_depth, min_samples_split, criterion, area=None):
        rf_parameters.clear()
        rf_parameters.extend([n_estimators, max_depth, min_samples_split, criterion])

        global rf_unmitigated, y_predict_RF
        rf_unmitigated = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                criterion=criterion)
        rf_unmitigated.fit(X_train, y_train)

        y_predict_RF = rf_unmitigated.predict(X_test)

        if area is not None:
            self.draw_table_metrics(area, y_predict_RF)

    def lightGBM(self, learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, l2_regularization, max_bins,
                 area=None):
        gbm_parameters.clear()
        gbm_parameters.extend(
            [learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, l2_regularization, max_bins])  # Save parameters

        global gbm_unmitigated, y_predict_gbm
        gbm_unmitigated = HistGradientBoostingClassifier(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                                                         min_samples_leaf=min_samples_leaf, max_bins=max_bins,
                                                         l2_regularization=l2_regularization, max_iter=max_iter)

        gbm_unmitigated.fit(X_train, y_train)

        y_predict_gbm = gbm_unmitigated.predict(X_test)

        if area is not None:
            self.draw_table_metrics(area, y_predict_gbm)

    def svm(self, c, kernel, gamma, area=None):
        svm_parameters.clear()
        svm_parameters.extend([c, kernel, gamma])

        global svm_unmitigated, y_predict_svm
        svm_unmitigated = svm.SVC(C=c, kernel=kernel, gamma=gamma)
        svm_unmitigated.fit(X_train, y_train)

        y_predict_svm = svm_unmitigated.predict(X_test)

        if area is not None:
            self.draw_table_metrics(area, y_predict_svm)

    def xgboost(self, subsample, max_depth, colsample, learning_rate, n_estimators, area=None):
        xgb_parameters.clear()
        xgb_parameters.extend([subsample, max_depth, colsample, learning_rate, n_estimators])

        global xgb_unmitigated, y_predict_xgb
        xgb_unmitigated = XGBClassifier(subsample=subsample, max_depth=max_depth, colsample_bytree=colsample,
                                        learning_rate=learning_rate,
                                        n_estimators=n_estimators)
        xgb_unmitigated.fit(X_train, y_train)

        y_predict_xgb = xgb_unmitigated.predict(X_test)
        if area is not None:
            self.draw_table_metrics(area, y_predict_xgb)

    def call_classifier(self, pos, best_param):
        if pos == 2:
            learning_rate = float(best_param['learning_rate'])
            max_iter = int(best_param['max_iter'])
            max_leaf_nodes = int(best_param['max_leaf_nodes'])
            min_samples_leaf = int(best_param['min_samples_leaf'])
            l2_regularization = float(best_param['l2_regularization'])
            max_bins = int(best_param['max_bins'])
            self.lightGBM(learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, l2_regularization, max_bins)
        elif pos == 0:
            n_estimators = int(best_param['n_estimators'])
            criterion = best_param['criterion']
            max_depth = int(best_param['max_depth'])
            min_samples_split = int(best_param['min_samples_split'])
            self.random_forest(n_estimators, max_depth, min_samples_split, criterion)
        elif pos == 1:
            learning_rate = float(best_param['learning_rate'])
            max_depth = int(best_param['max_depth'])
            n_estimators = int(best_param['n_estimators'])
            colsample = float(best_param['colsample_bytree'])
            subsample = float(best_param['subsample'])
            self.xgboost(subsample, max_depth, colsample, learning_rate, n_estimators)
        elif pos == 3:
            C = float(best_param['C'])
            kernel = best_param['kernel']
            try:
                gamma = float(best_param['gamma'])
            except:
                gamma = best_param['gamma']
            self.svm(C, kernel, gamma)

    def hyperparam(self, pos, metric, area):
        n = [int(x) for x in np.linspace(1, 32, 32, endpoint=True)]

        grid_models = [
            (RandomForestClassifier(),
             [{'n_estimators': [10, 50, 100, 200, 300, 400, 500], 'criterion': ['gini', 'entropy'],
               'max_depth': n,
               'min_samples_split': n[1:30:3]}]),
            (XGBClassifier(), [{'learning_rate': [0.01, 0.05, 0.1], 'max_depth': n,
                                'n_estimators': [10, 50, 100, 200, 300, 400, 500],
                                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}]),
            (HistGradientBoostingClassifier(),
             [{'learning_rate': [0.01, 0.05, 0.1], 'max_iter': stats.randint(10, 1000),
               'max_leaf_nodes': stats.randint(2, 500), 'min_samples_leaf': stats.randint(2, 300),
               'l2_regularization': stats.uniform(0.0, 100.0), 'max_bins': stats.randint(32, 255)}]),
            (svm.SVC(), [{'C': [0.1, 0.25, 0.5, 0.75, 1, 10, 100], 'kernel': ['linear', 'rbf'],
                          'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001]}])]

        n_iter = 100
        if pos == 3:
            n_iter = 32
        elif pos == 4:
            n_iter = 56
        grid = RandomizedSearchCV(estimator=grid_models[pos][0], param_distributions=grid_models[pos][1],
                                  scoring=metric, n_iter=n_iter, cv=2)
        grid.fit(X_train, y_train)
        best_accuracy = grid.best_score_
        best_param = grid.best_params_

        layout = QVBoxLayout()
        style = "border: None; font-size: 18px; font-family: Times New Roman;"
        text = QLabel('Mejor {} es : {:.2f}%'.format(metric, best_accuracy * 100))
        text.setStyleSheet(style)
        text.setAlignment(QtCore.Qt.AlignCenter)
        split_param = str(best_param).replace(',', '\n')
        text_2 = QLabel('Mejores parámetros: \n' + split_param)
        text_2.setStyleSheet(style)
        text_2.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(text)
        layout.addWidget(text_2)
        area.setLayout(layout)

        self.call_classifier(pos, best_param)

    def apply_XGBoost(self):
        pos = 3
        pos_area = 5
        pos_row = 4

        widgets['button_next_f3'][-1].setEnabled(True)

        self.clear_area_classifiers(pos, pos_area, pos_row=pos_row)
        area = widgets['graph_class'][-1].widget(pos).findChildren(QGridLayout)[0].itemAt(pos_area).widget()
        check_box = widgets['graph_class'][-1].widget(pos).findChildren(QCheckBox)[0]
        combo_box = widgets['graph_class'][-1].widget(pos).findChildren(QComboBox)[0]

        if check_box.isChecked():
            metric = combo_box.currentText()
            self.hyperparam(1, metric, area)
        else:
            children = widgets['graph_class'][-1].widget(pos).findChildren(QDoubleSpinBox)
            subsample = children[0].value()
            max_depth = widgets['graph_class'][-1].widget(pos).findChildren(QSpinBox)[0].value()
            colsample = children[1].value()
            learning_rate = children[2].value()
            n_estimators = widgets['graph_class'][-1].widget(pos).findChildren(QSpinBox)[1].value()

            self.xgboost(subsample, max_depth, colsample, learning_rate, n_estimators, area)

    def apply_RF(self):
        pos = 1
        pos_area = 4
        pos_row = 3

        widgets['button_next_f3'][-1].setEnabled(True)

        self.clear_area_classifiers(pos, pos_area, pos_row=pos_row)
        area = widgets['graph_class'][-1].widget(pos).findChildren(QGridLayout)[0].itemAt(pos_area).widget()
        check_box = widgets['graph_class'][-1].widget(pos).findChildren(QCheckBox)[0]
        combo_box = widgets['graph_class'][-1].widget(pos).findChildren(QComboBox)[1]

        if check_box.isChecked():
            metric = combo_box.currentText()
            self.hyperparam(0, metric, area)
        else:
            n_estimators = widgets['graph_class'][-1].widget(pos).findChildren(QSpinBox)[0].value()
            max_depth = widgets['graph_class'][-1].widget(pos).findChildren(QSpinBox)[1].value()
            if max_depth == 0:
                max_depth = None
            min_samples_split = widgets['graph_class'][-1].widget(pos).findChildren(QDoubleSpinBox)[0].value()
            if min_samples_split > 1:
                min_samples_split = math.ceil(min_samples_split)
            criterion = widgets['graph_class'][-1].widget(pos).findChildren(QComboBox)[0].currentText()

            self.random_forest(n_estimators, max_depth, min_samples_split, criterion, area)

    def apply_LightGBM(self):
        pos = 0
        pos_area = 4
        pos_row = 3
        widgets['button_next_f3'][-1].setEnabled(True)

        self.clear_area_classifiers(pos, pos_area, pos_row)
        area = widgets['graph_class'][-1].widget(pos).findChildren(QGridLayout)[0].itemAt(pos_area).widget()
        check_box = widgets['graph_class'][-1].widget(pos).findChildren(QCheckBox)[0]
        combo_box = widgets['graph_class'][-1].widget(pos).findChildren(QComboBox)[0]

        if check_box.isChecked():
            metric = combo_box.currentText()
            self.hyperparam(2, metric, area)
        else:
            learning_rate = widgets['graph_class'][-1].widget(pos).findChildren(QDoubleSpinBox)[0].value()
            max_iter = widgets['graph_class'][-1].widget(pos).findChildren(QSpinBox)[0].value()
            max_leaf_nodes = widgets['graph_class'][-1].widget(pos).findChildren(QSpinBox)[1].value()
            min_samples_leaf = widgets['graph_class'][-1].widget(pos).findChildren(QSpinBox)[2].value()
            l2_regularization = widgets['graph_class'][-1].widget(pos).findChildren(QDoubleSpinBox)[1].value()
            max_bins = widgets['graph_class'][-1].widget(pos).findChildren(QSpinBox)[3].value()

            self.lightGBM(learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, l2_regularization, max_bins, area)

    def apply_SVM(self):
        pos = 2
        pos_area = 3
        widgets['button_next_f3'][-1].setEnabled(True)

        self.clear_area_classifiers(pos, pos_area)
        area = widgets['graph_class'][-1].widget(pos).findChildren(QGridLayout)[0].itemAt(pos_area).widget()
        check_box = widgets['graph_class'][-1].widget(pos).findChildren(QCheckBox)[0]
        combo_box = widgets['graph_class'][-1].widget(pos).findChildren(QComboBox)[2]

        if check_box.isChecked():
            metric = combo_box.currentText()
            self.hyperparam(3, metric, area)
        else:
            c = widgets['graph_class'][-1].widget(pos).findChildren(QDoubleSpinBox)[0].value()
            kernel = widgets['graph_class'][-1].widget(pos).findChildren(QComboBox)[0].currentText()
            try:
                gamma = float(widgets['graph_class'][-1].widget(pos).findChildren(QComboBox)[1].currentText())
            except:
                gamma = widgets['graph_class'][-1].widget(pos).findChildren(QComboBox)[1].currentText()

            self.svm(c, kernel, gamma, area)

    def self_click_hyper(self, hyper, combo_box):
        check = hyper.sender()
        if check.isChecked():
            combo_box.setEnabled(True)
        else:
            combo_box.setEnabled(False)

    def add_widgets_f6(self):
        self.grid.addWidget(widgets['title'][-1], 0, 0, 1, 4)
        self.grid.addWidget(widgets['graph_class'][-1], 1, 0, 1, 4)
        self.grid.addWidget(widgets['button_back_f3'][-1], 3, 0, 1, 2)
        self.grid.addWidget(widgets['button_next_f3'][-1], 3, 2, 1, 2)

    def frame6(self, ruta, name, name_widget):
        clear_widgets()

        self.setFixedWidth(800)
        self.setFixedHeight(700)

        # Title
        graphs_label = QLabel("Clasificadores")
        graphs_label.setAlignment(QtCore.Qt.AlignCenter)
        graphs_label.setWordWrap(True)
        graphs_label.setStyleSheet(
            style_title
        )
        widgets['title'].append(graphs_label)

        # Classifiers
        tabWidget = QTabWidget()
        tabWidget.setStyleSheet("font-size: 16px;")
        style = "font-size: 16px;"
        text_tool_tip = "Debe marcar la opción Optimizar parámetros para habilitar este componente"

        # LightGBM
        widget_gbm = QWidget()
        widget_gbm.setObjectName("LightGBM")
        tabWidget.addTab(widget_gbm, "LightGBM")

        grid_GBM = QGridLayout()
        box_gbm = QHBoxLayout()

        label = QLabel('  max_leaf_nodes:')
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        combo_metric = QSpinBox()
        combo_metric.setMinimum(1)
        combo_metric.setMaximum(1000)
        combo_metric.setValue(31)
        combo_metric.setStyleSheet(style_param_spin)
        box_gbm.addWidget(label)
        box_gbm.addWidget(combo_metric)

        label = QLabel("     learning_rate:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_learning = QDoubleSpinBox()
        spin_learning.setMinimum(0.0001)
        spin_learning.setMaximum(1)
        spin_learning.setValue(0.1)
        spin_learning.setStyleSheet(style_param_spin)
        box_gbm.addWidget(label)
        box_gbm.addWidget(spin_learning)

        label = QLabel("max_iter:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_iter = QSpinBox()
        spin_iter.setMinimum(1)
        spin_iter.setMaximum(10000)
        spin_iter.setValue(100)
        spin_iter.setStyleSheet(style_param_spin)
        box_gbm.addWidget(label)
        box_gbm.addWidget(spin_iter)

        box_gbm_2 = QHBoxLayout()
        label = QLabel('min_samples_leaf:')
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_sample = QSpinBox()
        spin_sample.setMinimum(1)
        spin_sample.setMaximum(1000)
        spin_sample.setValue(20)
        spin_sample.setStyleSheet(style_param_spin)
        box_gbm_2.addWidget(label)
        box_gbm_2.addWidget(spin_sample)

        label = QLabel('l2_regularization:')
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_regularization = QDoubleSpinBox()
        spin_regularization.setMinimum(0.0)
        spin_regularization.setMaximum(1000.0)
        spin_regularization.setStyleSheet(style_param_spin)
        box_gbm_2.addWidget(label)
        box_gbm_2.addWidget(spin_regularization)

        label = QLabel('max_bins:')
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_max_bins = QSpinBox()
        spin_max_bins.setMinimum(1)
        spin_max_bins.setMaximum(255)
        spin_max_bins.setValue(255)
        spin_max_bins.setStyleSheet(style_param_spin)
        box_gbm_2.addWidget(label)
        box_gbm_2.addWidget(spin_max_bins)

        box_hyper_gbm = QHBoxLayout()
        hyperparameter_gbm = QCheckBox("Optimizar parámetros")
        hyperparameter_gbm.setStyleSheet(
            style
        )

        scoring_combo_gbm = QComboBox()
        scoring_combo_gbm.setStyleSheet(style_combo_box)
        scoring_combo_gbm.addItems(['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        scoring_combo_gbm.setEnabled(False)
        scoring_combo_gbm.setToolTip(text_tool_tip)
        box_hyper_gbm.addWidget(hyperparameter_gbm)
        box_hyper_gbm.addWidget(scoring_combo_gbm)

        hyperparameter_gbm.clicked.connect(lambda: self.self_click_hyper(hyperparameter_gbm, scoring_combo_gbm))

        button_gbm = create_mini_buttons("Aplicar")
        button_gbm.clicked.connect(lambda: self.apply_LightGBM())

        area_gbm = QFrame()
        area_gbm.setStyleSheet(
            "QFrame {border: 2px solid #814D1F;}"
        )

        grid_GBM.addLayout(box_gbm, 0, 0, 1, 4)
        grid_GBM.addLayout(box_gbm_2, 1, 0, 1, 4)
        grid_GBM.addLayout(box_hyper_gbm, 2, 0, 1, 2)
        grid_GBM.addWidget(button_gbm, 2, 3)
        grid_GBM.addWidget(area_gbm, 3, 0, 2, 4)
        widget_gbm.setLayout(grid_GBM)

        # Random Forest
        widget_RF = QWidget()
        widget_RF.setObjectName("RF")

        tabWidget.addTab(widget_RF, "Random Forest")

        grid_RF = QGridLayout()
        box = QHBoxLayout()

        label = QLabel("min_samples_split:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_box_1 = QDoubleSpinBox()
        spin_box_1.setStyleSheet(style_param_spin)
        spin_box_1.setMinimum(0.01)
        spin_box_1.setValue(2)
        box.addWidget(label)
        box.addWidget(spin_box_1)

        label = QLabel("n_estimators:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_box = QSpinBox()
        spin_box.setStyleSheet(style_param_spin)
        spin_box.setMinimum(1)
        spin_box.setMaximum(1000)
        spin_box.setValue(100)
        box.addWidget(label)
        box.addWidget(spin_box)

        box_2 = QHBoxLayout()
        label = QLabel("max_depth:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_box = QSpinBox()
        spin_box.setStyleSheet(style_param_spin)
        spin_box.setGeometry(spin_box_1.geometry())
        spin_box.setMinimum(0)
        spin_box.setValue(0)  # Default is None
        box_2.addWidget(label)
        box_2.addWidget(spin_box)

        list_criterion = QComboBox()
        label = QLabel("criterion:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        list_criterion.addItems(["gini", "entropy", "log_loss"])
        list_criterion.setStyleSheet(style_combo_box)
        box_2.addWidget(label)
        box_2.addWidget(list_criterion)

        box_hyper = QHBoxLayout()
        hyperparameter = QCheckBox("Optimizar parámetros")
        hyperparameter.setStyleSheet(
            style
        )

        scoring_combo = QComboBox()
        scoring_combo.setStyleSheet(style_combo_box)
        scoring_combo.addItems(['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        scoring_combo.setEnabled(False)
        scoring_combo.setToolTip(text_tool_tip)
        box_hyper.addWidget(hyperparameter)
        box_hyper.addWidget(scoring_combo)

        hyperparameter.clicked.connect(lambda: self.self_click_hyper(hyperparameter, scoring_combo))

        button_RF = create_mini_buttons("Aplicar")
        button_RF.clicked.connect(lambda: self.apply_RF())

        grid_RF.addLayout(box, 0, 0, 1, 4)
        grid_RF.addLayout(box_2, 1, 0, 1, 4)
        grid_RF.addLayout(box_hyper, 2, 0, 1, 2)
        grid_RF.addWidget(button_RF, 2, 3)

        area = QFrame()
        area.setStyleSheet(
            "QFrame {border: 2px solid #814D1F;}"
        )
        grid_RF.addWidget(area, 3, 0, 2, 4)

        widget_RF.setLayout(grid_RF)

        # SVM
        widget_SVM = QWidget()
        widget_SVM.setObjectName("SVM")
        tabWidget.addTab(widget_SVM, "SVM")

        grid_svm = QGridLayout()
        box_svm = QHBoxLayout()

        label = QLabel("C:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_svm = QDoubleSpinBox()
        spin_svm.setMinimum(1)
        spin_svm.setStyleSheet(style_param_spin)
        box_svm.addWidget(label)
        box_svm.addWidget(spin_svm)

        label = QLabel("kernel:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        combo_kernel = QComboBox()
        combo_kernel.addItems(['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'])
        combo_kernel.setStyleSheet(style_combo_box)
        box_svm.addWidget(label)
        box_svm.addWidget(combo_kernel)

        label = QLabel('gamma:')
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        combo_gamma = QComboBox()
        combo_gamma.addItems(['scale', 'auto', '1', '0.1', '0.01', '0.001', '0.0001'])
        combo_gamma.setStyleSheet(style_combo_box)
        box_svm.addWidget(label)
        box_svm.addWidget(combo_gamma)

        box_hyper_svm = QHBoxLayout()
        hyperparameter_svm = QCheckBox("Optimizar parámetros")
        hyperparameter_svm.setStyleSheet(
            style
        )

        scoring_combo_svm = QComboBox()
        scoring_combo_svm.setStyleSheet(style_combo_box)
        scoring_combo_svm.addItems(['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        scoring_combo_svm.setEnabled(False)
        scoring_combo_svm.setToolTip(text_tool_tip)
        box_hyper_svm.addWidget(hyperparameter_svm)
        box_hyper_svm.addWidget(scoring_combo_svm)

        hyperparameter_svm.clicked.connect(lambda: self.self_click_hyper(hyperparameter_svm, scoring_combo_svm))

        button_svm = create_mini_buttons("Aplicar")
        button_svm.clicked.connect(lambda: self.apply_SVM())

        area_svm = QFrame()
        area_svm.setStyleSheet(
            "QFrame {border: 2px solid #814D1F;}"
        )

        grid_svm.addLayout(box_svm, 0, 0, 1, 4)
        grid_svm.addLayout(box_hyper_svm, 1, 0, 1, 2)
        grid_svm.addWidget(button_svm, 1, 3)
        grid_svm.addWidget(area_svm, 2, 0, 2, 4)
        widget_SVM.setLayout(grid_svm)

        # XGBoost
        widget_XGBoost = QWidget()
        widget_XGBoost.setObjectName("XGBoost")
        tabWidget.addTab(widget_XGBoost, "XGBoost")

        grid_XGBoost = QGridLayout()
        box_XGBoost = QHBoxLayout()
        box_XGBoost_3 = QHBoxLayout()

        label = QLabel("max_depth:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_XGBoost_1 = QSpinBox()
        spin_XGBoost_1.setStyleSheet(style_param_spin)
        spin_XGBoost_1.setMinimum(1)
        spin_XGBoost_1.setValue(6)
        spin_XGBoost_1.setMaximum(1000)
        box_XGBoost.addWidget(label)
        box_XGBoost.addWidget(spin_XGBoost_1)

        label = QLabel("n_estimators:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_XGBoost = QSpinBox()
        spin_XGBoost.setStyleSheet(style_param_spin)
        spin_XGBoost.setMinimum(1)
        spin_XGBoost.setMaximum(1000)
        spin_XGBoost.setValue(100)
        box_XGBoost.addWidget(label)
        box_XGBoost.addWidget(spin_XGBoost)

        box_XGBoost_2 = QHBoxLayout()
        label = QLabel("subsample:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        spin_XGBoost_2 = QDoubleSpinBox()
        spin_XGBoost_2.setStyleSheet(style_param_spin)
        spin_XGBoost_2.setMinimum(0.0001)
        spin_XGBoost_2.setMaximum(1)
        spin_XGBoost_2.setValue(1)
        box_XGBoost_2.addWidget(label)
        box_XGBoost_2.addWidget(spin_XGBoost_2)

        colsample = QDoubleSpinBox()
        label = QLabel("colsample_bytree:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        colsample.setStyleSheet(style_param_spin)
        colsample.setMinimum(0.0001)
        colsample.setMaximum(1)
        colsample.setValue(1)
        box_XGBoost_2.addWidget(label)
        box_XGBoost_2.addWidget(colsample)

        learning_rate = QDoubleSpinBox()
        label = QLabel("learning_rate:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        learning_rate.setStyleSheet(style_param_spin)
        learning_rate.setMinimum(0.0001)
        learning_rate.setMaximum(1)
        learning_rate.setValue(0.3)
        box_XGBoost_3.addWidget(label)
        box_XGBoost_3.addWidget(learning_rate)

        box_hyper_XGBoost = QHBoxLayout()
        hyperparameter_XGBoost = QCheckBox("Optimizar parámetros")
        hyperparameter_XGBoost.setStyleSheet(
            style
        )

        scoring_combo_XGBoost = QComboBox()
        scoring_combo_XGBoost.setStyleSheet(style_combo_box)
        scoring_combo_XGBoost.addItems(['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        scoring_combo_XGBoost.setEnabled(False)
        scoring_combo_XGBoost.setToolTip(text_tool_tip)
        box_hyper_XGBoost.addWidget(hyperparameter_XGBoost)
        box_hyper_XGBoost.addWidget(scoring_combo_XGBoost)

        hyperparameter_XGBoost.clicked.connect(
            lambda: self.self_click_hyper(hyperparameter_XGBoost, scoring_combo_XGBoost))

        button_XGBoost = create_mini_buttons("Aplicar")
        button_XGBoost.clicked.connect(lambda: self.apply_XGBoost())

        grid_XGBoost.addLayout(box_XGBoost, 0, 0, 1, 4)
        grid_XGBoost.addLayout(box_XGBoost_2, 1, 0, 1, 4)
        grid_XGBoost.addLayout(box_XGBoost_3, 2, 2, 1, 2)
        grid_XGBoost.addLayout(box_hyper_XGBoost, 3, 0, 1, 2)
        grid_XGBoost.addWidget(button_XGBoost, 3, 3)

        area_XGBoost = QFrame()
        area_XGBoost.setStyleSheet(
            "QFrame {border: 2px solid #814D1F;}"
        )
        grid_XGBoost.addWidget(area_XGBoost, 4, 0, 1, 4)

        widget_XGBoost.setLayout(grid_XGBoost)

        widgets['graph_class'].append(tabWidget)

        button_back_f3 = create_buttons("Atrás", 5, 115)
        button_next_f3 = create_buttons("Siguiente", 5, 115)
        button_next_f3.setEnabled(False)
        button_next_f3.setToolTip("Para continuar debe ejecutar algún clasificador")

        button_back_f3.clicked.connect(lambda: self.frame5(ruta, name, name_widget))
        button_next_f3.clicked.connect(lambda: self.frame7(ruta, name, name_widget))

        widgets['button_back_f3'].append(button_back_f3)
        widgets['button_next_f3'].append(button_next_f3)

        self.add_widgets_f6()

    def apply_attr_sel(self):
        self.attr_selection = AttrSelection()
        self.attr_selection.show()

    def apply_split_train_test(self):
        global sensitive_attr, X_train, X_test, y_train, y_test, A_train, A_test

        sensitive_attr = widgets['list_widget'][-1].findChildren(QComboBox)[0].currentText()

        test_size = int(widgets['list_widget'][-1].findChildren(QSpinBox)[0].value()) / 100

        attributes = widgets['imputation_attr'][-1].findChildren(QCheckBox)
        list_drop = [attr.text() for attr in attributes if attr.isChecked()]
        list_drop.append(sensitive_attr)

        drop_data = new_dataset.drop(list_drop, axis=1)
        # Encoder numerical attributes
        drop_data = self.codification_2(drop_data)
        X, y = drop_data.iloc[:, :len(drop_data.columns) - 1], drop_data.iloc[:, -1]

        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, new_dataset[sensitive_attr],
                                                                             test_size=test_size,
                                                                             random_state=42, stratify=y)

        widgets['button4'][-1].setEnabled(True)
        widgets['button_next_f3'][-1].setEnabled(True)

    def show_attributes_split(self, attr):
        self.clear_area(3)
        area = widgets['imputation_attr'][-1]

        font_size = '12px' if name_dataset == 'students' else '14px'

        title_layout = QLabel("Seleccione" + "\n" + "los atributos" + "\n" + "que NO" + "\n" + "desee incluir")
        title_layout.setAlignment(QtCore.Qt.AlignCenter)
        title_layout.setStyleSheet(
            "border: 1px solid #C9C9C9;"
            "font-size: " + font_size + ";"
        )
        title_layout.setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))

        # List of attributes
        check_box = QHBoxLayout()
        check_box_v = QVBoxLayout()

        check_box.addWidget(title_layout)

        index = 0
        for column in list(new_dataset.columns):
            if column != attr and column != class_attribute:
                check_box_1 = QCheckBox(column)
                check_box_1.setStyleSheet("font-size: " + font_size + ";")
                if index >= len(new_dataset.columns) / 4:
                    check_box_v.addWidget(check_box_1)
                    check_box.addLayout(check_box_v)
                    check_box_v = QVBoxLayout()
                    index = 0
                else:
                    check_box_v.addWidget(check_box_1)
                    index += 1
        else:
            check_box.addLayout(check_box_v)

        area.setLayout(check_box)

    def update_area(self, combo):
        self.show_attributes_split(combo.currentText())

    def add_widgets_f5(self):
        self.grid.addWidget(widgets['title'][-1], 0, 0, 1, 4)
        self.grid.addWidget(widgets['list_widget'][-1], 1, 0, 1, 4)
        self.grid.addWidget(widgets['button5'][-1], 2, 3, 1, 2)
        self.grid.addWidget(widgets['imputation_attr'][-1], 3, 0, 2, 4)
        self.grid.addWidget(widgets['button4'][-1], 5, 2, 1, 2)
        self.grid.addWidget(widgets['button_back_f3'][-1], 6, 0, 1, 2)
        self.grid.addWidget(widgets['button_next_f3'][-1], 6, 2, 1, 2)

    def frame5(self, ruta, name, name_widget):
        clear_widgets()
        self.setFixedWidth(950)
        self.setFixedHeight(700)

        # Title
        graphs_label = QLabel("Separación del dataset en entrenamiento y prueba")
        graphs_label.setAlignment(QtCore.Qt.AlignCenter)
        graphs_label.setWordWrap(True)
        graphs_label.setStyleSheet(
            style_title
        )
        widgets['title'].append(graphs_label)

        # Parameters
        frame = QFrame()
        box_h = QHBoxLayout()
        style = "font-size: 16px;"

        label = QLabel("Atributo sensible:")
        label.setStyleSheet(style)
        label.setAlignment(QtCore.Qt.AlignCenter)
        box_h.addWidget(label)

        list_widget = QComboBox()
        list_widget.addItems(list(new_dataset.columns))
        list_widget.setStyleSheet(style_combo_box)
        list_widget.activated.connect(lambda: self.update_area(list_widget))
        box_h.addWidget(list_widget)

        l1 = QLabel("test_size (%): ")
        l1.setStyleSheet(style)
        l1.setAlignment(QtCore.Qt.AlignCenter)
        box_h.addWidget(l1)
        attr_sensible = QSpinBox()
        attr_sensible.setValue(25)
        attr_sensible.setMinimum(1)
        attr_sensible.setMaximum(100)
        attr_sensible.setStyleSheet(style_param_spin)
        box_h.addWidget(attr_sensible)

        frame.setLayout(box_h)
        widgets['list_widget'].append(frame)

        # Button Apply
        button_apply = create_mini_buttons("Aplicar")
        widgets['button5'].append(button_apply)
        button_apply.clicked.connect(lambda: self.apply_split_train_test())

        # Manual delete attributes
        area = QFrame()
        area.setStyleSheet(
            "QFrame {border: 2px solid #814D1F;}"
        )
        widgets['imputation_attr'].append(area)
        self.show_attributes_split(new_dataset.columns[0])

        # Buttons next and back
        r_margin = 45
        l_margin = 25
        button_back_f3 = create_buttons("Atrás", l_margin, r_margin)
        button_attr_selection = create_buttons("Selección de Atributos", l_margin, r_margin)
        button_next_f3 = create_buttons("Siguiente", l_margin, r_margin)
        button_attr_selection.setEnabled(False)
        button_next_f3.setEnabled(False)
        button_next_f3.setToolTip("Para activar la opción de continuar debe separar el dataset en entrenamiento "
                                  "y prueba (botón Aplicar)")

        button_back_f3.clicked.connect(lambda: self.frame4(ruta, name, name_widget))
        button_attr_selection.clicked.connect(lambda: self.apply_attr_sel())
        button_next_f3.clicked.connect(lambda: self.frame6(ruta, name, name_widget))

        widgets['button_back_f3'].append(button_back_f3)
        widgets['button4'].append(button_attr_selection)
        widgets['button_next_f3'].append(button_next_f3)

        self.add_widgets_f5()

    def call_graph(self, param, dataset_n, check_box):
        for wid in range(check_box.count()):
            if type(check_box.itemAt(wid).widget()) != QLabel:
                for i in range(check_box.itemAt(wid).count()):
                    item = check_box.itemAt(wid).itemAt(i)
                    if type(item.widget()) == QCheckBox and item.widget().isChecked():
                        dataset_n = dataset_n.drop(item.widget().text(), axis=1)

        self.window_other = AnotherWindow(param, dataset_n)
        self.window_other.show()

    def correlation_graph(self):
        self.clean_area_graph()
        area = widgets['graph_class'][-1]
        title_layout = QLabel("CORRELOGRAMA" + '\n' + "SELECCIONE SI DESEA NO" + '\n' + "GRAFICAR ALGÚN ATRIBUTO")
        title_layout.setStyleSheet(
            "border: 1px solid #C9C9C9;"
            "font-size: 12px;"
        )
        title_layout.setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))

        button_apply = create_mini_buttons("Graficar")

        # List of attributes
        check_box = QHBoxLayout()
        check_box_v = QVBoxLayout()

        if name_dataset == 'students':
            check_box_v.addWidget(title_layout)
        else:
            check_box.addWidget(title_layout)

        index = 1
        len_column = int(len(dataset.columns) / 4)
        for column in list(new_dataset.columns):
            check_box_1 = QCheckBox(column)
            if index >= len_column:
                check_box_v.addWidget(check_box_1)
                check_box.addLayout(check_box_v)
                check_box_v = QVBoxLayout()
                index = 0
            else:
                check_box_v.addWidget(check_box_1)
                index += 1
        else:
            check_box_v.addWidget(button_apply)
            check_box.addLayout(check_box_v)

        area.setLayout(check_box)

        # Diagram
        corr_dataset = new_dataset[:]
        if name_dataset == 'students':
            corr_dataset['Target'] = corr_dataset.apply(lambda row: 1 if (row['Target'] == 'Graduate') else 0, axis=1)
            corr_dataset['Target'].astype(int)
        elif name_dataset == 'law':
            corr_dataset['race'] = corr_dataset.apply(lambda row: 1 if (row['race'] == 'White') else 0, axis=1)
            corr_dataset['race'].astype(int)
        else:
            corr_dataset = self.codification(corr_dataset)[:]

        button_apply.clicked.connect(lambda: self.call_graph('correlograma', corr_dataset, check_box))

    def clean_area_graph(self):
        if len(widgets['graph_class']) != 0:
            widgets['graph_class'][-1].hide()
            for i in range(0, len(widgets['graph_class'])):
                widgets['graph_class'].pop()
            area = QFrame()
            area.setStyleSheet(
                "QFrame {border: 2px solid #814D1F;}"
            )
            widgets['graph_class'].append(area)
            self.grid.addWidget(widgets['graph_class'][-1], 4, 0, 2, 2)

    def call_graph_bar(self, check_box):
        data = pd.DataFrame()
        text = ''
        for wid in range(1, check_box.count()):
            for i in range(check_box.itemAt(wid).count()):
                item = check_box.itemAt(wid).itemAt(i)
                if type(item.widget()) == QRadioButton and item.widget().isChecked():
                    data[item.widget().text()] = new_dataset[item.widget().text()]
                    text = item.widget().text()

        self.window_other = AnotherWindow('bar', data[text], text)
        self.window_other.show()

    def bar_graph(self):
        self.clean_area_graph()
        area = widgets['graph_class'][-1]

        title_layout = QLabel("GRÁFICO DE BARRAS" + '\n' + "SELECCIONE EL ATRIBUTO" + '\n' + "QUE DESEA GRAFICAR")
        title_layout.setStyleSheet(
            "border: 1px solid #C9C9C9;"
            "font-size: 12px;"
        )
        title_layout.setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))

        button_apply = create_mini_buttons("Graficar")

        # List of attributes
        check_box = QHBoxLayout()
        check_box_v = QVBoxLayout()

        check_box.addWidget(title_layout)

        index = 0
        flag = True
        for column, value in result_columns.items():
            if value == 'Categórico' or value == 'Binario':
                check_box_1 = QRadioButton(column)
                if flag:
                    check_box_1.setChecked(True)
                    flag = False
                check_box_1.setStyleSheet("font-size: 12px;")
                if index >= 3:
                    check_box_v.addWidget(check_box_1)
                    check_box.addLayout(check_box_v)
                    check_box_v = QVBoxLayout()
                    index = 0
                else:
                    check_box_v.addWidget(check_box_1)
                    index += 1
        else:
            check_box_v.addWidget(button_apply)
            check_box.addLayout(check_box_v)

        area.setLayout(check_box)
        button_apply.clicked.connect(lambda: self.call_graph_bar(check_box))

    def call_box_plot(self, check_box):
        data = pd.DataFrame()
        list_columns = [];
        for wid in range(check_box.count()):
            if type(check_box.itemAt(wid).widget()) != QLabel:
                for i in range(check_box.itemAt(wid).count()):
                    item = check_box.itemAt(wid).itemAt(i)
                    if type(item.widget()) == QCheckBox:
                        list_columns.append(item.widget().text())
                        if item.widget().isChecked() and item.widget().text() == 'Todos':
                            data = new_dataset[list_columns[:-1]]
                        elif item.widget().isChecked():
                            data[item.widget().text()] = new_dataset[item.widget().text()]

        self.window_other = AnotherWindow('box', data)
        self.window_other.show()

    def box_plot(self):
        self.clean_area_graph()

        area = widgets['graph_class'][-1]

        title_layout = QLabel("BOX PLOT" + '\n' + "SELECCIONE ATRIBUTO(S)" + '\n' + "PARA GRAFICAR")
        title_layout.setStyleSheet(
            "border: 1px solid #C9C9C9;"
            "font-size: 12px;"
        )
        title_layout.setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))

        button_apply = create_mini_buttons("Graficar")

        # List of attributes
        check_box = QHBoxLayout()
        check_box_v = QVBoxLayout()

        if name_dataset == 'students':
            check_box_v.addWidget(title_layout)
        else:
            check_box.addWidget(title_layout)

        index = 1
        flag = True
        len_column = int(len(new_dataset.columns) / 4)
        for column, value in result_columns.items():
            if value == "Numérico":
                check_box_1 = QCheckBox(column)
                if flag:
                    check_box_1.setChecked(True)
                    flag = False
                if index >= len_column:
                    check_box_v.addWidget(check_box_1)
                    check_box.addLayout(check_box_v)
                    check_box_v = QVBoxLayout()
                    index = 0
                else:
                    check_box_v.addWidget(check_box_1)
                    index += 1
        else:
            check_box_1 = QCheckBox('Todos')
            check_box_v.addWidget(check_box_1)
            check_box_v.addWidget(button_apply)
            check_box.addLayout(check_box_v)

        area.setLayout(check_box)
        button_apply.clicked.connect(lambda: self.call_box_plot(check_box))

    def call_violin_plot(self, box):

        attrib_violin = {'x': box.itemAt(0).itemAt(1).widget().currentText(),
                         'y': box.itemAt(1).itemAt(1).widget().currentText(),
                         'hue': box.itemAt(2).itemAt(1).widget().currentText()}

        self.window_other = AnotherWindow('violin', attrib_violin)
        self.window_other.show()

    def violin_plot(self):
        self.clean_area_graph()

        area = widgets['graph_class'][-1]

        button_apply = create_mini_buttons("Graficar")

        check_box = QHBoxLayout()

        # Attribute value X
        check_box_v = QVBoxLayout()
        title_layout = QLabel("SELECCIONE EL ATRIBUTO DEL EJE X" + '\n')
        title_layout.setStyleSheet(
            "border: 1px solid #C9C9C9;"
            "font-size: 12px;"
        )
        title_layout.setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))
        list_attrib = QComboBox()
        list_attrib.addItems(list(new_dataset.columns))
        list_attrib.setStyleSheet(style_combo_box)
        check_box_v.addWidget(title_layout)
        check_box_v.addWidget(list_attrib)
        check_box_v.addStretch(1)

        check_box.addLayout(check_box_v)

        # Attribute value y
        check_box_v = QVBoxLayout()
        title_layout = QLabel("SELECCIONE UN ATRIBUTO" + '\n' + "PARA EL EJE Y (OPCIONAL)")
        title_layout.setStyleSheet(
            "border: 1px solid #C9C9C9;"
            "font-size: 12px;"
        )
        title_layout.setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))
        list_attrib = QComboBox()
        columns = list(new_dataset.columns)
        columns.insert(0, 'None')
        list_attrib.addItems(columns)
        list_attrib.setStyleSheet(style_combo_box)
        check_box_v.addWidget(title_layout)
        check_box_v.addWidget(list_attrib)
        check_box_v.addStretch(1)

        check_box.addLayout(check_box_v)

        # Attribute value hue
        check_box_v = QVBoxLayout()
        title_layout = QLabel("SELECCIONE UN ATRIBUTO PARA" + '\n' + "EL PARÁMETRO HUE (OPCIONAL)")
        title_layout.setStyleSheet(
            "border: 1px solid #C9C9C9;"
            "font-size: 12px;"
        )
        title_layout.setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))
        list_attrib = QComboBox()
        columns = list(new_dataset.columns)
        columns.insert(0, 'None')
        list_attrib.addItems(columns)
        list_attrib.setStyleSheet(style_combo_box)
        check_box_v.addWidget(title_layout)
        check_box_v.addWidget(list_attrib)
        check_box_v.addStretch(1)
        check_box_v.addWidget(button_apply)

        check_box.addLayout(check_box_v)

        title_layout = QLabel("VIOLIN PLOT")
        title_layout.setStyleSheet(
            "border: 1px solid #C9C9C9;"
            "font-size: 12px;"
        )
        title_layout.setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))
        box_vertical = QVBoxLayout()
        box_vertical.addWidget(title_layout)
        box_vertical.addLayout(check_box)

        area.setLayout(box_vertical)

        button_apply.clicked.connect(lambda: self.call_violin_plot(check_box))

    def get_check_box(self, check_box):
        data = []
        for wid in range(check_box.count()):
            for i in range(check_box.itemAt(wid).count()):
                item = check_box.itemAt(wid).itemAt(i)
                if type(item.widget()) == QCheckBox and item.widget().isChecked() and item.widget().text() == 'Todos':
                    data = dataset_bar_plot.columns
                elif type(item.widget()) == QCheckBox and item.widget().isChecked():
                    data.append(item.widget().text())
        return data

    def call_cross_tab(self, check_box):
        data_x = self.get_check_box(check_box.itemAt(0))
        data_columns = self.get_check_box(check_box.itemAt(1))
        data = [data_x, data_columns]

        if len(data_x) != 0 and len(data_columns) != 0:
            self.window_other = AnotherWindow('cross', data)
            self.window_other.show()

    def cross_tab_plot(self):
        self.clean_area_graph()

        area = widgets['graph_class'][-1]

        # Label X Attributes
        title_layout = QLabel("CROSS TABULATION" + '\n' + "SELECCIONE EL(LOS) ATRIBUTO(S) " + '\n' + "DEL EJE X")
        title_layout.setStyleSheet(
            "border: 1px solid #C9C9C9;"
            "font-size: 12px;"
        )
        title_layout.setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))

        # Label Columns Attributes
        title_layout_2 = QLabel("SELECCIONE EL(LOS) ATRIBUTO(S)" + '\n' + "DE LAS COLUMNAS")
        title_layout_2.setStyleSheet(
            "border: 1px solid #C9C9C9;"
            "font-size: 12px;"
        )
        title_layout_2.setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))

        button_apply = create_mini_buttons("Graficar")

        # List of attributes
        general_check_box = QHBoxLayout()

        # X Attributes
        check_box = QHBoxLayout()
        check_box_v = QVBoxLayout()
        check_box_v.addWidget(title_layout)

        # Columns Attributes
        check_box_column = QHBoxLayout()
        check_box_v_column = QVBoxLayout()
        check_box_v_column.addWidget(title_layout_2)

        global dataset_bar_plot
        dataset_bar_plot = pd.DataFrame()
        index = 0
        flag = True
        for column, value in result_columns.items():
            if value == 'Categórico' or value == 'Binario':
                dataset_bar_plot[column] = new_dataset[column]
                check_box_1 = QCheckBox(column)  # X Attributes
                check_box_2 = QCheckBox(column)
                if flag:
                    check_box_1.setChecked(True)
                    check_box_2.setChecked(True)
                    flag = False
                if index >= 6:
                    check_box_v.addWidget(check_box_1)
                    check_box.addLayout(check_box_v)
                    check_box_v = QVBoxLayout()

                    check_box_v_column.addWidget(check_box_2)
                    check_box_column.addLayout(check_box_v_column)
                    check_box_v_column = QVBoxLayout()

                    index = 0
                else:
                    check_box_v.addWidget(check_box_1)
                    check_box_v_column.addWidget(check_box_2)
                    index += 1
                if column == new_dataset.columns[-1]:
                    check_box_1 = QCheckBox('Todos')
                    check_box_v.addWidget(check_box_1)
                    check_box.addLayout(check_box_v)

                    check_box_2 = QCheckBox('Todos')
                    check_box_v_column.addWidget(check_box_2)
                    check_box_v_column.addWidget(button_apply)
                    check_box_column.addLayout(check_box_v_column)

        general_check_box.addLayout(check_box)
        general_check_box.addLayout(check_box_column)

        if name_dataset == 'car':
            dataset_bar_plot['OUTCOME'] = dataset_bar_plot.apply(
                lambda row: 'NO_CLAIM' if (row['OUTCOME'] == 0.0) else 'CLAIM', axis=1)

        area.setLayout(general_check_box)
        button_apply.clicked.connect(lambda: self.call_cross_tab(general_check_box))

    def codification_2(self, data):
        le = LabelEncoder()

        new_data = data[:]

        for category in new_data.columns:
            new_data[category] = le.fit_transform(new_data[category])

        return new_data

    def codification(self, data):
        le = LabelEncoder()
        categorical_data = []
        numerical_data = []
        new_data = data[:]
        for col in data.columns:
            if data[col].dtype == "O" or (data[col] < 0).values.any():
                categorical_data.append(col)
            else:
                numerical_data.append(col)

        for category in categorical_data:
            new_data[category] = le.fit_transform(new_data[category])

        return new_data

    def add_widgets_f4(self):
        self.grid.addWidget(widgets['graphs_label'][-1], 0, 0, 1, 2)
        self.grid.addWidget(widgets["button_corr"][-1], 1, 0)
        self.grid.addWidget(widgets["button_violin"][-1], 1, 1)
        self.grid.addWidget(widgets["button_bar"][-1], 2, 0)
        self.grid.addWidget(widgets["button_cross"][-1], 2, 1)
        self.grid.addWidget(widgets["button_point"][-1], 3, 0, 1, 2)
        self.grid.addWidget(widgets['graph_class'][-1], 4, 0, 2, 2)
        self.grid.addWidget(widgets['button_back_f3'][-1], 6, 0)
        self.grid.addWidget(widgets['button_next_f3'][-1], 6, 1)

    def frame4(self, ruta, name, name_widget):
        clear_widgets()
        self.setFixedWidth(950)
        self.setFixedHeight(720)
        # Title
        graphs_label = QLabel("Gráficos")
        graphs_label.setAlignment(QtCore.Qt.AlignCenter)
        graphs_label.setWordWrap(True)
        graphs_label.setStyleSheet(style_title)
        widgets["graphs_label"].append(graphs_label)

        # Graph Buttons
        r_margin = 50
        l_margin = 25

        button_corr = create_buttons("Correlograma", l_margin, r_margin)
        button_violin = create_buttons("Violín", l_margin, r_margin)
        button_bar = create_buttons("Gráfico de Barras", l_margin, r_margin)  # Si el len(index) es menor que 5
        button_cross = create_buttons("Tabulación Cruzada", l_margin, r_margin)

        button_point = create_buttons("Box Plot", l_margin, r_margin)

        widgets["button_corr"].append(button_corr)
        widgets["button_violin"].append(button_violin)
        widgets["button_bar"].append(button_bar)
        widgets["button_cross"].append(button_cross)
        widgets["button_point"].append(button_point)

        area = QFrame()
        area.setStyleSheet(
            "QFrame {border: 2px solid #814D1F;}"
        )

        widgets['graph_class'].append(area)

        button_corr.clicked.connect(lambda: self.correlation_graph())
        button_bar.clicked.connect(lambda: self.bar_graph())
        button_point.clicked.connect(lambda: self.box_plot())
        button_cross.clicked.connect(lambda: self.cross_tab_plot())
        button_violin.clicked.connect(lambda: self.violin_plot())

        # Buttons next and back
        button_back_f3 = create_buttons("Atrás", l_margin, r_margin)
        button_next_f3 = create_buttons("Siguiente", l_margin, r_margin)

        button_back_f3.clicked.connect(lambda: self.frame3(ruta, name, name_widget, False))
        button_next_f3.clicked.connect(lambda: self.frame5(ruta, name, name_widget))

        widgets['button_back_f3'].append(button_back_f3)
        widgets['button_next_f3'].append(button_next_f3)

        self.add_widgets_f4()

    def missing_values(self, check_dataset):
        columns = check_dataset.isnull().sum()
        list_columns = {}
        c_missing_values = []
        for col in columns.index:
            list_columns[col] = columns[col]
            if columns[col] != 0:
                c_missing_values.append(col)

        return list_columns, c_missing_values

    def create_table_missing(self, check_dataset):
        missing, c_missing_values = self.missing_values(check_dataset)
        columns = QTableWidget()
        columns.setColumnCount(2)
        columns.setHorizontalHeaderLabels(['Atributo', '# Valores \n Perdidos'])

        columns.resizeColumnToContents(0)
        columns.resizeColumnToContents(1)
        columns.setStyleSheet(
            "font-size: 14px;"
            "color: 'black';"
            "border-style: outset;"
            "border: 2px solid #814D1F;"
        )
        stylesheet = "::section{Background-color: #814D1F;color: white;}"
        columns.horizontalHeader().setStyleSheet(
            stylesheet
        )
        columns.verticalHeader().setFixedWidth(35)
        columns.verticalHeader().setStyleSheet(stylesheet)
        index = 0
        for key, value in missing.items():
            item1 = QTableWidgetItem(key)
            item2 = QTableWidgetItem(str(value))
            item2.setTextAlignment(QtCore.Qt.AlignCenter)
            columns.insertRow(index)
            columns.setItem(index, 0, item1)
            columns.setItem(index, 1, item2)
            if name_dataset == 'students':
                columns.resizeColumnToContents(0)
                columns.resizeColumnToContents(1)
            else:
                columns.setColumnWidth(0, 195)
                columns.setColumnWidth(1, 170)
            index += 1

        widgets["missing"].append(columns)
        return c_missing_values

    def disable_options(self, cs, k_neighbors, weights, cs1=None):
        radio_button = cs.sender()
        if cs1 is not None:
            cs1.setChecked(False)
        if radio_button.isChecked():
            k_neighbors.setEnabled(False)
            weights.setEnabled(False)

    def enable_options(self, cs, k_neighbors, weights, cs1=None):
        radio_button = cs.sender()
        if cs1 is not None:
            cs1.setChecked(False)
        if radio_button.isChecked():
            k_neighbors.setEnabled(True)
            weights.setEnabled(True)

    def imputation_knn(self, attr):
        # Get number of neighbors
        neighbors = 1
        for wid in range(widgets['k_neighbors'][-1].count()):
            if widgets['k_neighbors'][-1].itemAt(wid).widget().objectName() == 'n_neighbors':
                neighbors = widgets['k_neighbors'][-1].itemAt(wid).widget().value()
                break
        # Get weights
        weights = ''
        for wid in range(widgets['weights'][-1].count()):
            if widgets['weights'][-1].itemAt(wid).widget().objectName() == 'weights':
                weights = widgets['weights'][-1].itemAt(wid).widget().currentText()
                break

        # Building the model
        impute = KNNImputer(n_neighbors=neighbors, weights=weights)
        # Fit the model
        impute.fit(new_dataset[[attr]])
        new_dataset[attr] = impute.transform(new_dataset[[attr]]).ravel()

    def reset_missing_values(self):
        # Delete old table
        widgets['missing'][-1].hide()
        for i in range(0, len(widgets['missing'])):
            widgets['missing'].pop()

        # Create new table
        c_missing_values = self.create_table_missing(new_dataset)
        self.grid.addWidget(widgets["missing"][-1], 1, 0, 5, 2)
        return c_missing_values

    def apply_imputation(self):
        for w in range(widgets['cs1'][-1].count()):
            if widgets['cs1'][-1].itemAt(w).widget().isChecked():
                radio_button = widgets['cs1'][-1].itemAt(w).widget().objectName()
                attr = widgets['list_widget'][-1].currentText()
                widgets['list_widget'][-1].removeItem(widgets['list_widget'][-1].currentIndex())
                if radio_button == 'cs1':
                    drop_na_rows(attr)
                elif radio_button == 'cs2':
                    global new_dataset
                    new_dataset = new_dataset.drop([attr], axis=1)
                else:
                    self.imputation_knn(attr)
                break

        c_missing_values = self.reset_missing_values()
        if len(c_missing_values) == 0:
            widgets['button_next_f3'][-1].setEnabled(True)

    def rever(self):
        global dataset, new_dataset
        new_dataset = dataset[:]
        c_missing_values = self.reset_missing_values()
        widgets['list_widget'][-1].clear()
        widgets['list_widget'][-1].addItems(c_missing_values)
        widgets['button_next_f3'][-1].setEnabled(False)

    def update_result_columns(self):
        table = widgets['columns'][-1]
        for key, value in enumerate(result_columns.keys()):
            result_columns[value] = table.cellWidget(key, 1).currentText()

    def add_widgets_f3(self):
        self.grid.addWidget(widgets["imputation_label"][-1], 0, 0, 1, 5)
        self.grid.addWidget(widgets["missing"][-1], 1, 0, 5, 2)
        self.grid.addWidget(widgets["imputation_attr"][-1], 1, 2)
        self.grid.addWidget(widgets["list_widget"][-1], 1, 3)
        self.grid.addLayout(widgets["cs1"][-1], 2, 2, 2, 3)
        self.grid.addLayout(widgets["k_neighbors"][-1], 4, 2)
        self.grid.addLayout(widgets["weights"][-1], 4, 3)
        self.grid.addWidget(widgets["button_apply"][-1], 5, 2)
        self.grid.addWidget(widgets["button_rever"][-1], 5, 3)
        self.grid.addWidget(widgets["button_back_f3"][-1], 6, 0, 1, 2)
        self.grid.addWidget(widgets["button_next_f3"][-1], 6, 2, 1, 2)

    def frame3(self, ruta, name, name_widget, bool_update_result):
        if bool_update_result:
            self.update_result_columns()

        clear_widgets()
        self.setFixedWidth(900)
        self.setFixedHeight(600)

        # Title
        imputation_label = QLabel("Tratamiento de Valores Perdidos")
        imputation_label.setAlignment(QtCore.Qt.AlignCenter)
        imputation_label.setWordWrap(True)
        imputation_label.setStyleSheet(style_title)
        widgets["imputation_label"].append(imputation_label)

        # Missing values
        c_missing_values = self.create_table_missing(dataset)

        # Missing values imputation
        imputation_attr = QLabel("Seleccione el atributo")
        imputation_attr.setAlignment(QtCore.Qt.AlignCenter)
        imputation_attr.setWordWrap(True)
        imputation_attr.setStyleSheet(
            "font-size: 18px;"
            "font: 'Arial';"
        )
        widgets["imputation_attr"].append(imputation_attr)

        list_widget = QComboBox()
        list_widget.addItems(c_missing_values)
        list_widget.setStyleSheet(style_combo_box)
        widgets["list_widget"].append(list_widget)

        # Radio Buttons Missing values
        vbox = QVBoxLayout()

        cs1 = QRadioButton("Eliminar filas con nulos")
        cs2 = QRadioButton("Eliminar el atributo")
        cs3 = QRadioButton("Imputación con K-NN")

        stylesheet = "font-size: 18px; color: 'black'; padding: 15px;"
        cs1.setStyleSheet(stylesheet)
        cs2.setStyleSheet(stylesheet)
        cs3.setStyleSheet(stylesheet)

        cs1.setObjectName("cs1")
        cs2.setObjectName("cs2")
        cs3.setObjectName("cs3")

        vbox.addWidget(cs1)
        vbox.addWidget(cs2)
        vbox.addWidget(cs3)
        widgets["cs1"].append(vbox)

        # Parameters knn

        k_neighbors = QVBoxLayout()
        l1 = QLabel("n_neighbors:")
        l1.setAlignment(QtCore.Qt.AlignCenter)
        l1.setStyleSheet(stylesheet)

        k_neighbors_values = QSpinBox()
        k_neighbors_values.setMinimum(1)
        k_neighbors_values.setMaximum(20)
        k_neighbors_values.setStyleSheet(style_param_spin)
        k_neighbors_values.setEnabled(False)
        k_neighbors_values.setObjectName('n_neighbors')

        k_neighbors.addWidget(l1)
        k_neighbors.addWidget(k_neighbors_values)
        widgets['k_neighbors'].append(k_neighbors)

        # Parameters weights
        weights = QVBoxLayout()
        l2 = QLabel("Weight:")
        l2.setAlignment(QtCore.Qt.AlignCenter)
        l2.setStyleSheet(stylesheet)

        weights_values = QComboBox()
        weights_values.addItem('uniform')
        weights_values.addItem('distance')
        weights_values.setStyleSheet(style_combo_box)
        weights_values.setEnabled(False)
        weights_values.setObjectName('weights')

        weights.addWidget(l2)
        weights.addWidget(weights_values)
        widgets['weights'].append(weights)

        # Actions radio button click
        cs1.toggled.connect(lambda: self.disable_options(cs1, k_neighbors_values, weights_values))
        cs2.toggled.connect(lambda: self.disable_options(cs2, k_neighbors_values, weights_values, cs1))
        cs3.toggled.connect(lambda: self.enable_options(cs3, k_neighbors_values, weights_values, cs1))

        # Buttons Apply and Rever
        button_apply = create_mini_buttons("Aplicar", 8, 8)
        button_rever = create_mini_buttons("Revertir", 8, 8)

        button_apply.clicked.connect(lambda: self.apply_imputation())
        button_rever.clicked.connect(self.rever)

        widgets["button_apply"].append(button_apply)
        widgets["button_rever"].append(button_rever)

        # Buttons next and back
        button_back_f3 = create_buttons("Atrás", 25, 55)
        button_next_f3 = create_buttons("Siguiente", 25, 55)

        button_back_f3.clicked.connect(lambda: self.frame2(ruta, name, name_widget))
        button_next_f3.clicked.connect(lambda: self.frame4(ruta, name, name_widget))
        button_next_f3.setEnabled(False)

        if len(c_missing_values) == 0:
            cs1.setEnabled(False)
            cs2.setEnabled(False)
            cs3.setEnabled(False)
            button_apply.setEnabled(False)
            button_rever.setEnabled(False)
            button_next_f3.setEnabled(True)
            # ToolTips
            text = "No existen valores perdidos"
            cs1.setToolTip(text)
            cs2.setToolTip(text)
            cs3.setToolTip(text)
            button_apply.setToolTip(text)
            button_rever.setToolTip(text)
        else:
            cs1.setChecked(True)

        widgets['button_back_f3'].append(button_back_f3)
        widgets['button_next_f3'].append(button_next_f3)

        self.add_widgets_f3()

    def get_columns(self, ruta, name_widget):
        global dataset
        if name_widget == 'students':
            dataset = pd.read_csv(ruta, delimiter=';')
        else:
            dataset = pd.read_csv(ruta)

        # output class
        global class_attribute
        if name_widget == 'students':
            class_attribute = 'Target'
            dataset = dataset[~dataset['Target'].str.contains('Enrolled')]
        elif name_widget == 'law':
            dataset['fam_inc'] = dataset['fam_inc'].astype(int)
            dataset['male'] = dataset['male'].astype(int)
            dataset['fulltime'] = dataset['fulltime'].astype(int)
            dataset['tier'] = dataset['tier'].astype(int)
            dataset['pass_bar'] = dataset['pass_bar'].astype(int)
            dataset = dataset[dataset['zgpa'] > -6]
            class_attribute = 'pass_bar'
        else:
            dataset['VEHICLE_OWNERSHIP'] = dataset['VEHICLE_OWNERSHIP'].astype(int)
            dataset['MARRIED'] = dataset['MARRIED'].astype(int)
            dataset['CHILDREN'] = dataset['CHILDREN'].astype(int)
            dataset['OUTCOME'] = dataset['OUTCOME'].astype(int)
            class_attribute = 'OUTCOME'

        global new_dataset
        new_dataset = dataset[:]

        columns = list(dataset.columns)
        list_columns = {}
        for col in columns:
            list_columns[col] = str(dataset[col].dtype)

        return list_columns

    def distribution_classes(self):
        values = dataset[class_attribute].value_counts()
        index = dataset[class_attribute].value_counts().index

        graph_class = FigureCanvas(Figure(figsize=(5, 3)))
        graph_class.figure.set_facecolor('#C9C9C9')
        ax = graph_class.figure.subplots()
        bar_container = ax.bar(index, values, color=['#814D1F', '#C9C9C9'], hatch=['//'], edgecolor=['#814D1F'])
        ax.set(title='Distribución de las clases', ylim=(0, len(dataset)))
        ax.bar_label(bar_container, fmt='{:,.0f}')
        if name_dataset == 'car':
            ticks = ['NO CLAIM', 'CLAIM']
            index = dataset[class_attribute].value_counts().index
            ax.set_xticks(index, ticks)
        elif name_dataset == 'law':
            ticks = ['PASS BAR', 'NO PASS BAR']
            index = dataset[class_attribute].value_counts().index
            ax.set_xticks(index, ticks)
        else:
            ticks = dataset[class_attribute].unique()
            ax.set_xticks(ticks)
        ax.set_facecolor('#C9C9C9')
        widgets['graph_class'].append(graph_class)

    def add_widgets_f2(self):
        if name_dataset == 'students':
            self.grid.addWidget(widgets["title"][-1], 0, 0, 1, 4)
            self.grid.addWidget(widgets["columns"][-1], 1, 0, 5, 3)
            self.grid.addWidget(widgets["instances"][-1], 1, 3)
            self.grid.addWidget(widgets["graph_class"][-1], 2, 3, 4, 1)
        else:
            self.grid.addWidget(widgets["title"][-1], 0, 0, 1, 4)
            self.grid.addWidget(widgets["columns"][-1], 1, 0, 5, 1)
            self.grid.addWidget(widgets["instances"][-1], 1, 1, 1, 3)
            self.grid.addWidget(widgets["graph_class"][-1], 2, 1, 4, 3)
        self.grid.addLayout(widgets["button4"][-1], 6, 0, 1, 4)

    def frame2(self, ruta, name, name_widget):
        clear_widgets()
        self.setFixedWidth(900)
        self.setFixedHeight(710)

        global name_dataset
        name_dataset = name_widget

        title = QLabel("Dataset " + name)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setWordWrap(True)
        title.setStyleSheet(style_title)
        widgets["title"].append(title)

        global result_columns
        result_columns = self.get_columns(ruta, name_widget)
        columns = QTableWidget()
        columns.setColumnCount(2)
        columns.setHorizontalHeaderLabels(['Atributo', 'Tipo'])

        columns.setStyleSheet(
            "font-size: 14px;"
            "color: 'black';"
            "border-style: outset;"
            "border: 2px solid #814D1F;"
        )
        stylesheet = "::section{Background-color: #814D1F;color: white;}"

        columns.horizontalHeader().setStyleSheet(
            stylesheet
        )

        columns.verticalHeader().setFixedWidth(35)
        columns.verticalHeader().setStyleSheet(stylesheet)

        index = 0
        for key, value in result_columns.items():
            item1 = QTableWidgetItem(key)

            columns.insertRow(index)
            """"Create list of types"""
            list_types = QComboBox()
            list_types.addItems(["Binario", "Categórico", "Numérico"])
            if value == 'float64' or value == 'float86' or value == 'float' or value == 'float32':
                list_types.setCurrentText('Numérico')
            elif len(dataset[key].value_counts()) == 2:
                list_types.setCurrentText('Binario')
            elif 2 < len(dataset[key].value_counts()) < 7:
                list_types.setCurrentText('Categórico')
            else:
                list_types.setCurrentText('Numérico')
            list_types.setStyleSheet(style_combo_box)
            columns.setItem(index, 0, item1)
            columns.setCellWidget(index, 1, list_types)

            if name_dataset == 'law':
                columns.setColumnWidth(0, 195)
            else:
                columns.resizeColumnToContents(0)
            columns.resizeColumnToContents(1)
            index += 1

        widgets["columns"].append(columns)
        columns.setGeometry(QRect(19, 240, 321, 441))

        drop_duplicates()

        instances = QLabel()
        texto = "Número de instancias: " + str(len(dataset))
        instances.setText(texto)
        instances.setAlignment(QtCore.Qt.AlignCenter)
        instances.setWordWrap(True)
        instances.setStyleSheet(
            "font-size: 16px;"
            "color: 'black';"
            "font: 'Arial';"
            "border: 2px solid #814D1F;"
        )
        widgets["instances"].append(instances)

        # Distribution output class
        self.distribution_classes()

        # Botones de avanzar e ir atras
        layout_horizontal = QHBoxLayout()
        button4 = create_buttons("Atrás", 25, 55)
        button5 = create_buttons("Siguiente", 25, 55)
        layout_horizontal.addWidget(button4)
        layout_horizontal.addWidget(button5)

        button4.clicked.connect(lambda: self.show_frame1())
        button5.clicked.connect(lambda: self.frame3(ruta, name, name_widget, True))

        widgets["button4"].append(layout_horizontal)

        self.add_widgets_f2()

    def show_frame1(self):
        clear_widgets()
        self.frame1()

    def frame1(self):
        """
        Seleccionar el dataset a analizar
        """

        # Imagen tomada de https://www.groovypost.com/howto/fix-file-and-folder-thumbnails-on-windows/
        image = QPixmap(resource_path("images/file_explorer.jpg"))
        logo = QLabel()
        logo.setPixmap(image)
        logo.setAlignment(QtCore.Qt.AlignCenter)
        logo.setStyleSheet(
            "margin-bottom: 25px;"
            "padding: 0;"
        )

        self.setFixedWidth(image.width() - 10)
        self.setFixedHeight(700)

        widgets["logo"].append(logo)
        self.grid.addWidget(widgets["logo"][-1], 0, 0)

        question = QLabel("Seleccione el dataset a analizar")
        question.setAlignment(QtCore.Qt.AlignCenter)
        question.setWordWrap(True)
        question.setStyleSheet(
            "font-size: 22px;"
            "font-family: Times New Roman;"
            "color: 'black';"
        )
        widgets["question"].append(question)
        self.grid.addWidget(widgets["question"][-1], 1, 0, 1, 2)

        button1 = create_buttons("Abandono o éxito académico", 15, 15)
        button2 = create_buttons("Reclamo de seguro de coches", 15, 15)
        button3 = create_buttons("Admisión en la escuela de derecho", 15, 15)

        button1.clicked.connect(
            lambda: self.frame2(resource_path("data/students.csv"), "Abandono o éxito académico", "students"))
        button2.clicked.connect(
            lambda: self.frame2(resource_path("data/Car_Insurance_Claim.csv"), "Reclamo de seguro de coches", "car"))
        button3.clicked.connect(
            lambda: self.frame2(resource_path("data/law_school_clean.csv"), "Admisión en la escuela de derecho", "law"))

        widgets["students"].append(button1)
        widgets["car"].append(button2)
        widgets["law"].append(button3)

        self.grid.addWidget(widgets["students"][-1], 2, 0, alignment=QtCore.Qt.AlignCenter)
        self.grid.addWidget(widgets["car"][-1], 3, 0, alignment=QtCore.Qt.AlignCenter)
        self.grid.addWidget(widgets["law"][-1], 4, 0, alignment=QtCore.Qt.AlignCenter)


app = QApplication(sys.argv)
w = MainWindow()
w.setLayout(w.grid)
w.show()
app.exec()
