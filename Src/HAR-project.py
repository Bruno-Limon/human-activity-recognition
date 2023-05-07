# %% [markdown]
# # **<font color="#42f5f5"> 0.0 Basics</font>**

# %% [markdown]
# ## **<font color="#FBBF44">Funzioni</font>**

# %%
# Template per tutti i confusion matrix di classification

yticks = {1,2,3,4,5,6}
xticks = {1,2,3,4,5,6}

def draw_confusion_matrix(cm, model):
    plt.figure(figsize=(12,8))
    sns.heatmap(cm, xticklabels=xticks, yticklabels=yticks, annot=True, fmt="d", center=0, cmap='mako')
    plt.title( str(model)+" Confusion Matrix", fontsize=15)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.savefig('FigXX-'+str(model)+'ConfusionMatrix.png', dpi=600)
    plt.show()

def draw_confusion_matrix_tuned(cm, model):
    plt.figure(figsize=(12,8))
    sns.heatmap(cm, xticklabels=xticks, yticklabels=yticks, annot=True, fmt="d", center=0, cmap='mako')
    plt.title( str(model)+" Confusion Matrix after tuning", fontsize=15)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.savefig('FigXX-'+str(model)+'ConfusionMatrixTuned.png', dpi=600)
    plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">Controllo GPU</font>**

# %%
# import tensorflow as tf
# tf.test.gpu_device_name()

# %%
# !ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
# !pip install gputil
# !pip install psutil
# !pip install humanize
# import psutil
# import humanize
# import os
# import GPUtil as GPU
# GPUs = GPU.getGPUs()
# gpu = GPUs[0]
# def printm():
#  process = psutil.Process(os.getpid())
#  print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
#  print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
# printm()

# %%


# %% [markdown]
# ## **<font color="#FBBF44">1.0 Imports</font>**

# %%
# Basics
import pandas as pd
import numpy as np
import tensorflow as tf
from numpy import quantile, where, random, array
from collections import Counter
from collections import defaultdict
from itertools import cycle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, label_binarize
from sklearn.datasets import make_classification

# Visualization
import matplotlib.pyplot as plt
from matplotlib import pylab
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Outlier Detection
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

# Dimensionality Reduction
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, chi2
from sklearn.decomposition import PCA
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn import random_projection

# Imbalanced Learning
from imblearn import under_sampling, over_sampling
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Learners
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neural_network import MLPClassifier
# EnsembleClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor

# Evaluation
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report, r2_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from scipy import stats

# Hypertuning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, learning_curve, KFold
from sklearn import model_selection

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling1D
from keras.layers import Conv1D, Activation, Conv1D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import LSTM

# TimeSeries Analysis
!pip install tslearn
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.datasets import CachedDatasets

# TS Approximation
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.piecewise import OneD_SymbolicAggregateApproximation
from sklearn.metrics import pairwise_distances
from tslearn.metrics import dtw, dtw_path, cdist_dtw, subsequence_cost_matrix
from tslearn.utils import ts_size

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import cdist

# Shapelet
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict, LearningShapelets
!pip install pyts
from pyts.transformation import ShapeletTransform

# TS Clustering
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans

# warning
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## **<font color="#FBBF44">1.1 Data Loading</font>**

# %%
# Monto il Drive per accedere ai file
# Percorso: '/content/drive/MyDrive/UCI HAR Dataset/...'
from google.colab import drive
drive.mount('/content/drive')

# %%
!mkdir local_data

!cp -r /content/drive/MyDrive/"UCI HAR Dataset"/train /content/local_data
!cp -r /content/drive/MyDrive/"UCI HAR Dataset"/test /content/local_data

data=pd.read_csv('/content/local_data/train/X_train.txt', header=None, delim_whitespace=True)
subject=pd.read_csv('/content/local_data/train/subject_train.txt', header=None)
y_label=pd.read_csv('/content/local_data/train/y_train.txt', header=None)

data_test = pd.read_csv('/content/local_data/test/X_test.txt', header=None, delim_whitespace=True)
subject_test=pd.read_csv('/content/local_data/test/subject_test.txt', header=None)
y_label_test = pd.read_csv('/content/local_data/test/y_test.txt', header=None)

feature=open("/content/drive/MyDrive/UCI HAR Dataset/features.txt","r")
feature_list=list(feature)

data.columns=feature_list
data_test.columns=feature_list
data.head()
print(data.info())

# %% [markdown]
# ## **<font color="#FBBF44">1.3 Data Preparation</font>**

# %%
# Creamo degli array per X e y
X_train = data.values
y_train = np.array(y_label)
y_train = y_train.ravel()

X_test = data_test.values
y_test = np.array(y_label_test)
y_test = y_test.ravel()

y_train_bin = label_binarize(y_train, classes=[1, 2, 3, 4, 5, 6])
y_test_bin = label_binarize(y_test, classes=[1, 2, 3, 4, 5, 6])
n_classes = y_train_bin.shape[1]

print(X_train.shape)
print(y_train.shape)
print(y_train_bin)
print(y_train_bin.shape)

# %% [markdown]
# # **<font color="#42f5f5"> 1.0 DATA UNDERSTANDING</font>**
#
#
#
#
#
#
#

# %% [markdown]
# ## **<font color="#FBBF44">1.2 Data Understanding</font>**

# %%
full_data = pd.concat((data, data_test), axis=0)
full_label = pd.concat((y_label, y_label_test), axis=0)

full_data.info()
full_data.describe().T

# %%
# Controllo se ci sono null
full_data.isnull().values.any()

# %%
!pip install fitter
from fitter import Fitter, get_common_distributions, get_distributions

# %%
# import random
# randomlist = []
# random.seed()

# for i in range(0,5):
#   n = random.randint(0,560)
#   randomlist.append(n)

# print(randomlist)

# for i in randomlist:
#   sns.displot(data=data, x=data.iloc[:,i], kind='hist', bins=200, aspect=1.5)

# %%
X1 = data.iloc[:,2].values
f = Fitter(X1, distributions=get_common_distributions())
f.fit()

f.hist()
f.plot_pdf()
pylab.grid(False)
pylab.title('tBodyAcc-mean()-Z Distribution')
pylab.savefig('FigXX-DataDistribution1.png', dpi=600)

# %%
X1 = data.iloc[:,530].values
f = Fitter(X1, distributions=get_common_distributions())
f.fit()

f.hist()
f.plot_pdf()
pylab.grid(False)
pylab.title('fBodyBodyGyroMag-mad() Distribution')
pylab.savefig('FigXX-DataDistribution2.png', dpi=600)

# %%
X1 = data.iloc[:,53].values
f = Fitter(X1, distributions=get_common_distributions())
f.fit()

f.hist()
f.plot_pdf()
pylab.grid(False)
pylab.title('tGravityAcc-min()-Y Distribution')
pylab.savefig('FigXX-DataDistribution3.png', dpi=600)

# %%
X1 = data.iloc[:,191].values
f = Fitter(X1, distributions=get_common_distributions())
f.fit()

f.hist()
f.plot_pdf()
pylab.grid(False)
pylab.title('tBodyGyroJerk-arCoeff()-Y,4 Distribution')
pylab.savefig('FigXX-DataDistribution4.png', dpi=600)

# %% [markdown]
# # **<font color="#42f5f5">2.0 CLASSIFICATION TASK</font>**

# %% [markdown]
# ### **<font color="#34eb89">Decision Tree</font>**
#

# %%
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %%
clf = DecisionTreeClassifier(max_depth=10, max_leaf_nodes=50, min_samples_leaf=2, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %%
# Inizializziamo il classifier
clf = OneVsRestClassifier(DecisionTreeClassifier(random_state=0))
y_score = clf.fit(X_train, y_train_bin).predict_proba(X_test)

# Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcoliamo microaverage
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Mettiamo tutti i FPR insieme
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plottiamo i ROC curves
plt.figure()

# Macro
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="MacroAvg (AUC:{0:0.4f})".format(roc_auc["macro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

# Curve per ogni classe
colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1,
        label="Class {0} (AUC:{1:0.4f})".format((i+1), roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Decision Tree Classifier AUC-ROC Curve")
plt.legend(loc="lower right")
plt.savefig('FigXX-DTROC.png', dpi=600)
plt.show()

# %%
# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# MicroAvg calcola score di tutte le classi
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

_, ax = plt.subplots(figsize=(6, 4))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="MicroAvg", color="red")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Class {(i+1)}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([-.02, 1.02])
ax.set_ylim([0.0, 1.02])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Decision Tree Classifier Precision-Recall Curve")
plt.savefig('FigXX-DTPR.png', dpi=600)
plt.show()

# %% [markdown]
# ### **<font color="#34eb89">KNN</font>**

# %%
# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')
#     acc = []
#     for i in range(1,20):
#         neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
#         yhat = neigh.predict(X_test)
#         acc.append(metrics.accuracy_score(y_test, yhat))

#     plt.figure(figsize=(10,6))
#     plt.plot(range(1,20),acc,color = 'darkslategray', marker='o',markerfacecolor='paleturquoise', markersize=10)
#     plt.title('Optimal K')
#     plt.xlabel('K')
#     plt.ylabel('Accuracy')
#     plt.savefig('FigXX-OptimalKNN.png', dpi=600)
#     print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

# %%
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits = 4))

# %%
knn = KNeighborsClassifier(algorithm='brute', n_neighbors=7, weights='distance')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits = 4))

# %%
# Inizializziamo il classifier
clf = OneVsRestClassifier(KNeighborsClassifier())
y_score = clf.fit(X_train, y_train_bin).predict_proba(X_test)

# Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcoliamo microaverage
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Mettiamo tutti i FPR insieme
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plottiamo i ROC curves
plt.figure()

# Macro
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="MacroAvg (AUC:{0:0.4f})".format(roc_auc["macro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

# Curve per ogni classe
colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1,
        label="Class {0} (AUC:{1:0.4f})".format((i+1), roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("K-Nearest Neighbors Classifier AUC-ROC Curve")
plt.legend(loc="lower right")
plt.savefig('FigXX-KNNROC.png', dpi=600)
plt.show()

# %%
# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# MicroAvg calcola score di tutte le classi
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

_, ax = plt.subplots(figsize=(6, 4))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="MicroAvg", color="red")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Class {i+1}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([-.02, 1.02])
ax.set_ylim([0.0, 1.02])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("k-Nearest Neighbors Precision-Recall Curve")
plt.savefig('FigXX-KNNPR.png', dpi=600)
plt.show()

# %% [markdown]
# # **<font color="#42f5f5">3.0 OUTLIER DETECTION 1%</font>**

# %% [markdown]
# ## **<font color="#FBBF44">3.1 DBSCAN</font>**

# %%
# Calcoliamo i valori di distanza tra ogni record e il suo nearest neighbor
nbr = NearestNeighbors(n_neighbors=2)
nbrs = nbr.fit(X_train)
distances, indices = nbrs.kneighbors(X_train)

# Plottiamo la distanza dentro i valori del df e cerchiamo il "gomito" per vedere il punto di massima curvatura e quindi il valore ottimo di Eps
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(14,10))
plt.plot(distances, color = 'darkcyan')
plt.title('Fig. 4 - K-distance Graph to find optimal eps',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Eps',fontsize=14)
plt.savefig('FigXX-ElbowMethodDBSCAN.png', dpi=600)
plt.show()

# %%
dbscan = DBSCAN(eps=4.7,
                min_samples=2,
                n_jobs=-1
                )
dbscan.fit(X_train)

anomalies_db = where(dbscan.labels_==-1)
anomalies_db = X_train[anomalies_db]

# Vediamo i "cluster" che DBSCAN ha trovato e li contiamo, -1 e noise, quindi, outlier
np.unique(dbscan.labels_, return_counts=True)

# %% [markdown]
# ## **<font color="#FBBF44">3.2 Isolation Forest</font>**

# %%
isol = IsolationForest(bootstrap=True,
                       contamination=0.02,
                       max_samples=600,
                       n_estimators=1000,
                       n_jobs=-1
                       )
isol.fit(X_train)
outliers_isol = isol.predict(X_train)

anomalies_isol = where(outliers_isol==-1)
anomalies_isol = X_train[anomalies_isol]

np.unique(outliers_isol, return_counts=True)

# %% [markdown]
# ## **<font color="#FBBF44">3.3 ABOD</font>**

# %%
! pip install pyod
from pyod.models.abod import ABOD

# %%
abd = ABOD(n_neighbors=7,
           contamination=.02
           )
abd.fit(X_train)
outliers_abd = abd.predict(X_train)

anomalies_abd = where(outliers_abd==1)
anomalies_abd = X_train[anomalies_abd]

np.unique(outliers_abd, return_counts=True)

# %%
max_val = np.max(abd.decision_scores_[np.where(outliers_abd==1)])
min_val = np.min(abd.decision_scores_[np.where(outliers_abd==1)])

print(max_val)
print(min_val)

# %% [markdown]
# ## **<font color="#FBBF44">3.5 LOF</font>**

# %%
lof = LocalOutlierFactor(n_neighbors=7,
                         contamination=.02,
                         algorithm='kd_tree',
                         )
outliers_lof = lof.fit_predict(X_train)

anomalies_lof = where(outliers_lof==-1)
anomalies_lof = X_train[anomalies_lof]

np.unique(outliers_lof, return_counts=True)

# %%
max_val = np.max(lof.negative_outlier_factor_[np.where(outliers_lof==-1)])
min_val = np.min(lof.negative_outlier_factor_[np.where(outliers_lof==-1)])

print(max_val)
print(min_val)

plt.hist(lof.negative_outlier_factor_, bins=200)
plt.axvline(max_val, c='red')
plt.text(max_val, 250, 'outliers')
plt.title("Fig. 5 - LOF Outlier Factor Threshold")
plt.savefig('FigXX-LOFOutFactor.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">3.6 Dealing with Outliers</font>**

# %%
outliers_final_dbscan = where(dbscan.labels_ == -1)
outliers_final_isol = where(outliers_isol == -1)
outliers_final_abd = where(outliers_abd == 1)
outliers_final_lof = where(outliers_lof == -1)

tot = []
for x in outliers_final_dbscan:
    tot.extend(x)
for x in outliers_final_isol:
    tot.extend(x)
for x in outliers_final_abd:
    tot.extend(x)
for x in outliers_final_lof:
    tot.extend(x)
print(tot)

# %%
#cerco tutti i valori che appaiono nella lista degli outliers più di una volta e li inserisco in una lista per poi trasformarla in array
listadoppi = []
import collections
for item, count in collections.Counter(tot).items():
    if count > 2:
        listadoppi.append(item)
print(listadoppi)
finale_out = np.array(listadoppi)

anomalies_final = X_train[finale_out]
print(len(anomalies_final))

# %% [markdown]
# ### **<font color="#34eb89">4.6.1 Removing Outliers</font>**

# %%
df_wo_outliers = data.copy()
df_wo_outliers.drop(finale_out, inplace=True)
df_wo_outliers.shape

# %%
X_train2 = df_wo_outliers.values
y_train2 = y_train
y_train2 = np.delete(y_train2, finale_out)

# %%
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train2, y_train2)
y_pred = clf.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %%
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train2, y_train2)
y_pred = knn.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ### **<font color="#34eb89">4.6.2 Transforming Outliers</font>**

# %%
df_outliers_mean = data.copy()
df_outliers_mean.drop(finale_out, inplace=True)

means=[] #contiene le medie di tutte le 561 features del dataset SENZA outliers
for col in df_outliers_mean.columns:
    means.append(df_outliers_mean[col].mean())
len(df_outliers_mean)

# %%
for i in finale_out: # crea una riga in più per ogni outliers e ci inserisco means in tutte le featuers
    df_outliers_mean.loc[i] = means

df_outliers_mean.info()  # a questo punto sono stati reinseriti gli outliers con le medie "pulite" del train
# infatti nella cella prima sono 81 in meno, ora di nuovo 7352

# %%
df_outliers_mean = df_outliers_mean.sort_index()
df_outliers_mean.describe().T

# %%
X_train3 = df_outliers_mean.values
y_train3 = y_train

# %%
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train3, y_train3)
y_pred = clf.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %%
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train3, y_train3)
y_pred = knn.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ## **<font color="#FBBF44">3.7 Rechecking Outliers</font>**

# %% [markdown]
# ### **<font color="#34eb89">4.7.1 Preprocessing</font>**

# %%
# Cerchiamo valore ottimo di componenti
pca = PCA(n_components=10)
Principal_components=pca.fit_transform(X_train3)

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()

# %%
pca = PCA(n_components=2)
X_train3_pca = pca.fit_transform(X_train3)
X_train3_pca.shape

# %%
plt.scatter(X_train3_pca[:, 0], X_train3_pca[:, 1], c=y_train, cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
plt.show()

# %%
X_test_pca = pca.transform(X_test)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train3_pca, y_train3)

y_pred = clf.predict(X_test_pca)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ### **<font color="#34eb89">4.7.2 DBSCAN</font>**

# %%
# Calcoliamo i valori di distanza tra ogni record e il suo nearest neighbor
nbr = NearestNeighbors(n_neighbors=2)
nbrs = nbr.fit(X_train3_pca)
distances, indices = nbrs.kneighbors(X_train3_pca)

# Plottiamo la distanza dentro i valori del df e cerchiamo il "gomito" per vedere il punto di massima curvatura e quindi il valore ottimo di Eps
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Eps',fontsize=14)
plt.show()

# %%
dbscan = DBSCAN(eps=.3,
                min_samples=2,
                n_jobs=-1
                )
dbscan.fit(X_train3_pca)

# Vediamo i "cluster" che DBSCAN ha trovato e li contiamo, -1 e noise, quindi, outlier
np.unique(dbscan.labels_, return_counts=True)

# %%
anomalies_db = where(dbscan.labels_==-1)
anomalies_db = X_train3_pca[anomalies_db]

plt.scatter(X_train3_pca[:,0], X_train3_pca[:,1], s=2)
plt.scatter(anomalies_db[:,0], anomalies_db[:,1], color='red', s=2)
plt.show()

# %% [markdown]
# ### **<font color="#34eb89">4.7.3 Isolation Forest</font>**

# %%
# model = IsolationForest(random_state=47)

#  param_grid = {'n_estimators': [1000, 1500],
#                'max_samples': [10],
#                'contamination': ['auto', 0.0001, 0.0002],
#                'bootstrap': [True],
#                'n_jobs': [-1]}

#  grid_search = model_selection.GridSearchCV(model,
#                                             param_grid,
#                                             scoring="neg_mean_squared_error",
#                                             refit=True,
#                                             cv=10,
#                                             return_train_score=True)
# grid_search.fit(X_train_pca, y_train)

# best_model = grid_search.fit(X_train_pca, y_train)
# print('Optimum parameters', best_model.best_params_)

# %%
isol = IsolationForest(bootstrap=True,
                       contamination=0.02,
                       max_samples=600,
                       n_estimators=1000,
                       n_jobs=-1
                       )
isol.fit(X_train3_pca)
outliers_isol = isol.predict(X_train3_pca)
np.unique(outliers_isol, return_counts=True)

# %%
anomalies_isol = where(outliers_isol==-1)
anomalies_isol = X_train3_pca[anomalies_isol]

plt.scatter(X_train3_pca[:,0], X_train3_pca[:,1], s=2)
plt.scatter(anomalies_isol[:,0], anomalies_isol[:,1], color='red', s=2)
plt.show()

# %% [markdown]
# ### **<font color="#34eb89">4.3 ABOD</font>**

# %%
! pip install pyod
from pyod.models.abod import ABOD

# %%
param_grid = {'contamination': [0.1, 0.2,0.3,0.4,0.5],
              'n_neighbors': [5,6,7,8,9,10],
              'method': ["fast", "default"], }

# %%
abd = ABOD(n_neighbors=7,
           contamination=.02
           )
abd.fit(X_train3_pca)
outliers_abd = abd.predict(X_train3_pca)
np.unique(outliers_abd, return_counts=True)

# %%
# param_grid = {'contamination': [0.1, 0.2,0.3,0.4,0.5],
#               'n_neighbors': [5,6,7,8,9,10],
#               'method': ["fast", "default"], }

# abd_gridsearch=GridSearchCV(estimator=abd, param_grid=param_grid, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# print(abd_gridsearch.best_params_)

# %%
anomalies_abd = where(outliers_abd==1)
anomalies_abd = X_train3_pca[anomalies_abd]

plt.scatter(X_train3_pca[:,0], X_train3_pca[:,1], s=2)
plt.scatter(anomalies_abd[:,0], anomalies_abd[:,1], color='red', s=2)
plt.show()

# %% [markdown]
# ### **<font color="#34eb89">4.7.4 LOF</font>**

# %%
lof = LocalOutlierFactor(n_neighbors=98,
                         contamination=.02,
                         algorithm='kd_tree',
                         )
outliers_lof = lof.fit_predict(X_train3_pca)
np.unique(outliers_lof, return_counts=True)

# %%
max_val = np.max(lof.negative_outlier_factor_[np.where(outliers_lof==-1)])
min_val = np.min(lof.negative_outlier_factor_[np.where(outliers_lof==-1)])

print(max_val)
print(min_val)

# %%
plt.hist(lof.negative_outlier_factor_, bins=200)
plt.axvline(max_val, c='red')
plt.text(max_val, 250, 'outliers')
plt.show()

# %%
anomalies_lof = where(outliers_lof==-1)
anomalies_lof = X_train3_pca[anomalies_lof]

plt.scatter(X_train3_pca[:,0], X_train3_pca[:,1], s=2)
plt.scatter(anomalies_lof[:,0], anomalies_lof[:,1], color='red', s=2)
plt.show()

# %%


# %% [markdown]
# # **<font color="#42f5f5">4.0 IMBALANCED LEARNING</font>**

# %%
plt.figure(figsize=(8,4), tight_layout=True)
colors = sns.color_palette('mako')

classe, count = np.unique(y_train, return_counts=True)

plt.bar(classe, count, color=colors[0:6])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Training set Class Frequency Distribution')
plt.savefig('FigXX-TrainsetFrequency.png', dpi=600)
plt.show()

print(y_label.value_counts())

# %%
plt.figure(figsize=(8,4), tight_layout=True)
colors = sns.color_palette('mako')

classe, count = np.unique(y_test, return_counts=True)

plt.bar(classe, count, color=colors[0:6])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Testing set Class Frequency Distribution')
plt.savefig('FigXX-TestsetFrequency.png', dpi=600)
plt.show()

print(y_label.value_counts())

# %% [markdown]
# ## **<font color="#FBBF44">4.0.0 Preprocess</font>**

# %%
# Droppiamo le classi nel test per poi evaluare bene la performance del dataset ribilanciato
data_test_imb = data_test.copy()
data_test_imb['activity'] = y_label_test[0].values

classe1 = np.array(y_label_test[y_label_test[0]==1].index)
classe4 = np.array(y_label_test[y_label_test[0]==4].index)
classe5 = np.array(y_label_test[y_label_test[0]==5].index)
classe6 = np.array(y_label_test[ y_label_test[0]==6].index)
classes2remove = np.concatenate((classe1,classe4,classe5,classe6))

data_test_imb.drop(data_test_imb.index[classes2remove], inplace=True)
print("Test records: ", len(data_test_imb))
print(data_test_imb['activity'].value_counts())

X_test_imb = data_test_imb.iloc[:, 0:561].values
y_test_imb = data_test_imb['activity'].values

# %% [markdown]
# ### **<font color="#34eb89">4.0.1 Imbalancing for Oversampling</font>**

# %%
data_imb_over = data.copy()
data_imb_over['activity'] = y_label[0].values

# Seleziono le classi che voglio droppare. Lasciamo solo 2 e 3, 6 era troppo facile da predictare-->
classe1 = np.array(y_label[y_label[0]==1].index)
classe4 = np.array(y_label[y_label[0]==4].index)
classe5 = np.array(y_label[y_label[0]==5].index)
classe6 = np.array(y_label[ y_label[0]==6].index)
classes2remove = np.concatenate((classe1,classe4,classe5,classe6))

data_imb_over.drop(data_imb_over.index[classes2remove], inplace=True)
print("Records before imbalancing class 3: ", len(data_imb_over))
print(data_imb_over['activity'].value_counts())

# Records a droppare per sbilanciare la classe 3
rows2remove = np.random.choice((data_imb_over[data_imb_over['activity']==3].index), 942, replace=False)
print("\nRecords to remove from class 3: ", len(rows2remove))
data_imb_over.drop(rows2remove, inplace=True)

print("\nRecords after imbalancing class 3: ", len(data_imb_over))
print(data_imb_over['activity'].value_counts(), "\n")

X_train_imb_over = data_imb_over.iloc[:, 0:561].values
y_train_imb_over = data_imb_over['activity'].values

# X_train_imb, X_val_imb, y_train_imb, y_val_imb = train_test_split(data_imb1, y_label_imb, test_size=0.30, random_state= 8)

print(X_train_imb_over.shape)
print(X_train_imb_over.shape)

# %%
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_imb_over, y_train_imb_over)
y_pred_imb = clf.predict(X_test_imb)

print('Accuracy %s' % accuracy_score(y_test_imb, y_pred_imb))
print('F1-score %s' % f1_score(y_test_imb, y_pred_imb, average=None))
print(classification_report(y_test_imb, y_pred_imb))

# %%
plt.figure(figsize=(8,4), tight_layout=True)
colors = sns.color_palette('mako')

classe, count = np.unique(y_train_imb_over, return_counts=True)

plt.bar(classe, count, color=colors[3:4])
plt.xlabel('Class')
plt.xticks([2, 3])
plt.ylabel('Count')
plt.title('Frequency Distribution of Imbalanced Training set for Oversampling')
plt.savefig('FigXX-ImbalancedOverDistribution.png', dpi=600)
plt.show()

# %% [markdown]
# ### **<font color="#34eb89">4.0.2 Imbalancing for Undersampling</font>**

# %%
data_imb_under = data.copy()
data_imb_under['activity'] = y_label[0].values

# Seleziono le classi che voglio droppare. Lasciamo solo 2 e 3, 6 era troppo facile da predictare-->
classe1 = np.array(y_label[y_label[0]==1].index)
classe4 = np.array(y_label[y_label[0]==4].index)
classe5 = np.array(y_label[y_label[0]==5].index)
classe6 = np.array(y_label[ y_label[0]==6].index)
classes2remove = np.concatenate((classe1,classe4,classe5,classe6))

data_imb_under.drop(data_imb_under.index[classes2remove], inplace=True)
print("Records before imbalancing class 3: ", len(data_imb_under))
print(data_imb_under['activity'].value_counts())

# Copiamo e incolliamo la classe 2 per sbilanciarla
data_imb_under2 = data_imb_under[data_imb_under['activity']==2]
data_imb_under = data_imb_under.append(data_imb_under2, ignore_index = True)
data_imb_under = data_imb_under.append(data_imb_under2, ignore_index = True)
data_imb_under = data_imb_under.append(data_imb_under2, ignore_index = True)
data_imb_under = data_imb_under.append(data_imb_under2, ignore_index = True)
data_imb_under = data_imb_under.append(data_imb_under2, ignore_index = True)
data_imb_under = data_imb_under.append(data_imb_under2, ignore_index = True)
data_imb_under = data_imb_under.append(data_imb_under2, ignore_index = True)

print("\nRecords after imbalancing class 3: ", len(data_imb_under))
print(data_imb_under['activity'].value_counts(), "\n")

X_train_imb_under = data_imb_under.iloc[:, 0:561].values
y_train_imb_under = data_imb_under['activity'].values

# X_train_imb, X_val_imb, y_train_imb, y_val_imb = train_test_split(data_imb1, y_label_imb, test_size=0.30, random_state= 8)

print(X_train_imb_under.shape)
print(y_train_imb_under.shape)

# %%
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_imb_under, y_train_imb_under)
y_pred_imb = clf.predict(X_test_imb)

print('Accuracy %s' % accuracy_score(y_test_imb, y_pred_imb))
print('F1-score %s' % f1_score(y_test_imb, y_pred_imb, average=None))
print(classification_report(y_test_imb, y_pred_imb))

# %%
plt.figure(figsize=(8,4), tight_layout=True)
colors = sns.color_palette('mako')

classe, count = np.unique(y_train_imb_under, return_counts=True)

plt.bar(classe, count, color=colors[3:4])
plt.xlabel('Class')
plt.xticks([2, 3])
plt.ylabel('Count')
plt.title('Frequency Distribution of Imbalanced Training set for Undersampling')
plt.savefig('FigXX-ImbalancedUnderDistribution.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">4.1 Oversampling</font>**

# %%
# Prima  dell'oversampling
pca = PCA(n_components=2)
X_train_imbover_pca = pca.fit_transform(X_train_imb_over)

classes = ['2', '3']
scatter = plt.scatter(X_train_imbover_pca[:, 0], X_train_imbover_pca[:, 1], c=y_train_imb_over, cmap=plt.cm.tab20b, edgecolor='white', linewidth = .7, alpha = .7)
plt.title("Imbalanced Classes before oversampling")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig('FigXX-ImbalancedOversampling.png', dpi=600)

# %% [markdown]
# ### **<font color="#34eb89">4.1.1 Random Over Sampler</font>**

# %%
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train_imb_over, y_train_imb_over)

print('Oversampled y_train %s' % Counter(y_train_ros))

# %%
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_ros, y_train_ros)

y_pred = clf.predict(X_test_imb)

print('Accuracy %s' % accuracy_score(y_test_imb, y_pred))
print('F1-score %s' % f1_score(y_test_imb, y_pred, average=None))
print(classification_report(y_test_imb, y_pred))

# %%
# Dopo l'oversampling
pca = PCA(n_components=2)
X_train_ros_pca = pca.fit_transform(X_train_ros)

classes = ['2', '3']
scatter = plt.scatter(X_train_ros_pca[:, 0], X_train_ros_pca[:, 1], c=y_train_ros, cmap=plt.cm.tab20b, edgecolor='white', linewidth = .7, alpha = .7)
plt.title("Rebalanced Classes using RandomOverSampler")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig('FigXX-ImbalancedOversamplingROS.png', dpi=600)

# %% [markdown]
# ### **<font color="#34eb89">4.1.2 SMOTE</font>**

# %%
sm = SMOTE(random_state=42, k_neighbors = 1)
X_train_sm, y_train_sm = sm.fit_resample(X_train_imb_over, y_train_imb_over)

print('Oversampled y_train %s' % Counter(y_train_sm))

# %%
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_sm, y_train_sm)

y_pred = clf.predict(X_test_imb)

print('Accuracy %s' % accuracy_score(y_test_imb, y_pred))
print('F1-score %s' % f1_score(y_test_imb, y_pred, average=None))
print(classification_report(y_test_imb, y_pred))

# %%
# Dopo l'oversampling
pca = PCA(n_components=2)
X_train_sm_pca = pca.fit_transform(X_train_sm)
print(X_train_sm_pca.shape)

classes = ['2', '3']
scatter = plt.scatter(X_train_sm_pca[:, 0], X_train_sm_pca[:, 1], c=y_train_sm, cmap=plt.cm.tab20b, edgecolor='white', linewidth = .7, alpha = .7)
plt.title("Rebalanced Classes using SMOTE")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig('FigXX-ImbalancedOversamplingSMOTE.png', dpi=600)

# %% [markdown]
# ## **<font color="#FBBF44">4.2 Undersampling</font>**

# %%
# Prima  dell'undersampling
pca = PCA(n_components=2)
X_train_imbunder_pca = pca.fit_transform(X_train_imb_under)

classes = ['2', '3']
scatter = plt.scatter(X_train_imbunder_pca[:, 0], X_train_imbunder_pca[:, 1], c=y_train_imb_under, cmap=plt.cm.tab20b, edgecolor='white', linewidth = .7, alpha = .7)
plt.title("Imbalanced Classes before undersampling")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig('FigXX-ImbalancedUndersampling.png', dpi=600)

# %% [markdown]
# ### **<font color="#34eb89">4.2.1 Random Under Sampler</font>**

# %%
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_imb_under, y_train_imb_under)

print('Undersampled y_train %s' % Counter(y_train_rus))

# %%
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_rus, y_train_rus)

y_pred = clf.predict(X_test_imb)

print('Accuracy %s' % accuracy_score(y_test_imb, y_pred))
print('F1-score %s' % f1_score(y_test_imb, y_pred, average=None))
print(classification_report(y_test_imb, y_pred))

# %%
# Dopo l'undersampling
pca = PCA(n_components=2)
X_train_rus_pca = pca.fit_transform(X_train_rus)

classes = ['2', '3']
scatter = plt.scatter(X_train_rus_pca[:, 0], X_train_rus_pca[:, 1], c=y_train_rus, cmap=plt.cm.tab20b, edgecolor='white', linewidth = .7, alpha = .7)
plt.title("Rebalanced Classes using RandomUnderSampling")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig('FigXX-ImbalancedUndersamplingRUS.png', dpi=600)

# %% [markdown]
# ### **<font color="#34eb89">4.2.2 Condensed Nearest Neighbor</font>**

# %%
# cnn = CondensedNearestNeighbour(n_neighbors=98, sampling_strategy='all')
# X_train_cnn, y_train_cnn = cnn.fit_resample(X_train_imb_under, y_train_imb_under)

# print('Undersampled y_train %s' % Counter(y_train_cnn))

# %%
# clf = DecisionTreeClassifier(random_state=0)
# clf.fit(X_train_cnn, y_train_cnn)

# y_pred = clf.predict(X_test_cnn)

# print('Accuracy %s' % accuracy_score(y_test_cnn, y_pred))
# print('F1-score %s' % f1_score(y_test_cnn, y_pred, average=None))
# print(classification_report(y_test_cnn, y_pred))

# %%
# pca = PCA(n_components=2)
# X_train_cnn_pca = pca.fit_transform(X_train_cnn)
# print(X_train_cnn_pca.shape)

# plt.scatter(X_train_cnn_pca[:, 0], X_train_cnn_pca[:, 1], c=y_train_cnn, cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
# plt.show()

# %% [markdown]
# # **<font color="#42f5f5">5.0 EXPLOIT DIMENSIONALITY REDUCTION</font>**
#

# %% [markdown]
# ## **<font color="#ff9ed0">-------------- Feature Selection --------------</font>**
#

# %% [markdown]
# ## **<font color="#FBBF44">5.1 Variance Threshold</font>**

# %%
#Rimaniamo solo con le colonne con high variance, togliamo le colonne che sono almeno 75% similari, <.25
sel_var = VarianceThreshold(threshold=.10)
X_train_sel_var = sel_var.fit_transform(X_train)

selected = sel_var.get_support()
features = array(feature_list)

# print("Selected features:")
# print(features[selected])

X_train_sel_var.shape

# %%
X_test_sel_var = sel_var.transform(X_test)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_sel_var, y_train)

y_pred = clf.predict(X_test_sel_var)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %%
knn = KNeighborsClassifier()
knn.fit(X_train_sel_var, y_train)
y_pred = knn.predict(X_test_sel_var)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits = 4))

# %% [markdown]
# ## **<font color="#FBBF44">5.2 Univariate Feature Selection</font>**

# %%
normalized_df = (data-data.min())/(data.max()-data.min())
normalized_df_test = (data_test-data_test.min())/(data_test.max()-data_test.min())

X = normalized_df.iloc[:,0:561]
y = y_label

Xtest = normalized_df_test.iloc[:,0:561]
ytest = y_label_test

# %%
sel_uni = SelectKBest(score_func=chi2, k=131)
X_train_sel_uni = sel_uni.fit(X,y)

dfscores = pd.DataFrame(X_train_sel_uni.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']

# print(featureScores.nlargest(35,'Score'))

X_test_sel_uni = sel_uni.fit_transform(Xtest,ytest)

# %%
# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')
#     acc = []
#     for i in range(1,200):
#         sel_uni = SelectKBest(score_func=chi2, k=i)
#         X_train_sel_uni = sel_uni.fit_transform(X,y)
#         X_test_sel_uni = sel_uni.transform(Xtest)

#         clf = DecisionTreeClassifier(random_state=0).fit(X_train_sel_uni,y_train)
#         y_pred = clf.predict(X_test_sel_uni)
#         acc.append(metrics.accuracy_score(y_test, y_pred))

#     print("Maximum accuracy:-",max(acc),"at Component =",acc.index(max(acc)))

# %%
    plt.figure(figsize=(10,6))
    plt.plot(range(1,200),acc,color = 'darkslategray', marker='o',markerfacecolor='paleturquoise', markersize=10)
    plt.title('Fig. 9 - Optimal K for Univariate Feature Selection')
    plt.savefig('FigXX-KUNIVA.png', dpi=600)
    plt.ylabel('Accuracy')

# %%
X_test_sel_uni = sel_uni.transform(Xtest)
X_train_sel_uni = sel_uni.transform(X)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_sel_uni, y_train)

y_pred = clf.predict(X_test_sel_uni)

print('Accuracy %s' % accuracy_score(ytest, y_pred))
print('F1-score %s' % f1_score(ytest, y_pred, average=None))
print(classification_report(ytest, y_pred, digits=4))

# %% [markdown]
# ## **<font color="#FBBF44">5.3 Select from Model</font>**

# %%
sel_mod = SelectFromModel(LogisticRegression())
X_train_sel_mod = sel_mod.fit_transform(X_train, y_train)

selected = sel_mod.get_support()
features = array(feature_list)

print("Selected features:")
print(features[selected])

X_train_sel_mod.shape

# %%
X_test_sel_mod = sel_mod.transform(X_test)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_sel_mod, y_train)

y_pred = clf.predict(X_test_sel_mod)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ## **<font color="#ff9ed0">-------------- Feature Projection --------------</font>**
#

# %% [markdown]
# ## **<font color="#FBBF44">5.4 Principal Component Analysis (PCA)</font>**

# %%
# Cerchiamo valore ottimo di componenti
pca = PCA(n_components=25)
Principal_components=pca.fit_transform(X_train)

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2, c='c')
plt.title('Fig. 10 - PCA Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.savefig('FigXX-PCAScreePlot.png', dpi=600)
plt.show()

# %%
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

classes = ['1', '2', '3', '4', '5', '6']
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=plt.cm.Set3, edgecolor='white', linewidth = .7, alpha = .7)
plt.title("PCA")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig('FigXX-PCA.png', dpi=600)

# %%
pca = PCA(n_components=3)
X_train_pca3 = pca.fit_transform(X_train)
X_train_pca3.shape

x = X_train_pca3[:, 0]
y = X_train_pca3[:, 1]
z = X_train_pca3[:, 2]

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter3D(x, y, z, c=y_train)
ax.set_xlabel('PC1', fontweight ='bold')
ax.set_ylabel('PC2', fontweight ='bold')
ax.set_zlabel('PC3', fontweight ='bold')
plt.title("PCA With 3 components")
plt.savefig('FigXXPCA3D.png', dpi=600)
plt.show()

# %%
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    acc = []
    for i in range(1,50):
        pca = PCA(n_components=i)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf = DecisionTreeClassifier(random_state=0).fit(X_train_pca,y_train)
        y_pred = clf.predict(X_test_pca)
        acc.append(metrics.accuracy_score(y_test, y_pred))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,50),acc,color = 'darkslategray', marker='o',markerfacecolor='paleturquoise', markersize=10)
    plt.title('Optimal Number of Components for PCA')
    plt.ylabel('Accuracy')
    print("Maximum accuracy:-",max(acc),"at Component =",acc.index(max(acc)))

# %%
pca = PCA(n_components=22)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ## **<font color="#FBBF44">5.4 Principal Component Analysis (PCA) USANDO DATASET DA SELECTION MODEL</font>**

# %%
pca = PCA(n_components=2)
X_train_pca_mod2 = pca.fit_transform(X_train_sel_mod)

classes = ['1', '2', '3', '4', '5', '6']
scatter = plt.scatter(X_train_pca_mod2[:, 0], X_train_pca_mod2[:, 1], c=y_train, cmap=plt.cm.Set3, edgecolor='white', linewidth = .7, alpha = .7)
plt.title("PCA using Selected Features")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig('FigXX-PCAModelSelect.png', dpi=600)

# %%
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    acc = []
    for i in range(1,50):
        pca = PCA(n_components=i)
        X_train_pca = pca.fit_transform(X_train_sel_mod)
        X_test_pca = pca.transform(X_test_sel_mod)

        clf = DecisionTreeClassifier(random_state=0).fit(X_train_pca,y_train)
        y_pred = clf.predict(X_test_pca)
        acc.append(metrics.accuracy_score(y_test, y_pred))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,50),acc,color = 'darkslategray', marker='o',markerfacecolor='paleturquoise', markersize=10)
    plt.title('Optimal Number of Components for PCA')
    plt.ylabel('Accuracy')
    print("Maximum accuracy:-",max(acc),"at Component =",acc.index(max(acc)))

# %%
pca = PCA(n_components=17)
X_train_pca_mod2 = pca.fit_transform(X_train_sel_mod)
X_train_pca_mod2.shape

# %%
X_test_pca_mod2 = pca.transform(X_test_sel_mod)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_pca_mod2, y_train)

y_pred = clf.predict(X_test_pca_mod2)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ## **<font color="#FBBF44">5.5 Gaussian Random Projection</font>**

# %%
rsp = random_projection.GaussianRandomProjection(n_components=2)
X_train_rsp = rsp.fit_transform(X_train)

classes = ['1', '2', '3', '4', '5', '6']
scatter = plt.scatter(X_train_rsp[:, 0], X_train_rsp[:, 1], c=y_train, cmap=plt.cm.Set3, edgecolor='white', linewidth = .7, alpha = .7)
plt.title("Gaussian Random Projection")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig('FigXX-GRP.png', dpi=600)

# %%
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    acc = []
    for i in range(1,50):
        rsp = random_projection.GaussianRandomProjection(n_components=i)
        X_train_rsp = rsp.fit_transform(X_train)
        X_test_rsp = rsp.transform(X_test)

        clf = DecisionTreeClassifier(random_state=0).fit(X_train_rsp,y_train)
        y_pred = clf.predict(X_test_rsp)
        acc.append(metrics.accuracy_score(y_test, y_pred))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,50),acc,color = 'darkslategray', marker='o',markerfacecolor='paleturquoise', markersize=10)
    plt.title('Optimal Number of Components for GRP')
    plt.ylabel('Accuracy')
    print("Maximum accuracy:-",max(acc),"at Component =",acc.index(max(acc)))

# %%
rsp = random_projection.GaussianRandomProjection(n_components=27)
X_train_rsp = rsp.fit_transform(X_train)
X_test_rsp = rsp.transform(X_test)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_rsp, y_train)

y_pred = clf.predict(X_test_rsp)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ## **<font color="#FBBF44">5.6 Multi Dimensional Scaling (MDS)</font>**

# %% [markdown]
# ### **<font color="#34eb89">3.6.1 Classic MDS</font>**

# %%
#mds = MDS(n_components=2)
#X_train_mds = mds.fit_transform(X_train)
#X_train_mds.shape

# %%
#plt.scatter(X_train_mds[:, 0], X_train_mds[:, 1], c=y_train, cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
#plt.show()

# %%
#clf = DecisionTreeClassifier(random_state=0)
#clf.fit(X_train_mds, y_train)

#y_pred = clf.predict(X_train_mds)

#print('Accuracy %s' % accuracy_score(y_train, y_pred))
#print('F1-score %s' % f1_score(y_train, y_pred, average=None))
#print(classification_report(y_train, y_pred))

# %% [markdown]
# ### **<font color="#34eb89">3.6.2 ISOmap</font>**
#

# %%
iso = Isomap(n_components=2)
X_train_iso = iso.fit_transform(X_train)

classes = ['1', '2', '3', '4', '5', '6']
scatter = plt.scatter(X_train_iso[:, 0], X_train_iso[:, 1], c=y_train, cmap=plt.cm.Set3, edgecolor='white', linewidth = .7, alpha = .7)
plt.title("Isomap")
plt.ylim([-45, 40])
plt.legend(handles=scatter.legend_elements()[0], labels=classes, ncol=6, loc='lower left')
plt.savefig('FigXX-Isomap.png', dpi=600)

# %%
# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')
#     acc = []
#     for i in range(1,50):
#         iso = Isomap(n_components=i)
#         X_train_iso = iso.fit_transform(X_train)
#         X_test_iso = iso.transform(X_test)

#         clf = DecisionTreeClassifier(random_state=0).fit(X_train_iso,y_train)
#         y_pred = clf.predict(X_test_iso)
#         acc.append(metrics.accuracy_score(y_test, y_pred))
#         print(i, "accuracy = ", acc)

#     plt.figure(figsize=(10,6))
#     plt.plot(range(1,50),acc,color = 'darkslategray', marker='o',markerfacecolor='paleturquoise', markersize=10)
#     plt.title('Optimal Number of Components for ISOmap')
#     plt.ylabel('Accuracy')
#     print("Maximum accuracy:-",max(acc),"at Component =",acc.index(max(acc)))

# %%
iso = Isomap(n_components=75)
X_train_iso = iso.fit_transform(X_train)
X_test_iso = iso.fit_transform(X_test)

# %%
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_iso, y_train)

y_pred = clf.predict(X_test_iso)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ### **<font color="#34eb89">3.6.3 t-SNE</font>**

# %%
tsne = TSNE(n_components=2)
X_train_tsne = tsne.fit_transform(X_train)

classes = ['1', '2', '3', '4', '5', '6']
scatter = plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train, cmap=plt.cm.Set3, edgecolor='white', linewidth = .7, alpha = .7)
plt.title("t-SNE")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig('FigXX-tSNE.png', dpi=600)

# %%
X_test_tsne = tsne.fit_transform(X_test)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_tsne, y_train)

y_pred = clf.predict(X_test_tsne)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ## **<font color="#ff9ed0">-------------- Outlier Detection con Dimensionality Reduction --------------</font>**
#

# %%
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# %% [markdown]
# ## **<font color="#FBBF44">5.8 DBSCAN</font>**

# %%
# Calcoliamo i valori di distanza tra ogni record e il suo nearest neighbor
nbr = NearestNeighbors(n_neighbors=2)
nbrs = nbr.fit(X_train_pca)
distances, indices = nbrs.kneighbors(X_train_pca)

# Plottiamo la distanza dentro i valori del df e cerchiamo il "gomito" per vedere il punto di massima curvatura e quindi il valore ottimo di Eps
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances, color = 'darkcyan')
plt.title('K-distance Graph',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Eps',fontsize=14)
plt.savefig('FigXX-ElbowMethodDBSCAN.png', dpi=600)
plt.show()

# %%
dbscan = DBSCAN(eps=.18,
                min_samples=2,
                n_jobs=-1
                )
dbscan.fit(X_train_pca)

# Vediamo i "cluster" che DBSCAN ha trovato e li contiamo, -1 e noise, quindi, outlier
np.unique(dbscan.labels_, return_counts=True)

# %%
anomalies_db = where(dbscan.labels_==-1)
anomalies_db = X_train_pca[anomalies_db]

plt.scatter(X_train_pca[:,0], X_train_pca[:,1], color='skyblue', s=2)
plt.scatter(anomalies_db[:,0], anomalies_db[:,1], color='orangered', s=2)
plt.title("Outliers found by DBSCAN")
plt.savefig('FigXX-OutlierDBSCAN.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">5.9 Isolation Forest</font>**

# %%
# model = IsolationForest(random_state=47)

#  param_grid = {'n_estimators': [1000, 1500],
#                'max_samples': [10],
#                'contamination': ['auto', 0.0001, 0.0002],
#                'bootstrap': [True],
#                'n_jobs': [-1]}

#  grid_search = model_selection.GridSearchCV(model,
#                                             param_grid,
#                                             scoring="neg_mean_squared_error",
#                                             refit=True,
#                                             cv=10,
#                                             return_train_score=True)
# grid_search.fit(X_train_pca, y_train)

# best_model = grid_search.fit(X_train_pca, y_train)
# print('Optimum parameters', best_model.best_params_)

# %%
isol = IsolationForest(bootstrap=True,
                       contamination=0.02,
                       max_samples=600,
                       n_estimators=1000,
                       n_jobs=-1
                       )
isol.fit(X_train_pca)
outliers_isol = isol.predict(X_train_pca)
np.unique(outliers_isol, return_counts=True)


# %%
anomalies_isol = where(outliers_isol==-1)
anomalies_isol = X_train_pca[anomalies_isol]

plt.scatter(X_train_pca[:,0], X_train_pca[:,1], color='skyblue', s=2)
plt.scatter(anomalies_isol[:,0], anomalies_isol[:,1], color='orangered', s=2)
plt.title("Outliers found by Isolation Forest")
plt.savefig('FigXX-OutlierIsol.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">5.10 ABOD</font>**

# %%
! pip install pyod
from pyod.models.abod import ABOD

# %%
param_grid = {'contamination': [0.1, 0.2,0.3,0.4,0.5],
              'n_neighbors': [5,6,7,8,9,10],
              'method': ["fast", "default"], }

# %%
abd = ABOD(n_neighbors=7,
           contamination=.02
           )
abd.fit(X_train_pca)
outliers_abd = abd.predict(X_train_pca)
np.unique(outliers_abd, return_counts=True)

# %%
# param_grid = {'contamination': [0.1, 0.2,0.3,0.4,0.5],
#               'n_neighbors': [5,6,7,8,9,10],
#               'method': ["fast", "default"], }

# abd_gridsearch=GridSearchCV(estimator=abd, param_grid=param_grid, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# print(abd_gridsearch.best_params_)

# %%
max_val = np.max(abd.decision_scores_[np.where(outliers_abd==1)])
min_val = np.min(abd.decision_scores_[np.where(outliers_abd==1)])

print(max_val)
print(min_val)

# %%
anomalies_abd = where(outliers_abd==1)
anomalies_abd = X_train_pca[anomalies_abd]

plt.scatter(X_train_pca[:,0], X_train_pca[:,1], color='skyblue', s=2)
plt.scatter(anomalies_abd[:,0], anomalies_abd[:,1], color='orangered', s=2)
plt.title("Outliers found by ABOD")
plt.savefig('FigXX-OutlierABOD.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">5.11 GRUBB'S</font>**

# %%
# ! pip install outlier_utils
# from outliers import smirnov_grubbs as grubbs

# %%
# #flat_list = [item for sublist in t for item in sublist]
# flat_list = []
# for sublist in X_train:
#     for item in sublist:
#         flat_list.append(item)
# len(flat_list)

# %%
# dataG=flat_list
# grubbs.test(dataG, alpha=.05)

# %%
# var=500

# for i in range(var):
#   out1=grubbs.max_test_indices(X_train[:,i], alpha=.05)

# len(out1)

# %%
# anomalies_gb = X_train_pca[out1]

# plt.scatter(X_train_pca[:,0], X_train_pca[:,1], s=2)
# plt.scatter(anomalies_gb[:,0], anomalies_gb[:,1], color='red', s=2)
# plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">5.12 LOF</font>**

# %%
lof = LocalOutlierFactor(n_neighbors=98,
                         contamination=.02,
                         algorithm='kd_tree',
                         )
outliers_lof = lof.fit_predict(X_train_pca)
np.unique(outliers_lof, return_counts=True)

# %%
max_val = np.max(lof.negative_outlier_factor_[np.where(outliers_lof==-1)])
min_val = np.min(lof.negative_outlier_factor_[np.where(outliers_lof==-1)])

print(max_val)
print(min_val)

# %%
plt.hist(lof.negative_outlier_factor_, bins=200)
plt.axvline(max_val, c='red')
plt.text(max_val, 250, 'outliers')
plt.show()

# %%
anomalies_lof = where(outliers_lof==-1)
anomalies_lof = X_train_pca[anomalies_lof]

plt.scatter(X_train_pca[:,0], X_train_pca[:,1], color='skyblue', s=2)
plt.scatter(anomalies_lof[:,0], anomalies_lof[:,1], color='orangered', s=2)
plt.title("Outliers found by Local Outlier Factor")
plt.savefig('FigXX-OutlierLOF.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">5.13 Dealing with Outliers</font>**

# %%
outliers_final_dbscan = where(dbscan.labels_ == -1)
outliers_final_isol = where(outliers_isol == -1)
outliers_final_abd = where(outliers_abd == 1)
outliers_final_lof = where(outliers_lof == -1)

# %%
tot = []
for x in outliers_final_dbscan:
    tot.extend(x)
for x in outliers_final_isol:
    tot.extend(x)
for x in outliers_final_abd:
    tot.extend(x)
for x in outliers_final_lof:
    tot.extend(x)
print(tot)

# %%
# cerco tutti i valori che appaiono nella lista degli outliers più di una volta e li inserisco in una lista per poi trasformarla in array
listadoppi = []
import collections
for item, count in collections.Counter(tot).items():
    if count > 2:
        listadoppi.append(item)
print(listadoppi)
finale_out = np.array(listadoppi)


# %%
anomalies_final = X_train_pca[finale_out]
print(len(anomalies_final))

plt.scatter(X_train_pca[:,0], X_train_pca[:,1], color='skyblue', s=2)
plt.scatter(anomalies_final[:,0], anomalies_final[:,1], color='orangered', s=2)
plt.title("Outliers found by at least 3/4 methods")
plt.savefig('FigXX-OutlierFinal.png', dpi=600)
plt.show()

# %% [markdown]
# ### **<font color="#34eb89">4.6.1 Removing Outliers</font>**

# %%
df_wo_outliers = data.copy()
df_wo_outliers.drop(finale_out, inplace=True)
df_wo_outliers.describe().T

# %%
X_train2 = df_wo_outliers.values
y_train2 = y_train
y_train2 = np.delete(y_train2, finale_out)

# %%
pca = PCA(n_components=2)
X_train_pca2 = pca.fit_transform(X_train2)
X_train_pca2.shape

# %%
plt.scatter(X_train_pca2[:,0], X_train_pca2[:,1], color='skyblue', s=2)
plt.title("Outliers Removed")
plt.ylim(-5.5,9)
plt.xlim(-8,20.5)
plt.savefig('FigXX-OutlierRemoved.png', dpi=600)
plt.show()

# %%
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train2, y_train2)
y_pred = clf.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %%
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train2, y_train2)
y_pred = knn.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %%
classes = ['1', '2', '3', '4', '5', '6']
scatter = plt.scatter(X_train_pca2[:, 0], X_train_pca2[:, 1], c=y_train2, cmap=plt.cm.Set3, edgecolor='white', linewidth = .7, alpha = .7)
plt.title("PCA after removing Outleiers")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig('FigXX-PCAOutliersRemoved.png', dpi=600)

# %% [markdown]
# ### **<font color="#34eb89">4.6.2 Transforming Outliers</font>**

# %%
df_outliers_mean = data.copy()
df_outliers_mean.drop(finale_out, inplace=True)

means=[] #contiene le medie di tutte le 561 features del dataset SENZA outliers
for col in df_outliers_mean.columns:
    means.append(df_outliers_mean[col].mean())
len(df_outliers_mean)

# %%
for i in finale_out: # crea una riga in più per ogni outliers e ci inserisco means in tutte le featuers
    df_outliers_mean.loc[i] = means

df_outliers_mean.info()  # a questo punto sono stati reinseriti gli outliers con le medie "pulite" del train
# infatti nella cella prima sono 81 in meno, ora di nuovo 7352

# %%
df_outliers_mean = df_outliers_mean.sort_index()
df_outliers_mean.describe().T

# %%
X_train3 = df_outliers_mean.values
y_train3 = y_train

# %%
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train3, y_train3)
y_pred = clf.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %%
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train3, y_train3)
y_pred = knn.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ## **<font color="#FBBF44">5.14 Rechecking Outliers</font>**

# %% [markdown]
# ### **<font color="#34eb89">4.7.1 Preprocessing</font>**

# %%
# Cerchiamo valore ottimo di componenti
pca = PCA(n_components=10)
Principal_components=pca.fit_transform(X_train3)

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()

# %%
pca = PCA(n_components=2)
X_train3_pca = pca.fit_transform(X_train3)
X_train3_pca.shape

# %%
plt.scatter(X_train3_pca[:, 0], X_train3_pca[:, 1], c=y_train, cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
plt.show()

# %%
X_test_pca = pca.transform(X_test)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train3_pca, y_train3)

y_pred = clf.predict(X_test_pca)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ### **<font color="#34eb89">4.7.2 DBSCAN</font>**

# %%
# Calcoliamo i valori di distanza tra ogni record e il suo nearest neighbor
nbr = NearestNeighbors(n_neighbors=2)
nbrs = nbr.fit(X_train3_pca)
distances, indices = nbrs.kneighbors(X_train3_pca)

# Plottiamo la distanza dentro i valori del df e cerchiamo il "gomito" per vedere il punto di massima curvatura e quindi il valore ottimo di Eps
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Eps',fontsize=14)
plt.show()

# %%
dbscan = DBSCAN(eps=.3,
                min_samples=2,
                n_jobs=-1
                )
dbscan.fit(X_train3_pca)

# Vediamo i "cluster" che DBSCAN ha trovato e li contiamo, -1 e noise, quindi, outlier
np.unique(dbscan.labels_, return_counts=True)

# %%
anomalies_db = where(dbscan.labels_==-1)
anomalies_db = X_train3_pca[anomalies_db]

plt.scatter(X_train3_pca[:,0], X_train3_pca[:,1], s=2)
plt.scatter(anomalies_db[:,0], anomalies_db[:,1], color='red', s=2)
plt.show()

# %% [markdown]
# ### **<font color="#34eb89">4.7.3 Isolation Forest</font>**

# %%
# model = IsolationForest(random_state=47)

#  param_grid = {'n_estimators': [1000, 1500],
#                'max_samples': [10],
#                'contamination': ['auto', 0.0001, 0.0002],
#                'bootstrap': [True],
#                'n_jobs': [-1]}

#  grid_search = model_selection.GridSearchCV(model,
#                                             param_grid,
#                                             scoring="neg_mean_squared_error",
#                                             refit=True,
#                                             cv=10,
#                                             return_train_score=True)
# grid_search.fit(X_train_pca, y_train)

# best_model = grid_search.fit(X_train_pca, y_train)
# print('Optimum parameters', best_model.best_params_)

# %%
isol = IsolationForest(bootstrap=True,
                       contamination=0.02,
                       max_samples=600,
                       n_estimators=1000,
                       n_jobs=-1
                       )
isol.fit(X_train3_pca)
outliers_isol = isol.predict(X_train3_pca)
np.unique(outliers_isol, return_counts=True)

# %%
anomalies_isol = where(outliers_isol==-1)
anomalies_isol = X_train3_pca[anomalies_isol]

plt.scatter(X_train3_pca[:,0], X_train3_pca[:,1], s=2)
plt.scatter(anomalies_isol[:,0], anomalies_isol[:,1], color='red', s=2)
plt.show()

# %% [markdown]
# ### **<font color="#34eb89">4.3 ABOD</font>**

# %%
! pip install pyod
from pyod.models.abod import ABOD

# %%
param_grid = {'contamination': [0.1, 0.2,0.3,0.4,0.5],
              'n_neighbors': [5,6,7,8,9,10],
              'method': ["fast", "default"], }

# %%
abd = ABOD(n_neighbors=7,
           contamination=.02 #Tentativo di GridSearch per ABOD
           )
abd.fit(X_train3_pca)
outliers_abd = abd.predict(X_train3_pca)
np.unique(outliers_abd, return_counts=True)

# %%
# param_grid = {'contamination': [0.1, 0.2,0.3,0.4,0.5],
#               'n_neighbors': [5,6,7,8,9,10],
#               'method': ["fast", "default"], }

# abd_gridsearch=GridSearchCV(estimator=abd, param_grid=param_grid, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# print(abd_gridsearch.best_params_)

# %%
anomalies_abd = where(outliers_abd==1)
anomalies_abd = X_train3_pca[anomalies_abd]

plt.scatter(X_train3_pca[:,0], X_train3_pca[:,1], s=2)
plt.scatter(anomalies_abd[:,0], anomalies_abd[:,1], color='red', s=2)
plt.show()

# %% [markdown]
# ### **<font color="#34eb89">4.7.4 LOF</font>**

# %%
lof = LocalOutlierFactor(n_neighbors=98,
                         contamination=.02,
                         algorithm='kd_tree',
                         )
outliers_lof = lof.fit_predict(X_train3_pca)
np.unique(outliers_lof, return_counts=True)

# %%
max_val = np.max(lof.negative_outlier_factor_[np.where(outliers_lof==-1)])
min_val = np.min(lof.negative_outlier_factor_[np.where(outliers_lof==-1)])

print(max_val)
print(min_val)

# %%
plt.hist(lof.negative_outlier_factor_, bins=200)
plt.axvline(max_val, c='red')
plt.text(max_val, 250, 'outliers')
plt.show()

# %%
anomalies_lof = where(outliers_lof==-1)
anomalies_lof = X_train3_pca[anomalies_lof]

plt.scatter(X_train3_pca[:,0], X_train3_pca[:,1], s=2)
plt.scatter(anomalies_lof[:,0], anomalies_lof[:,1], color='red', s=2)
plt.show()

# %% [markdown]
# # **<font color="#42f5f5">6.0 ADVANCED CLASSIFICATION</font>**

# %% [markdown]
# ## **<font color="#FBBF44">6.0.0 Super Model</font>**

# %% [markdown]
# ### **<font color="#34eb89">Crossvalidation</font>**

# %%
# Lista dei modeli a utilizzare
model_dt = DecisionTreeClassifier()
model_knn = KNeighborsClassifier()
model_gaussnb = GaussianNB()
model_bernb = BernoulliNB()
model_logreg = LogisticRegression()
model_linsvc = LinearSVC()
model_svc = SVC()
model_mlp = MLPClassifier()
model_rfc = RandomForestClassifier()
model_etc = ExtraTreesClassifier()
model_bag = BaggingClassifier()
model_gbc = GradientBoostingClassifier()
model_xgb = XGBClassifier()

# Dizionario a percorrere per chiamare ogni modello
models = {
    'Decision Tree': model_dt,
    'K Neighbors': model_knn,
    'Gaussian Naive Bayes': model_gaussnb,
    'Bernoulli Naive Bayes': model_bernb,
    'Logistic Regression': model_logreg,
    'Linear SVM': model_linsvc,
    'SVM': model_svc,
    'MLP': model_mlp,
    'Random Forest': model_rfc,
    'ExtraTrees': model_etc,
    'Bagging': model_bag,
    'GradientBoost': model_gbc,
    'XGBoost': model_xgb
    }

# Lista vuota dove mettere i valori accuracy di ogni metodo con cross-validation
validation_scores = {}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# %%
# Prima "vista" del performance di ogni modello, inizializzandoli senza alcun parametri,
for name, model in models.items():
    print(f"{name}'s KFold starting")
    score = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=kf, n_jobs=-1, verbose=0).mean()
    print(f"{name}'s cross validation score: {score:.6f}\n")
    validation_scores[name] = score

# %%
plt.figure(figsize=(8,4), tight_layout=True)
colors = sns.color_palette('Set3')

plt.barh(list(validation_scores.keys()), list(validation_scores.values()), color=colors[1:16])
plt.title("Cross-validation Scores")
plt.xlabel('Performance')
plt.savefig('FigXXCrossValidationScore.png', dpi=600)
plt.show()

# %%
# Lista dei modeli a utilizzare con hypertuning
model_dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_leaf_nodes=50,
                       min_samples_leaf=2, random_state=0)
model_knn = KNeighborsClassifier(algorithm='brute', n_neighbors=1, weights='distance')
model_gaussnb = GaussianNB(var_smoothing=0.0001)
model_bernb = BernoulliNB(alpha=0)
model_logreg = LogisticRegression(penalty='l1',  random_state=0, solver='liblinear')
model_linsvc = LinearSVC(C=1, dual=False, fit_intercept=False, random_state=0)
model_svc = SVC(C=5, random_state=0)
model_mlp = MLPClassifier(alpha=0.05, hidden_layer_sizes=(555,), learning_rate='adaptive',
              max_iter=10000, solver='sgd')
model_rfc = RandomForestClassifier(criterion='entropy', n_estimators=300, oob_score=True, random_state=0)
model_etc = ExtraTreesClassifier(n_estimators=300, random_state=0)
model_bag = BaggingClassifier(max_samples=0.5, n_estimators=300, random_state=0)
model_gbc = GradientBoostingClassifier(n_estimators=300, random_state=0)
model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=-1, importance_type=None, predictor='auto',
              interaction_constraints='', learning_rate=0.25, max_delta_step=0,
              max_depth=2, min_child_weight=1, monotone_constraints='()',
              n_estimators=300, n_jobs=8, num_parallel_tree=1, objective='multi:softprob',
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
              seed=0, subsample=1, tree_method='exact', validate_parameters=1,
              verbosity=None)

# Dizionario a percorrere per chiamare ogni modello
models = {
    'Gaussian Naive Bayes': model_gaussnb,
    'Bernoulli Naive Bayes': model_bernb,
    'MLP': model_mlp,
    'Random Forest': model_rfc,
    'ExtraTrees': model_etc,
    'Bagging': model_bag,
    'GradientBoost': model_gbc,
    'XGBoost': model_xgb,
    'SVC': model_svc,
    'Linear SVC': model_linsvc,
    'Logistic Regression': model_logreg,
    'Decision Tree': model_dt,
    'K Neighbors': model_knn
    }

# Lista vuota dove mettere i valori accuracy di ogni metodo con cross-validation
tuned_validation_scores = {}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# %%
# Crossvalidation con parametri tunnati
import math
for name, model in models.items():
    print(f"{name}'s KFold starting")
    score = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=kf, n_jobs=-1, verbose=0).mean()
    prev_score = round(validation_scores[name],6)
    print(f"{name}'s cross validation score before tunning: ", prev_score)
    print(f"{name}'s cross validation score after tunning: {score:.6f}")
    tuned_validation_scores[name] = score
    print("Improvement over base model: ", round(((score-prev_score)/prev_score)*100,4), "%\n")

# %%
plt.figure(figsize=(8,4), tight_layout=True)
colors = sns.color_palette('Set3')

plt.barh(list(tuned_validation_scores.keys()), list(tuned_validation_scores.values()), color=colors)
plt.title("Cross-validation Scores after tuning")
plt.xlabel('Performance')
plt.savefig('FigXX-TunedCrossValidationScore.png', dpi=600)
plt.show()

# %% [markdown]
# ### **<font color="#34eb89">RandomSearch</font>**

# %%
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2,4,6,8,10,12],
    'max_leaf_nodes': [2,4,6,8,10,20,30,40,50],
    'min_samples_leaf': [2,3,4,5],
    'random_state': [0],
    }

knn_param_grid = {
    #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
    'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21,23], #default: 5
    'weights': ['uniform', 'distance'], #default = ‘uniform’
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

gaussnb_param_grid = {
    'priors': [None],
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    }

bernb_param_grid = {
    'alpha': [0,.5,1],
#     'binarize': [],
#    'class_prior': [False, True],
#    'fit_prior': []
    }

logreg_param_grid = {
    #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
    'fit_intercept': [True, False], #default: True
    'penalty': ['l1','l2'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs
    'random_state': [0]
    }

linsvc_param_grid = {
    'C': [.25, .5, .75, 1],
#     'class_weight': [],
    'dual': [False],
    'fit_intercept': [False, True],
#     'intercept_scaling': [],
    'loss': ['hinge', 'squared_hinge'],
#     'max_iter': [],
#     'multi_class': [],
#     'penalty': [],
     'random_state': [0],
#     'tol': []
    }

svc_param_grid = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
    #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [.1,.5,1], #default=1.0
    'gamma': [.25, .5, 1.0], #edfault: auto
    'decision_function_shape': ['ovo', 'ovr'], #default:ovr
    'random_state': [0]
    }

mlp_param_grid = {
    'hidden_layer_sizes': [(368,), (555,), (100,)],
    'activation': ['identity', 'logistic', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    'max_iter': [200, 1000, 5000, 10000]
    }

rfc_param_grid = {
    #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    'n_estimators': [10, 50, 100, 300], #default=10
    'criterion': ['gini', 'entropy'], #default=”gini”
    'max_depth': [2, 4, 6, 8, 10, None], #default=None
    'oob_score': [True], #default=False
    'random_state': [0]
    }

etc_param_grid = {
    #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
    'n_estimators': [10, 50, 100, 300], #default=10
    'criterion': ['gini', 'entropy'], #default=”gini”
    'max_depth': [2, 4, 6, 8, 10, None], #default=None
    'random_state': [0]
    }

bag_param_grid = {
    #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
    'n_estimators': [10, 50, 100, 300], #default=10
    'max_samples': [.5, 1], #default=1.0
    'random_state': [0]
    }

gbc_param_grid = {
    #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
    'loss': ['deviance', 'exponential'], #default=’deviance’
    'learning_rate': [.05], #default=0.1
    'n_estimators': [300], #default=100
    'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
    'max_depth': [2, 4, 6, 8, 10, None], #default=3
    'random_state': [0]
    }

xgb_param_grid = {
    #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
    'learning_rate': [.01, .03, .05, .1, .25], #default: .3
    'max_depth': [2,4,6,8,10], #default 2
    'n_estimators': [10, 50, 100, 300],
    'seed': [0]
    }

models_params = {
#     'Decision Tree': [model_dt, dt_param_grid],
#     'K Neighbors': [model_knn, knn_param_grid],
#     'Gaussian Naive Bayes': [model_gaussnb, gaussnb_param_grid],
#     'Bernoulli Naive Bayes': [model_bernb, bernb_param_grid],
#     'Logistic Regression': [model_logreg, logreg_param_grid],
#     'Linear SVM': [model_linsvc, linsvc_param_grid],
#     'SVM': [model_svc, svc_param_grid],
#     'MLP': [model_mlp, mlp_param_grid],
#     'Random Forest': [model_rfc, rfc_param_grid],
#     'ExtraTrees': [model_etc, etc_param_grid],
#     'Bagging': [model_bag, bag_param_grid],
#     'GradientBoost': [model_gbc, gbc_param_grid],
#     'XGBoost': [model_xgb, xgb_param_grid]
    }

# %%
random_models = {}
random_validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Random search starting')
    search = RandomizedSearchCV(estimator = model,
                                param_distributions = param,
                                n_iter = 100,
                                cv = kf,
                                verbose=2,
                                random_state=42,
                                n_jobs = -1).fit(X_train, y_train)
    random_models[name] = search.best_estimator_
    random_validation_scores[name] = search.best_score_
    print(f'Best score: {search.best_score_}')
    print("Best parameters: ", random_models[name], '\n')

# %% [markdown]
# ### **<font color="#34eb89">Gridsearch</font>**

# %%
final_models = {}
final_validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Grid search starting')
    search = GridSearchCV(model,
                          param,
                          cv=kf,
                          n_jobs=-1,
                          verbose=1,
                          scoring='accuracy').fit(X_train, y_train)
    final_models[name] = search.best_estimator_
    final_validation_scores[name] = search.best_score_
    print(f'Best score: {search.best_score_}')
    print("Best parameters: ", final_models[name], '\n')

# %% [markdown]
# ### **<font color="#34eb89">ROC e PR Senza Hypertuning</font>**

# %%
# Lista dei modeli a utilizzare
gauss_nb = OneVsRestClassifier(GaussianNB())
bern_nb = OneVsRestClassifier(BernoulliNB())
mlp = OneVsRestClassifier(MLPClassifier())
rfc = OneVsRestClassifier(RandomForestClassifier())
etc = OneVsRestClassifier(ExtraTreesClassifier())
bag = OneVsRestClassifier(BaggingClassifier())
gbc = OneVsRestClassifier(GradientBoostingClassifier())
xgb = OneVsRestClassifier(XGBClassifier())
svc = OneVsRestClassifier(SVC())
lin_svc = OneVsRestClassifier(LinearSVC())
log_reg = OneVsRestClassifier(LogisticRegression())
dtc = OneVsRestClassifier(DecisionTreeClassifier())
knn = OneVsRestClassifier(KNeighborsClassifier())

# Dizionario a percorrere per chiamare ogni modello
models = {
    'Gaussian Naive Bayes': gauss_nb,
    'Bernoulli Naive Bayes': bern_nb,
    'MLP': mlp,
    'Random Forest': rfc,
    'ExtraTrees': etc,
    'Bagging': bag,
    'GradientBoost': gbc,
    'XGBoost': xgb,
    'SVC': svc,
    'LinearSVC': lin_svc,
    'Logistic Regression': log_reg,
    'Decision Tree': dtc,
    'K Nearest Neighbor': knn
    }

# %%
# FARE ATTENZIONE A NON ESEGUIRLA DUE VOLTE SE NON SI VUOLE PERDERE UN SACCO DI TEMPO
for name, model in models.items():
    model.fit(X_train, y_train_bin)
    print(f"{name} fitted")

# %%
y_score1 = gauss_nb.predict_proba(X_test)
y_score2 = bern_nb.predict_proba(X_test)
y_score3 = mlp.predict_proba(X_test)
y_score4 = rfc.predict_proba(X_test)
y_score5 = etc.predict_proba(X_test)
y_score6 = bag.predict_proba(X_test)
y_score7 = xgb.predict_proba(X_test)
y_score8 = gbc.predict_proba(X_test)
y_score9 = svc.decision_function(X_test)
y_score10 = lin_svc.decision_function(X_test)
y_score13 = log_reg.predict_proba(X_test)
y_score11 = dtc.predict_proba(X_test)
y_score12 = knn.predict_proba(X_test)

# %%
plt.figure(figsize=(10,7))
lw = 2

#################### GAUSSIAN NAIVE BAYES ####################
fpr1 = {}
tpr1 = {}
roc_auc1 = {}
for i in range(n_classes):
    fpr1[i], tpr1[i], _ = roc_curve(y_test_bin[:, i], y_score1[:, i])
    roc_auc1[i] = auc(fpr1[i], tpr1[i])

fpr1["micro"], tpr1["micro"], _ = roc_curve(y_test_bin.ravel(), y_score1.ravel())
roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])

plt.plot(
    fpr1["micro"],
    tpr1["micro"],
    label="Gaussian Naïve Bayes (AUC:{0:0.2f})".format(roc_auc1["micro"]),
    color="navy",
    linestyle=":",
    linewidth=lw,
)

#################### BERNOULLI NAIVE BAYES ####################
fpr2 = {}
tpr2 = {}
roc_auc2 = {}
for i in range(n_classes):
    fpr2[i], tpr2[i], _ = roc_curve(y_test_bin[:, i], y_score2[:, i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])

fpr2["micro"], tpr2["micro"], _ = roc_curve(y_test_bin.ravel(), y_score2.ravel())
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])

plt.plot(
    fpr2["micro"],
    tpr2["micro"],
    label="Bernoulli Naïve Bayes (AUC:{0:0.2f})".format(roc_auc2["micro"]),
    color="turquoise",
    linestyle=":",
    linewidth=lw,
)

#################### MULTILAYER PERCEPTRON ####################
fpr3 = {}
tpr3 = {}
roc_auc3 = {}
for i in range(n_classes):
    fpr3[i], tpr3[i], _ = roc_curve(y_test_bin[:, i], y_score3[:, i])
    roc_auc3[i] = auc(fpr3[i], tpr3[i])

fpr3["micro"], tpr3["micro"], _ = roc_curve(y_test_bin.ravel(), y_score3.ravel())
roc_auc3["micro"] = auc(fpr3["micro"], tpr3["micro"])

plt.plot(
    fpr3["micro"],
    tpr3["micro"],
    label="MultiLayer Perceptron (AUC:{0:0.2f})".format(roc_auc3["micro"]),
    color="darkorange",
    linestyle=":",
    linewidth=lw,
)

#################### RANDOM FOREST ####################
fpr4 = {}
tpr4 = {}
roc_auc4 = {}
for i in range(n_classes):
    fpr4[i], tpr4[i], _ = roc_curve(y_test_bin[:, i], y_score4[:, i])
    roc_auc4[i] = auc(fpr4[i], tpr4[i])

fpr4["micro"], tpr4["micro"], _ = roc_curve(y_test_bin.ravel(), y_score4.ravel())
roc_auc4["micro"] = auc(fpr4["micro"], tpr4["micro"])

plt.plot(
    fpr4["micro"],
    tpr4["micro"],
    label="Random Forest (AUC:{0:0.2f})".format(roc_auc4["micro"]),
    color="cornflowerblue",
    linestyle=":",
    linewidth=lw,
)

#################### EXTRA TREES ####################
fpr5 = {}
tpr5 = {}
roc_auc5 = {}
for i in range(n_classes):
    fpr5[i], tpr5[i], _ = roc_curve(y_test_bin[:, i], y_score5[:, i])
    roc_auc5[i] = auc(fpr5[i], tpr5[i])

fpr5["micro"], tpr5["micro"], _ = roc_curve(y_test_bin.ravel(), y_score5.ravel())
roc_auc5["micro"] = auc(fpr5["micro"], tpr5["micro"])

plt.plot(
    fpr5["micro"],
    tpr5["micro"],
    label="Extra Trees (AUC:{0:0.2f})".format(roc_auc5["micro"]),
    color="darkred",
    linestyle=":",
    linewidth=lw,
)

#################### BAGGING ####################
fpr6 = {}
tpr6 = {}
roc_auc6 = {}
for i in range(n_classes):
    fpr6[i], tpr6[i], _ = roc_curve(y_test_bin[:, i], y_score6[:, i])
    roc_auc6[i] = auc(fpr6[i], tpr6[i])

fpr6["micro"], tpr6["micro"], _ = roc_curve(y_test_bin.ravel(), y_score6.ravel())
roc_auc6["micro"] = auc(fpr6["micro"], tpr6["micro"])

plt.plot(
    fpr6["micro"],
    tpr6["micro"],
    label="Bagging (AUC:{0:0.2f})".format(roc_auc6["micro"]),
    color="purple",
    linestyle=":",
    linewidth=lw,
)

#################### XGBOOST ####################
fpr7 = {}
tpr7 = {}
roc_auc7 = {}
for i in range(n_classes):
    fpr7[i], tpr7[i], _ = roc_curve(y_test_bin[:, i], y_score7[:, i])
    roc_auc7[i] = auc(fpr7[i], tpr7[i])

fpr7["micro"], tpr7["micro"], _ = roc_curve(y_test_bin.ravel(), y_score7.ravel())
roc_auc7["micro"] = auc(fpr7["micro"], tpr7["micro"])

plt.plot(
    fpr7["micro"],
    tpr7["micro"],
    label="XGBoost (AUC:{0:0.2f})".format(roc_auc7["micro"]),
    color="olivedrab",
    linestyle=":",
    linewidth=lw,
)

#################### GRADIENT BOOSTING CLASSIFIER ####################
fpr8 = {}
tpr8 = {}
roc_auc8 = {}
for i in range(n_classes):
    fpr3[i], tpr3[i], _ = roc_curve(y_test_bin[:, i], y_score3[:, i])
    roc_auc3[i] = auc(fpr3[i], tpr3[i])

fpr3["micro"], tpr3["micro"], _ = roc_curve(y_test_bin.ravel(), y_score3.ravel())
roc_auc3["micro"] = auc(fpr3["micro"], tpr3["micro"])

plt.plot(
    fpr3["micro"],
    tpr3["micro"],
    label="GradientBoosting (AUC:{0:0.2f})".format(roc_auc3["micro"]),
    color="blue",
    linestyle=":",
    linewidth=lw,
)

#################### SVC ####################
fpr9 = {}
tpr9 = {}
roc_auc9 = {}
for i in range(n_classes):
    fpr9[i], tpr9[i], _ = roc_curve(y_test_bin[:, i], y_score9[:, i])
    roc_auc9[i] = auc(fpr9[i], tpr9[i])

fpr9["micro"], tpr9["micro"], _ = roc_curve(y_test_bin.ravel(), y_score9.ravel())
roc_auc9["micro"] = auc(fpr9["micro"], tpr9["micro"])

plt.plot(
    fpr9["micro"],
    tpr9["micro"],
    label="SVC (AUC:{0:0.2f})".format(roc_auc9["micro"]),
    color="lime",
    linestyle=":",
    linewidth=lw,
)

#################### LINEAR SVC ####################
fpr10 = {}
tpr10 = {}
roc_auc10 = {}
for i in range(n_classes):
    fpr10[i], tpr10[i], _ = roc_curve(y_test_bin[:, i], y_score10[:, i])
    roc_auc10[i] = auc(fpr10[i], tpr10[i])

fpr10["micro"], tpr10["micro"], _ = roc_curve(y_test_bin.ravel(), y_score10.ravel())
roc_auc10["micro"] = auc(fpr10["micro"], tpr10["micro"])

plt.plot(
    fpr10["micro"],
    tpr10["micro"],
    label="Linear SVC (AUC:{0:0.2f})".format(roc_auc10["micro"]),
    color="yellow",
    linestyle=":",
    linewidth=lw,
)

#################### LOGISTIC REGRESSION ####################
fpr13 = {}
tpr13 = {}
roc_auc13 = {}
for i in range(n_classes):
    fpr13[i], tpr13[i], _ = roc_curve(y_test_bin[:, i], y_score13[:, i])
    roc_auc13[i] = auc(fpr13[i], tpr13[i])

fpr13["micro"], tpr13["micro"], _ = roc_curve(y_test_bin.ravel(), y_score13.ravel())
roc_auc13["micro"] = auc(fpr13["micro"], tpr13["micro"])

plt.plot(
    fpr13["micro"],
    tpr13["micro"],
    label="Logistic Regression (AUC:{0:0.2f})".format(roc_auc13["micro"]),
    color="darkgray",
    linestyle=":",
    linewidth=lw,
)

#################### DECISION TREE ####################
fpr11 = {}
tpr11 = {}
roc_auc11 = {}
for i in range(n_classes):
    fpr11[i], tpr11[i], _ = roc_curve(y_test_bin[:, i], y_score11[:, i])
    roc_auc11[i] = auc(fpr11[i], tpr11[i])

fpr11["micro"], tpr11["micro"], _ = roc_curve(y_test_bin.ravel(), y_score11.ravel())
roc_auc11["micro"] = auc(fpr11["micro"], tpr11["micro"])

plt.plot(
    fpr11["micro"],
    tpr11["micro"],
    label="Decision Tree (AUC:{0:0.2f})".format(roc_auc11["micro"]),
    color="coral",
    linestyle=":",
    linewidth=lw,
)

#################### K NEAREST NEIGHBOR ####################
fpr12 = {}
tpr12 = {}
roc_auc12 = {}
for i in range(n_classes):
    fpr12[i], tpr12[i], _ = roc_curve(y_test_bin[:, i], y_score12[:, i])
    roc_auc12[i] = auc(fpr12[i], tpr12[i])

fpr12["micro"], tpr12["micro"], _ = roc_curve(y_test_bin.ravel(), y_score12.ravel())
roc_auc12["micro"] = auc(fpr12["micro"], tpr12["micro"])

plt.plot(
    fpr12["micro"],
    tpr12["micro"],
    label="K Nearest Neighbor (AUC:{0:0.2f})".format(roc_auc12["micro"]),
    color="pink",
    linestyle=":",
    linewidth=lw,
)

###############################################################

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.savefig('FigXX-ROCcomparison.png', dpi=600)
plt.show()

# %%
_, ax = plt.subplots(figsize=(11, 10))

#################### GAUSSIAN NAIVE BAYES ####################
precision1 = dict()
recall1 = dict()
average_precision1 = dict()
for i in range(n_classes):
    precision1[i], recall1[i], _ = precision_recall_curve(y_test_bin[:, i], y_score1[:, i])
    average_precision1[i] = average_precision_score(y_test_bin[:, i], y_score1[:, i])

precision1["micro"], recall1["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score1.ravel())
average_precision1["micro"] = average_precision_score(y_test_bin, y_score1, average="micro")

display = PrecisionRecallDisplay(
    recall=recall1["micro"],
    precision=precision1["micro"],
    average_precision=average_precision1["micro"],
)
display.plot(ax=ax, name="Gaussian Naïve Bayes", color="navy")

#################### BERNOULLI NAIVE BAYES ####################
precision2 = dict()
recall2 = dict()
average_precision2 = dict()
for i in range(n_classes):
    precision2[i], recall2[i], _ = precision_recall_curve(y_test_bin[:, i], y_score2[:, i])
    average_precision2[i] = average_precision_score(y_test_bin[:, i], y_score2[:, i])

precision2["micro"], recall2["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score2.ravel())
average_precision2["micro"] = average_precision_score(y_test_bin, y_score2, average="micro")

display = PrecisionRecallDisplay(
        recall=recall2["micro"],
        precision=precision2["micro"],
        average_precision=average_precision2["micro"],
    )
display.plot(ax=ax, name=f"Bernoulli Naïve Bayes", color="turquoise")

#################### MULTI LAYER PERCEPTRON ####################
precision3 = dict()
recall3 = dict()
average_precision3 = dict()
for i in range(n_classes):
    precision3[i], recall3[i], _ = precision_recall_curve(y_test_bin[:, i], y_score3[:, i])
    average_precision3[i] = average_precision_score(y_test_bin[:, i], y_score3[:, i])

precision3["micro"], recall3["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score3.ravel())
average_precision3["micro"] = average_precision_score(y_test_bin, y_score3, average="micro")

display = PrecisionRecallDisplay(
        recall=recall3["micro"],
        precision=precision3["micro"],
        average_precision=average_precision3["micro"],
    )
display.plot(ax=ax, name=f"MultiLayer Perceptron", color='darkorange')

#################### RANDOM FOREST ####################
precision4 = dict()
recall4 = dict()
average_precision4 = dict()
for i in range(n_classes):
    precision4[i], recall4[i], _ = precision_recall_curve(y_test_bin[:, i], y_score4[:, i])
    average_precision4[i] = average_precision_score(y_test_bin[:, i], y_score4[:, i])

precision4["micro"], recall4["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score4.ravel())
average_precision4["micro"] = average_precision_score(y_test_bin, y_score4, average="micro")

display = PrecisionRecallDisplay(
        recall=recall4["micro"],
        precision=precision4["micro"],
        average_precision=average_precision4["micro"],
    )
display.plot(ax=ax, name=f"Random Forest", color='cornflowerblue')

#################### EXTRA TREES ####################
precision5 = dict()
recall5 = dict()
average_precision5 = dict()
for i in range(n_classes):
    precision5[i], recall5[i], _ = precision_recall_curve(y_test_bin[:, i], y_score5[:, i])
    average_precision5[i] = average_precision_score(y_test_bin[:, i], y_score5[:, i])

precision5["micro"], recall5["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score5.ravel())
average_precision5["micro"] = average_precision_score(y_test_bin, y_score5, average="micro")

display = PrecisionRecallDisplay(
        recall=recall5["micro"],
        precision=precision5["micro"],
        average_precision=average_precision5["micro"],
    )
display.plot(ax=ax, name=f"Extra Trees", color='darkred')

#################### BAGGING ####################

precision6 = dict()
recall6 = dict()
average_precision6 = dict()
for i in range(n_classes):
    precision6[i], recall6[i], _ = precision_recall_curve(y_test_bin[:, i], y_score6[:, i])
    average_precision6[i] = average_precision_score(y_test_bin[:, i], y_score6[:, i])

precision6["micro"], recall6["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score6.ravel())
average_precision6["micro"] = average_precision_score(y_test_bin, y_score6, average="micro")

display = PrecisionRecallDisplay(
        recall=recall6["micro"],
        precision=precision6["micro"],
        average_precision=average_precision6["micro"],
    )
display.plot(ax=ax, name=f"Bagging", color='purple')

#################### XGBOOST ####################
precision7 = dict()
recall7 = dict()
average_precision7 = dict()
for i in range(n_classes):
    precision7[i], recall7[i], _ = precision_recall_curve(y_test_bin[:, i], y_score7[:, i])
    average_precision7[i] = average_precision_score(y_test_bin[:, i], y_score7[:, i])

precision7["micro"], recall7["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score7.ravel())
average_precision7["micro"] = average_precision_score(y_test_bin, y_score7, average="micro")

display = PrecisionRecallDisplay(
        recall=recall7["micro"],
        precision=precision7["micro"],
        average_precision=average_precision7["micro"],
    )
display.plot(ax=ax, name=f"XGBoost", color='olivedrab')

#################### GRADIENT BOOSTING ####################
precision8 = dict()
recall8 = dict()
average_precision8 = dict()
for i in range(n_classes):
    precision8[i], recall8[i], _ = precision_recall_curve(y_test_bin[:, i], y_score8[:, i])
    average_precision8[i] = average_precision_score(y_test_bin[:, i], y_score8[:, i])

precision8["micro"], recall8["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score8.ravel())
average_precision8["micro"] = average_precision_score(y_test_bin, y_score8, average="micro")

display = PrecisionRecallDisplay(
        recall=recall8["micro"],
        precision=precision8["micro"],
        average_precision=average_precision8["micro"],
    )
display.plot(ax=ax, name=f"Gradient Boosting", color='blue')

#################### SVC ####################
precision9 = dict()
recall9 = dict()
average_precision9 = dict()
for i in range(n_classes):
    precision9[i], recall9[i], _ = precision_recall_curve(y_test_bin[:, i], y_score9[:, i])
    average_precision9[i] = average_precision_score(y_test_bin[:, i], y_score9[:, i])

precision9["micro"], recall9["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score9.ravel())
average_precision9["micro"] = average_precision_score(y_test_bin, y_score9, average="micro")

display = PrecisionRecallDisplay(
        recall=recall9["micro"],
        precision=precision9["micro"],
        average_precision=average_precision9["micro"],
    )
display.plot(ax=ax, name=f"SVC", color='lime')

#################### LINEAR SVC ####################
precision10 = dict()
recall10 = dict()
average_precision10 = dict()
for i in range(n_classes):
    precision10[i], recall10[i], _ = precision_recall_curve(y_test_bin[:, i], y_score10[:, i])
    average_precision10[i] = average_precision_score(y_test_bin[:, i], y_score10[:, i])

precision10["micro"], recall10["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score10.ravel())
average_precision10["micro"] = average_precision_score(y_test_bin, y_score10, average="micro")

display = PrecisionRecallDisplay(
        recall=recall10["micro"],
        precision=precision10["micro"],
        average_precision=average_precision10["micro"],
    )
display.plot(ax=ax, name=f"Linear SVC", color='yellow')

#################### LOGISTIC REGRESSION ####################
precision13 = dict()
recall13= dict()
average_precision13 = dict()
for i in range(n_classes):
    precision13[i], recall13[i], _ = precision_recall_curve(y_test_bin[:, i], y_score13[:, i])
    average_precision13[i] = average_precision_score(y_test_bin[:, i], y_score13[:, i])

precision13["micro"], recall13["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score13.ravel())
average_precision13["micro"] = average_precision_score(y_test_bin, y_score13, average="micro")

display = PrecisionRecallDisplay(
        recall=recall13["micro"],
        precision=precision13["micro"],
        average_precision=average_precision13["micro"],
    )
display.plot(ax=ax, name=f"Logistic Regression", color='darkgray')

#################### DECISION TREE ####################
precision11 = dict()
recall11 = dict()
average_precision11 = dict()
for i in range(n_classes):
    precision11[i], recall11[i], _ = precision_recall_curve(y_test_bin[:, i], y_score11[:, i])
    average_precision11[i] = average_precision_score(y_test_bin[:, i], y_score11[:, i])

precision11["micro"], recall11["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score11.ravel())
average_precision11["micro"] = average_precision_score(y_test_bin, y_score11, average="micro")

display = PrecisionRecallDisplay(
        recall=recall11["micro"],
        precision=precision11["micro"],
        average_precision=average_precision11["micro"],
    )
display.plot(ax=ax, name=f"Decision Tree", color='coral')

#################### K NEAREST NEIGHBOR ####################
precision12 = dict()
recall12 = dict()
average_precision12 = dict()
for i in range(n_classes):
    precision12[i], recall12[i], _ = precision_recall_curve(y_test_bin[:, i], y_score12[:, i])
    average_precision12[i] = average_precision_score(y_test_bin[:, i], y_score12[:, i])

precision12["micro"], recall12["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score12.ravel())
average_precision12["micro"] = average_precision_score(y_test_bin, y_score12, average="micro")

display = PrecisionRecallDisplay(
        recall=recall12["micro"],
        precision=precision12["micro"],
        average_precision=average_precision12["micro"],
    )
display.plot(ax=ax, name=f"K Nearest Neighbor", color='pink')

###############################################################
handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([0.0, 1.01])
ax.set_ylim([0.0, 1.05])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Precision-Recall Curve Comparison")
plt.savefig('FigXX-PRcomparison.png', dpi=600)
plt.show()

# %% [markdown]
# ### **<font color="#34eb89">ROC e PR Con Hypertuning</font>**

# %%
# Lista dei modeli a utilizzare
gauss_nb = OneVsRestClassifier(GaussianNB(var_smoothing=0.0001))
bern_nb = OneVsRestClassifier(BernoulliNB(alpha=0))
mlp = OneVsRestClassifier(MLPClassifier(alpha=0.05, hidden_layer_sizes=(555,), learning_rate='adaptive',
              max_iter=10000, solver='sgd'))
rfc = OneVsRestClassifier(RandomForestClassifier(criterion='entropy', n_estimators=300, oob_score=True, random_state=0))
etc = OneVsRestClassifier(ExtraTreesClassifier(n_estimators=300, random_state=0))
bag = OneVsRestClassifier(BaggingClassifier(max_samples=0.5, n_estimators=300, random_state=0))
gbc = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=300, random_state=0))
xgb = OneVsRestClassifier(XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=-1, importance_type=None, predictor='auto',
              interaction_constraints='', learning_rate=0.25, max_delta_step=0,
              max_depth=2, min_child_weight=1, monotone_constraints='()',
              n_estimators=300, n_jobs=8, num_parallel_tree=1, objective='multi:softprob',
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
              seed=0, subsample=1, tree_method='exact', validate_parameters=1,
              verbosity=None))
svc = OneVsRestClassifier(SVC(C=5, random_state=0))
lin_svc = OneVsRestClassifier(LinearSVC(C=1, dual=False, fit_intercept=False, random_state=0))
log_reg = OneVsRestClassifier(LogisticRegression(penalty='l1',  random_state=0, solver='liblinear'))
dtc = OneVsRestClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=10, max_leaf_nodes=50,
                       min_samples_leaf=2, random_state=0))
knn = OneVsRestClassifier(KNeighborsClassifier(algorithm='brute', n_neighbors=1, weights='distance'))

# Dizionario a percorrere per chiamare ogni modello
models = {
    'Gaussian Naive Bayes': gauss_nb,
    'Bernoulli Naive Bayes': bern_nb,
    'MLP': mlp,
    'Random Forest': rfc,
    'ExtraTrees': etc,
    'Bagging': bag,
    'GradientBoost': gbc,
    'XGBoost': xgb,
    'SVC': svc,
    'LinearSVC': lin_svc,
    'Logistic Regression': log_reg,
    'Decision Tree': dtc,
    'K Nearest Neighbor': knn
    }

# %%
# FARE ATTENZIONE A NON ESEGUIRLA DUE VOLTE SE NON SI VUOLE PERDERE UN SACCO DI TEMPO
for name, model in models.items():
    model.fit(X_train, y_train_bin)
    print(f"{name} fitted")

# %%
y_score1 = gauss_nb.predict_proba(X_test)
y_score2 = bern_nb.predict_proba(X_test)
y_score3 = mlp.predict_proba(X_test)
y_score4 = rfc.predict_proba(X_test)
y_score5 = etc.predict_proba(X_test)
y_score6 = bag.predict_proba(X_test)
y_score7 = xgb.predict_proba(X_test)
y_score8 = gbc.predict_proba(X_test)
y_score9 = svc.decision_function(X_test)
y_score10 = lin_svc.decision_function(X_test)
y_score13 = log_reg.predict_proba(X_test)
y_score11 = dtc.predict_proba(X_test)
y_score12 = knn.predict_proba(X_test)

# %%
plt.figure(figsize=(10,7))
lw = 2

#################### GAUSSIAN NAIVE BAYES ####################
fpr1 = {}
tpr1 = {}
roc_auc1 = {}
for i in range(n_classes):
    fpr1[i], tpr1[i], _ = roc_curve(y_test_bin[:, i], y_score1[:, i])
    roc_auc1[i] = auc(fpr1[i], tpr1[i])

fpr1["micro"], tpr1["micro"], _ = roc_curve(y_test_bin.ravel(), y_score1.ravel())
roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])

plt.plot(
    fpr1["micro"],
    tpr1["micro"],
    label="Gaussian Naïve Bayes (AUC:{0:0.2f})".format(roc_auc1["micro"]),
    color="navy",
    linestyle=":",
    linewidth=lw,
)

#################### BERNOULLI NAIVE BAYES ####################
fpr2 = {}
tpr2 = {}
roc_auc2 = {}
for i in range(n_classes):
    fpr2[i], tpr2[i], _ = roc_curve(y_test_bin[:, i], y_score2[:, i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])

fpr2["micro"], tpr2["micro"], _ = roc_curve(y_test_bin.ravel(), y_score2.ravel())
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])

plt.plot(
    fpr2["micro"],
    tpr2["micro"],
    label="Bernoulli Naïve Bayes (AUC:{0:0.2f})".format(roc_auc2["micro"]),
    color="turquoise",
    linestyle=":",
    linewidth=lw,
)

#################### MULTILAYER PERCEPTRON ####################
fpr3 = {}
tpr3 = {}
roc_auc3 = {}
for i in range(n_classes):
    fpr3[i], tpr3[i], _ = roc_curve(y_test_bin[:, i], y_score3[:, i])
    roc_auc3[i] = auc(fpr3[i], tpr3[i])

fpr3["micro"], tpr3["micro"], _ = roc_curve(y_test_bin.ravel(), y_score3.ravel())
roc_auc3["micro"] = auc(fpr3["micro"], tpr3["micro"])

plt.plot(
    fpr3["micro"],
    tpr3["micro"],
    label="MultiLayer Perceptron (AUC:{0:0.2f})".format(roc_auc3["micro"]),
    color="darkorange",
    linestyle=":",
    linewidth=lw,
)

#################### RANDOM FOREST ####################
fpr4 = {}
tpr4 = {}
roc_auc4 = {}
for i in range(n_classes):
    fpr4[i], tpr4[i], _ = roc_curve(y_test_bin[:, i], y_score4[:, i])
    roc_auc4[i] = auc(fpr4[i], tpr4[i])

fpr4["micro"], tpr4["micro"], _ = roc_curve(y_test_bin.ravel(), y_score4.ravel())
roc_auc4["micro"] = auc(fpr4["micro"], tpr4["micro"])

plt.plot(
    fpr4["micro"],
    tpr4["micro"],
    label="Random Forest (AUC:{0:0.2f})".format(roc_auc4["micro"]),
    color="cornflowerblue",
    linestyle=":",
    linewidth=lw,
)

#################### EXTRA TREES ####################
fpr5 = {}
tpr5 = {}
roc_auc5 = {}
for i in range(n_classes):
    fpr5[i], tpr5[i], _ = roc_curve(y_test_bin[:, i], y_score5[:, i])
    roc_auc5[i] = auc(fpr5[i], tpr5[i])

fpr5["micro"], tpr5["micro"], _ = roc_curve(y_test_bin.ravel(), y_score5.ravel())
roc_auc5["micro"] = auc(fpr5["micro"], tpr5["micro"])

plt.plot(
    fpr5["micro"],
    tpr5["micro"],
    label="Extra Trees (AUC:{0:0.2f})".format(roc_auc5["micro"]),
    color="darkred",
    linestyle=":",
    linewidth=lw,
)

#################### BAGGING ####################
fpr6 = {}
tpr6 = {}
roc_auc6 = {}
for i in range(n_classes):
    fpr6[i], tpr6[i], _ = roc_curve(y_test_bin[:, i], y_score6[:, i])
    roc_auc6[i] = auc(fpr6[i], tpr6[i])

fpr6["micro"], tpr6["micro"], _ = roc_curve(y_test_bin.ravel(), y_score6.ravel())
roc_auc6["micro"] = auc(fpr6["micro"], tpr6["micro"])

plt.plot(
    fpr6["micro"],
    tpr6["micro"],
    label="Bagging (AUC:{0:0.2f})".format(roc_auc6["micro"]),
    color="purple",
    linestyle=":",
    linewidth=lw,
)

#################### XGBOOST ####################
fpr7 = {}
tpr7 = {}
roc_auc7 = {}
for i in range(n_classes):
    fpr7[i], tpr7[i], _ = roc_curve(y_test_bin[:, i], y_score7[:, i])
    roc_auc7[i] = auc(fpr7[i], tpr7[i])

fpr7["micro"], tpr7["micro"], _ = roc_curve(y_test_bin.ravel(), y_score7.ravel())
roc_auc7["micro"] = auc(fpr7["micro"], tpr7["micro"])

plt.plot(
    fpr7["micro"],
    tpr7["micro"],
    label="XGBoost (AUC:{0:0.2f})".format(roc_auc7["micro"]),
    color="olivedrab",
    linestyle=":",
    linewidth=lw,
)

#################### GRADIENT BOOSTING CLASSIFIER ####################
fpr8 = {}
tpr8 = {}
roc_auc8 = {}
for i in range(n_classes):
    fpr3[i], tpr3[i], _ = roc_curve(y_test_bin[:, i], y_score3[:, i])
    roc_auc3[i] = auc(fpr3[i], tpr3[i])

fpr3["micro"], tpr3["micro"], _ = roc_curve(y_test_bin.ravel(), y_score3.ravel())
roc_auc3["micro"] = auc(fpr3["micro"], tpr3["micro"])

plt.plot(
    fpr3["micro"],
    tpr3["micro"],
    label="GradientBoosting (AUC:{0:0.2f})".format(roc_auc3["micro"]),
    color="blue",
    linestyle=":",
    linewidth=lw,
)

#################### SVC ####################
fpr9 = {}
tpr9 = {}
roc_auc9 = {}
for i in range(n_classes):
    fpr9[i], tpr9[i], _ = roc_curve(y_test_bin[:, i], y_score9[:, i])
    roc_auc9[i] = auc(fpr9[i], tpr9[i])

fpr9["micro"], tpr9["micro"], _ = roc_curve(y_test_bin.ravel(), y_score9.ravel())
roc_auc9["micro"] = auc(fpr9["micro"], tpr9["micro"])

plt.plot(
    fpr9["micro"],
    tpr9["micro"],
    label="SVC (AUC:{0:0.2f})".format(roc_auc9["micro"]),
    color="lime",
    linestyle=":",
    linewidth=lw,
)

#################### LINEAR SVC ####################
fpr10 = {}
tpr10 = {}
roc_auc10 = {}
for i in range(n_classes):
    fpr10[i], tpr10[i], _ = roc_curve(y_test_bin[:, i], y_score10[:, i])
    roc_auc10[i] = auc(fpr10[i], tpr10[i])

fpr10["micro"], tpr10["micro"], _ = roc_curve(y_test_bin.ravel(), y_score10.ravel())
roc_auc10["micro"] = auc(fpr10["micro"], tpr10["micro"])

plt.plot(
    fpr10["micro"],
    tpr10["micro"],
    label="Linear SVC (AUC:{0:0.2f})".format(roc_auc10["micro"]),
    color="yellow",
    linestyle=":",
    linewidth=lw,
)

#################### LOGISTIC REGRESSION ####################
fpr13 = {}
tpr13 = {}
roc_auc13 = {}
for i in range(n_classes):
    fpr13[i], tpr13[i], _ = roc_curve(y_test_bin[:, i], y_score13[:, i])
    roc_auc13[i] = auc(fpr13[i], tpr13[i])

fpr13["micro"], tpr13["micro"], _ = roc_curve(y_test_bin.ravel(), y_score13.ravel())
roc_auc13["micro"] = auc(fpr13["micro"], tpr13["micro"])

plt.plot(
    fpr13["micro"],
    tpr13["micro"],
    label="Logistic Regression (AUC:{0:0.2f})".format(roc_auc13["micro"]),
    color="darkgray",
    linestyle=":",
    linewidth=lw,
)

#################### DECISION TREE ####################
fpr11 = {}
tpr11 = {}
roc_auc11 = {}
for i in range(n_classes):
    fpr11[i], tpr11[i], _ = roc_curve(y_test_bin[:, i], y_score11[:, i])
    roc_auc11[i] = auc(fpr11[i], tpr11[i])

fpr11["micro"], tpr11["micro"], _ = roc_curve(y_test_bin.ravel(), y_score11.ravel())
roc_auc11["micro"] = auc(fpr11["micro"], tpr11["micro"])

plt.plot(
    fpr11["micro"],
    tpr11["micro"],
    label="Decision Tree (AUC:{0:0.2f})".format(roc_auc11["micro"]),
    color="coral",
    linestyle=":",
    linewidth=lw,
)

#################### K NEAREST NEIGHBOR ####################
fpr12 = {}
tpr12 = {}
roc_auc12 = {}
for i in range(n_classes):
    fpr12[i], tpr12[i], _ = roc_curve(y_test_bin[:, i], y_score12[:, i])
    roc_auc12[i] = auc(fpr12[i], tpr12[i])

fpr12["micro"], tpr12["micro"], _ = roc_curve(y_test_bin.ravel(), y_score12.ravel())
roc_auc12["micro"] = auc(fpr12["micro"], tpr12["micro"])

plt.plot(
    fpr12["micro"],
    tpr12["micro"],
    label="K Nearest Neighbor (AUC:{0:0.2f})".format(roc_auc12["micro"]),
    color="pink",
    linestyle=":",
    linewidth=lw,
)

###############################################################

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.savefig('FigXX-ROCcomparison.png', dpi=600)
plt.show()

# %%
_, ax = plt.subplots(figsize=(11, 10))

#################### GAUSSIAN NAIVE BAYES ####################
precision1 = dict()
recall1 = dict()
average_precision1 = dict()
for i in range(n_classes):
    precision1[i], recall1[i], _ = precision_recall_curve(y_test_bin[:, i], y_score1[:, i])
    average_precision1[i] = average_precision_score(y_test_bin[:, i], y_score1[:, i])

precision1["micro"], recall1["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score1.ravel())
average_precision1["micro"] = average_precision_score(y_test_bin, y_score1, average="micro")

display = PrecisionRecallDisplay(
    recall=recall1["micro"],
    precision=precision1["micro"],
    average_precision=average_precision1["micro"],
)
display.plot(ax=ax, name="Gaussian Naïve Bayes", color="navy")

#################### BERNOULLI NAIVE BAYES ####################
precision2 = dict()
recall2 = dict()
average_precision2 = dict()
for i in range(n_classes):
    precision2[i], recall2[i], _ = precision_recall_curve(y_test_bin[:, i], y_score2[:, i])
    average_precision2[i] = average_precision_score(y_test_bin[:, i], y_score2[:, i])

precision2["micro"], recall2["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score2.ravel())
average_precision2["micro"] = average_precision_score(y_test_bin, y_score2, average="micro")

display = PrecisionRecallDisplay(
        recall=recall2["micro"],
        precision=precision2["micro"],
        average_precision=average_precision2["micro"],
    )
display.plot(ax=ax, name=f"Bernoulli Naïve Bayes", color="turquoise")

#################### MULTI LAYER PERCEPTRON ####################
precision3 = dict()
recall3 = dict()
average_precision3 = dict()
for i in range(n_classes):
    precision3[i], recall3[i], _ = precision_recall_curve(y_test_bin[:, i], y_score3[:, i])
    average_precision3[i] = average_precision_score(y_test_bin[:, i], y_score3[:, i])

precision3["micro"], recall3["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score3.ravel())
average_precision3["micro"] = average_precision_score(y_test_bin, y_score3, average="micro")

display = PrecisionRecallDisplay(
        recall=recall3["micro"],
        precision=precision3["micro"],
        average_precision=average_precision3["micro"],
    )
display.plot(ax=ax, name=f"MultiLayer Perceptron", color='darkorange')

#################### RANDOM FOREST ####################
precision4 = dict()
recall4 = dict()
average_precision4 = dict()
for i in range(n_classes):
    precision4[i], recall4[i], _ = precision_recall_curve(y_test_bin[:, i], y_score4[:, i])
    average_precision4[i] = average_precision_score(y_test_bin[:, i], y_score4[:, i])

precision4["micro"], recall4["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score4.ravel())
average_precision4["micro"] = average_precision_score(y_test_bin, y_score4, average="micro")

display = PrecisionRecallDisplay(
        recall=recall4["micro"],
        precision=precision4["micro"],
        average_precision=average_precision4["micro"],
    )
display.plot(ax=ax, name=f"Random Forest", color='cornflowerblue')

#################### EXTRA TREES ####################
precision5 = dict()
recall5 = dict()
average_precision5 = dict()
for i in range(n_classes):
    precision5[i], recall5[i], _ = precision_recall_curve(y_test_bin[:, i], y_score5[:, i])
    average_precision5[i] = average_precision_score(y_test_bin[:, i], y_score5[:, i])

precision5["micro"], recall5["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score5.ravel())
average_precision5["micro"] = average_precision_score(y_test_bin, y_score5, average="micro")

display = PrecisionRecallDisplay(
        recall=recall5["micro"],
        precision=precision5["micro"],
        average_precision=average_precision5["micro"],
    )
display.plot(ax=ax, name=f"Extra Trees", color='darkred')

#################### BAGGING ####################

precision6 = dict()
recall6 = dict()
average_precision6 = dict()
for i in range(n_classes):
    precision6[i], recall6[i], _ = precision_recall_curve(y_test_bin[:, i], y_score6[:, i])
    average_precision6[i] = average_precision_score(y_test_bin[:, i], y_score6[:, i])

precision6["micro"], recall6["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score6.ravel())
average_precision6["micro"] = average_precision_score(y_test_bin, y_score6, average="micro")

display = PrecisionRecallDisplay(
        recall=recall6["micro"],
        precision=precision6["micro"],
        average_precision=average_precision6["micro"],
    )
display.plot(ax=ax, name=f"Bagging", color='purple')

#################### XGBOOST ####################
precision7 = dict()
recall7 = dict()
average_precision7 = dict()
for i in range(n_classes):
    precision7[i], recall7[i], _ = precision_recall_curve(y_test_bin[:, i], y_score7[:, i])
    average_precision7[i] = average_precision_score(y_test_bin[:, i], y_score7[:, i])

precision7["micro"], recall7["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score7.ravel())
average_precision7["micro"] = average_precision_score(y_test_bin, y_score7, average="micro")

display = PrecisionRecallDisplay(
        recall=recall7["micro"],
        precision=precision7["micro"],
        average_precision=average_precision7["micro"],
    )
display.plot(ax=ax, name=f"XGBoost", color='olivedrab')

#################### GRADIENT BOOSTING ####################
precision8 = dict()
recall8 = dict()
average_precision8 = dict()
for i in range(n_classes):
    precision8[i], recall8[i], _ = precision_recall_curve(y_test_bin[:, i], y_score8[:, i])
    average_precision8[i] = average_precision_score(y_test_bin[:, i], y_score8[:, i])

precision8["micro"], recall8["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score8.ravel())
average_precision8["micro"] = average_precision_score(y_test_bin, y_score8, average="micro")

display = PrecisionRecallDisplay(
        recall=recall8["micro"],
        precision=precision8["micro"],
        average_precision=average_precision8["micro"],
    )
display.plot(ax=ax, name=f"Gradient Boosting", color='blue')

#################### SVC ####################
precision9 = dict()
recall9 = dict()
average_precision9 = dict()
for i in range(n_classes):
    precision9[i], recall9[i], _ = precision_recall_curve(y_test_bin[:, i], y_score9[:, i])
    average_precision9[i] = average_precision_score(y_test_bin[:, i], y_score9[:, i])

precision9["micro"], recall9["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score9.ravel())
average_precision9["micro"] = average_precision_score(y_test_bin, y_score9, average="micro")

display = PrecisionRecallDisplay(
        recall=recall9["micro"],
        precision=precision9["micro"],
        average_precision=average_precision9["micro"],
    )
display.plot(ax=ax, name=f"SVC", color='lime')

#################### LINEAR SVC ####################
precision10 = dict()
recall10 = dict()
average_precision10 = dict()
for i in range(n_classes):
    precision10[i], recall10[i], _ = precision_recall_curve(y_test_bin[:, i], y_score10[:, i])
    average_precision10[i] = average_precision_score(y_test_bin[:, i], y_score10[:, i])

precision10["micro"], recall10["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score10.ravel())
average_precision10["micro"] = average_precision_score(y_test_bin, y_score10, average="micro")

display = PrecisionRecallDisplay(
        recall=recall10["micro"],
        precision=precision10["micro"],
        average_precision=average_precision10["micro"],
    )
display.plot(ax=ax, name=f"Linear SVC", color='yellow')

#################### LOGISTIC REGRESSION ####################
precision13 = dict()
recall13= dict()
average_precision13 = dict()
for i in range(n_classes):
    precision13[i], recall13[i], _ = precision_recall_curve(y_test_bin[:, i], y_score13[:, i])
    average_precision13[i] = average_precision_score(y_test_bin[:, i], y_score13[:, i])

precision13["micro"], recall13["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score13.ravel())
average_precision13["micro"] = average_precision_score(y_test_bin, y_score13, average="micro")

display = PrecisionRecallDisplay(
        recall=recall13["micro"],
        precision=precision13["micro"],
        average_precision=average_precision13["micro"],
    )
display.plot(ax=ax, name=f"Logistic Regression", color='darkgray')

#################### DECISION TREE ####################
precision11 = dict()
recall11 = dict()
average_precision11 = dict()
for i in range(n_classes):
    precision11[i], recall11[i], _ = precision_recall_curve(y_test_bin[:, i], y_score11[:, i])
    average_precision11[i] = average_precision_score(y_test_bin[:, i], y_score11[:, i])

precision11["micro"], recall11["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score11.ravel())
average_precision11["micro"] = average_precision_score(y_test_bin, y_score11, average="micro")

display = PrecisionRecallDisplay(
        recall=recall11["micro"],
        precision=precision11["micro"],
        average_precision=average_precision11["micro"],
    )
display.plot(ax=ax, name=f"Decision Tree", color='coral')

#################### K NEAREST NEIGHBOR ####################
precision12 = dict()
recall12 = dict()
average_precision12 = dict()
for i in range(n_classes):
    precision12[i], recall12[i], _ = precision_recall_curve(y_test_bin[:, i], y_score12[:, i])
    average_precision12[i] = average_precision_score(y_test_bin[:, i], y_score12[:, i])

precision12["micro"], recall12["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score12.ravel())
average_precision12["micro"] = average_precision_score(y_test_bin, y_score12, average="micro")

display = PrecisionRecallDisplay(
        recall=recall12["micro"],
        precision=precision12["micro"],
        average_precision=average_precision12["micro"],
    )
display.plot(ax=ax, name=f"K Nearest Neighbor", color='pink')

###############################################################
handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([0.0, 1.01])
ax.set_ylim([0.0, 1.05])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Precision-Recall Curve Comparison")
plt.savefig('FigXX-PRcomparison.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">6.1 Naive Bayes</font>**

# %% [markdown]
# ### **<font color="#34eb89">6.1.1 Gaussian Naive Bayes</font>**

# %%
#Improve classification
gauss_nb = GaussianNB()
gauss_nb.fit(X_train_pca_mod2, y_train)
y_pred = gauss_nb.predict(X_test_pca_mod2)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %%
gauss_nb = GaussianNB()
gauss_nb.fit(X_train, y_train)
y_pred = gauss_nb.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'Gaussian Naïve Bayes'
cm_gauss_nb = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_gauss_nb, model)

# %%
gauss_nb = GaussianNB(var_smoothing=.0001)
gauss_nb.fit(X_train, y_train)
y_pred = gauss_nb.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'Gaussian Naïve Bayes'
cm_gauss_nb = confusion_matrix(y_test,y_pred)
draw_confusion_matrix_tuned(cm_gauss_nb, model)

# %%
# Inizializziamo il classifier
gauss_nb = OneVsRestClassifier(GaussianNB())
y_score = gauss_nb.fit(X_train, y_train_bin).predict_proba(X_test)

# Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcoliamo microaverage
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Mettiamo tutti i FPR insieme
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

# Macro
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="MacroAvg (AUC:{0:0.2f})".format(roc_auc["macro"]),
    color="aqua",
    linestyle=":",
    linewidth=3,
)

# Curve per ogni classe
colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1,
        label="Class {0} (AUC:{1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Gaussian Naïve Bayes ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Gaussian Naïve Bayes ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# MicroAvg calcola score di tutte le classi
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

_, ax = plt.subplots(figsize=(7, 8))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="MicroAvg Precision-Recall", color="red")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-Recall for class {i}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([-.02, 1.02])
ax.set_ylim([0.0, 1.02])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Gaussian Naïve Bayes Precision-Recall Curve")

plt.show()

# %%
# MicroAvg
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot()
_ = display.ax_.set_title("Gaussian Naïve Bayes Precision-Recall Curve")

# %% [markdown]
# ### **<font color="#34eb89">6.1.2 Bernoulli Naive Bayes</font>**

# %%
bern_nb = BernoulliNB()
bern_nb.fit(X_train, y_train)
y_pred = bern_nb.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'Bernoulli Naïve Bayes'
cm_bern_nb = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_bern_nb, model)

# %%
# Inizializziamo il classifier
bern_nb = OneVsRestClassifier(BernoulliNB())
y_score = bern_nb.fit(X_train, y_train_bin).predict_proba(X_test)

# Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcoliamo microaverage
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Mettiamo tutti i FPR insieme
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

# Macro
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="MacroAvg (AUC:{0:0.2f})".format(roc_auc["macro"]),
    color="aqua",
    linestyle=":",
    linewidth=3,
)

# Curve per ogni classe
colors = cycle(["black", "darkorange", "blue", "pink", "purple", "green"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1,
        label="Class {0} (AUC:{1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Bernoulli Naïve Bayes ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Bernoulli Naïve Bayes ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# MicroAvg calcola score di tutte le classi
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

_, ax = plt.subplots(figsize=(7, 8))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="MicroAvg Precision-Recall", color="red")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-Recall for class {i}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([-.02, 1.02])
ax.set_ylim([0.0, 1.02])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Bernoulli Naïve Bayes Precision-Recall Curve")

plt.show()

# %%
# MicroAvg
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot()
_ = display.ax_.set_title("Bernoulli Naïve Bayes Precision-Recall Curve")

# %% [markdown]
# ## **<font color="#FBBF44">6.2 Logistic Regression</font>**

# %%
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'Logistic Regression'
cm_logreg = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_logreg, model)

# %%
# Inizializziamo il classifier
log_reg = OneVsRestClassifier(LogisticRegression())
y_score = log_reg.fit(X_train, y_train_bin).predict_proba(X_test)

# Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcoliamo microaverage
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Mettiamo tutti i FPR insieme
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

# Macro
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="MacroAvg (AUC:{0:0.2f})".format(roc_auc["macro"]),
    color="aqua",
    linestyle=":",
    linewidth=3,
)

# Curve per ogni classe
colors = cycle(["black", "darkorange", "blue", "pink", "purple", "green"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1,
        label="Class {0} (AUC:{1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# MicroAvg calcola score di tutte le classi
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

_, ax = plt.subplots(figsize=(7, 8))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="MicroAvg Precision-Recall", color="red")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-Recall for class {i}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([-.02, 1.02])
ax.set_ylim([0.0, 1.02])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Logistic Regression Precision-Recall Curve")

plt.show()

# %%
# MicroAvg
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot()
_ = display.ax_.set_title("Logistic Regression Precision-Recall Curve")

# %% [markdown]
# ### **<font color="#34eb89">6.2.1 Logistic Function</font>**

# %%
data_log = data.copy()
data_log['activity'] = y_label[0].values

# Seleziono le classi che voglio droppare. Lasciamo solo 2 e 3, 6 era troppo facile da predictare-->
classe1 = np.array(y_label[y_label[0]==1].index)
classe4 = np.array(y_label[y_label[0]==4].index)
classe5 = np.array(y_label[y_label[0]==5].index)
classe6 = np.array(y_label[ y_label[0]==6].index)
classes2remove = np.concatenate((classe1,classe4,classe5,classe6))

data_log.drop(data_log.index[classes2remove], inplace=True)

data_logtrans = data_log.copy()
data_logtrans['activity'] = data_log['activity'].replace(to_replace=[2,3], value=[0, 1])

X_train_log = data_log.iloc[:, 0:561].values
y_train_log = data_logtrans['activity'].values

# %%
# Droppiamo le classi nel test per poi evaluare bene la performance del dataset ribilanciato
data_test_log = data_test.copy()
data_test_log['activity'] = y_label_test[0].values

classe1 = np.array(y_label_test[y_label_test[0]==1].index)
classe4 = np.array(y_label_test[y_label_test[0]==4].index)
classe5 = np.array(y_label_test[y_label_test[0]==5].index)
classe6 = np.array(y_label_test[ y_label_test[0]==6].index)
classes2remove = np.concatenate((classe1,classe4,classe5,classe6))

data_test_log.drop(data_test_log.index[classes2remove], inplace=True)

data_test_logtrans = data_test_log.copy()
data_test_logtrans['activity'] = data_test_log['activity'].replace(to_replace=[2,3], value=[0, 1])

X_test_log= data_test_log.iloc[:, 0:561].values
y_test_log = data_test_logtrans['activity'].values

# %%
pca = PCA(n_components=1)
X_train_log_pca = pca.fit_transform(X_train_log)
X_test_log_pca = pca.transform(X_test_log)

# %%
clf = LogisticRegression(random_state=0)
clf.fit(X_train_log_pca.T[0].reshape(-1,1), y_train_log)
y_pred = clf.predict(X_test_log_pca.T[0].reshape(-1,1))

# %%
classes = ['Class 2', 'Class 3']
scatter = plt.scatter(X_train_log_pca.T[0], y_train_log, c=y_train_log, cmap=plt.cm.tab20b, edgecolor='white', linewidth = .7, alpha = .7)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title("Class distribution against Principal Component 1")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig('FigXX-LogisticClass.png', dpi=600)

# %%
from scipy.special import expit
loss = expit(sorted(X_test_log_pca.T[0].reshape(-1,1)) * clf.coef_ + clf.intercept_).ravel()

classes = ['Class 2', 'Class 3']
scatter = plt.scatter(X_train_log_pca.T[0].reshape(-1,1), y_train_log, c=y_train_log, cmap=plt.cm.tab20b, edgecolor='white', linewidth = .7, alpha = .7)
plt.plot(sorted(X_test_log_pca.T[0].reshape(-1,1)), loss, color='darkorange', linewidth=3)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title("Fig. 13 - Logistic Function comparison with Linear Function")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig('FigXX-LogisticFuntion.png', dpi=600)

# %%
reg = LinearRegression()
reg.fit(X_train_log_pca.T[0].reshape(-1,1), y_train_log)

# %%
loss = expit(sorted(X_test_log_pca.T[0].reshape(-1,1)) * clf.coef_ + clf.intercept_).ravel()

classes = ['Class 2', 'Class 3', 'Lin function', 'Lin function' ]
scatter = plt.scatter(X_train_log_pca.T[0].reshape(-1,1), y_train_log, c=y_train_log, cmap=plt.cm.tab20b, edgecolor='white', linewidth = .7, alpha = .7)
plt.plot(sorted(X_test_log_pca.T[0].reshape(-1,1)), loss, color='darkorange', linewidth=3, label='Log function')
plt.plot(sorted(X_test_log_pca.T[0].reshape(-1,1)), reg.coef_ * sorted(X_test_log_pca.T[0].reshape(-1,1)) + reg.intercept_, color='cadetblue', linewidth=3, label='Lin function')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title("Logistic and Linear function")
plt.legend(loc="lower right")
plt.savefig('FigXX-LogisticLinearFuntion.png', dpi=600)

# %% [markdown]
# ## **<font color="#FBBF44">6.3 Support Vector Machines</font>**

# %% [markdown]
# ### **<font color="#34eb89">6.3.1 Linear SVM</font>**

# %%
lin_svc = LinearSVC()
lin_svc.fit(X_train, y_train)

y_pred = lin_svc.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'Linear Support Vector Classification'
cm_lin_svc = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_lin_svc, model)

# %%
lin_svc = LinearSVC(C=1, dual=False, fit_intercept=False, random_state=0)
lin_svc.fit(X_train, y_train)

y_pred = lin_svc.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'Linear Support Vector Classification after tuning'
cm_lin_svc = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_lin_svc, model)

# %%
# Inizializziamo il classifier
lin_svc = OneVsRestClassifier(LinearSVC())
y_score = lin_svc.fit(X_train, y_train_bin).decision_function(X_test)

# Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcoliamo microaverage
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Mettiamo tutti i FPR insieme
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

# Macro
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="MacroAvg (AUC:{0:0.2f})".format(roc_auc["macro"]),
    color="aqua",
    linestyle=":",
    linewidth=3,
)

# Curve per ogni classe
colors = cycle(["black", "darkorange", "blue", "pink", "purple", "green"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1,
        label="Class {0} (AUC:{1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Linear SVC ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Linear SVC ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# MicroAvg calcola score di tutte le classi
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

_, ax = plt.subplots(figsize=(7, 8))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="MicroAvg Precision-Recall", color="red")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-Recall for class {i}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([-.02, 1.02])
ax.set_ylim([0.0, 1.02])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Linear SVC Precision-Recall Curve")

plt.show()

# %%
# MicroAvg
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot()
_ = display.ax_.set_title("Linear SVC Precision-Recall Curve")

# %% [markdown]
# ### **<font color="#34eb89">6.3.2 Non-linear SVM</font>**

# %%
svc = SVC(C=.5, random_state=42)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'Support Vector Classification'
cm_svc = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_svc, model)

# %%
# Inizializziamo il classifier
svc = OneVsRestClassifier(LinearSVC())
y_score = svc.fit(X_train, y_train_bin).decision_function(X_test)

# Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcoliamo microaverage
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Mettiamo tutti i FPR insieme
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

# Macro
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="MacroAvg (AUC:{0:0.2f})".format(roc_auc["macro"]),
    color="aqua",
    linestyle=":",
    linewidth=3,
)

# Curve per ogni classe
colors = cycle(["black", "darkorange", "blue", "pink", "purple", "green"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1,
        label="Class {0} (AUC:{1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVC ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVC ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# MicroAvg calcola score di tutte le classi
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

_, ax = plt.subplots(figsize=(7, 8))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="MicroAvg Precision-Recall", color="red")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-Recall for class {i}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([-.02, 1.02])
ax.set_ylim([0.0, 1.02])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("SVC Precision-Recall Curve")

plt.show()

# %%
# MicroAvg
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot()
_ = display.ax_.set_title("SVC Precision-Recall Curve")

# %% [markdown]
# ## **<font color="#FBBF44">6.4 Neural Networks</font>**

# %%
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

# %%
mlp = MLPClassifier()
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'Multi Layer Perceptron'
cm_mlp = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_mlp, model)

# %%
mlp1 = MLPClassifier().fit(X_train2, y_train2)
mlp2 = MLPClassifier().fit(X_val, y_val)

# %%
plt.plot(mlp1.loss_curve_, color = 'blue', label = 'train')
plt.plot(mlp2.loss_curve_, color = 'orange', label = 'validation')
plt.title("Multi Layer Perceptron Loss Curve")
plt.legend(loc="upper right")
plt.savefig('FigXX-MLPLossCurve.png', dpi=600)
plt.show()

# %%
mlp = MLPClassifier(alpha=0.05, hidden_layer_sizes=(555,), learning_rate='adaptive', max_iter=1000, solver='sgd')
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'Multi Layer Perceptron after tuning'
cm_mlp = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_mlp, model)

# %%
mlp1 = MLPClassifier(alpha=0.05, hidden_layer_sizes=(555,), learning_rate='adaptive', max_iter=1000, solver='sgd').fit(X_train2, y_train2)
mlp2 = MLPClassifier(alpha=0.05, hidden_layer_sizes=(555,), learning_rate='adaptive', max_iter=1000, solver='sgd').fit(X_val, y_val)

# %%
plt.plot(mlp1.loss_curve_, color = 'blue', label = 'train')
plt.plot(mlp2.loss_curve_, color = 'orange', label = 'validation')
plt.title("Multi Layer Perceptron Loss Curve after tuning")
plt.legend(loc="upper right")
plt.savefig('FigXX-MLPTunedLossCurve.png', dpi=600)
plt.show()

# %%
mlp = MLPClassifier(alpha=0.0001, hidden_layer_sizes=(555,), learning_rate='adaptive', max_iter=1000, solver='sgd')
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

# %%
import warnings
from sklearn.exceptions import ConvergenceWarning

params = [
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "momentum": 0,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "momentum": 0.9,
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "momentum": 0,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "momentum": 0.9,
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {"solver": "adam", "learning_rate_init": 0.01},
]

labels = [
    "constant learning-rate",
    "constant with momentum",
    "constant with Nesterov's momentum",
    "inv-scaling learning-rate",
    "inv-scaling with momentum",
    "inv-scaling with Nesterov's momentum",
    "adam",
]

plot_args = [
    {"c": "navy", "linestyle": "-"},
    {"c": "orangered", "linestyle": "--"},
    {"c": "darkslategray", "linestyle": "-"},
    {"c": "purple", "linestyle": "--"},
    {"c": "limegreen", "linestyle": "-"},
    {"c": "turquoise", "linestyle": "--"},
    {"c": "gold", "linestyle": "-"},
]

# %%
def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name, fontsize=16)

    X = MinMaxScaler().fit_transform(X)
    mlps = []

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(random_state=0, max_iter=400, **param)

        # some parameter combinations will not converge as can be seen on the
        # plots so they are ignored here
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module="sklearn"
            )
            mlp.fit(X, y)

        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, **args)


fig, axes = plt.subplots(figsize=(15, 10))

# load / generate some toydatasets
data_sets = [(X_train, y_train)]

plot_on_dataset(X_train, y_train, axes, name="Multi-layer Perceptron Comparison on Learning Strategies")
plt.xlabel("Iteration")
fig.legend(axes.get_lines(), labels, ncol=3, loc="upper center", fontsize=12)
plt.savefig('FigXX-MLPLearningComparison.png', dpi=600)
plt.show()

# %%
# Inizializziamo il classifier
mlp = OneVsRestClassifier(MLPClassifier())
y_score = mlp.fit(X_train, y_train_bin).predict_proba(X_test)

# Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcoliamo microaverage
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Mettiamo tutti i FPR insieme
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

# Macro
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="MacroAvg (AUC:{0:0.2f})".format(roc_auc["macro"]),
    color="aqua",
    linestyle=":",
    linewidth=3,
)

# Curve per ogni classe
colors = cycle(["black", "darkorange", "blue", "pink", "purple", "green"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1,
        label="Class {0} (AUC:{1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi Layer Perceptron ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi Layer Perceptron ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# MicroAvg calcola score di tutte le classi
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

_, ax = plt.subplots(figsize=(7, 8))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="MicroAvg Precision-Recall", color="red")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-Recall for class {i}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([-.02, 1.02])
ax.set_ylim([0.0, 1.02])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Multi Layer Perceptron Precision-Recall Curve")

plt.show()

# %%
# MicroAvg
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot()
_ = display.ax_.set_title("Multi Layer Perceptron Precision-Recall Curve")

# %% [markdown]
# ## **<font color="#FBBF44">6.5 Ensemble Methods</font>**

# %% [markdown]
# ### **<font color="#34eb89">6.5.1 Random Forest</font>**

# %%
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'Random Forest'
cm_rfc = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_rfc, model)

# %%
# Inizializziamo il classifier
rfc = OneVsRestClassifier(RandomForestClassifier())
y_score = rfc.fit(X_train, y_train_bin).predict_proba(X_test)

# Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcoliamo microaverage
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Mettiamo tutti i FPR insieme
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

# Macro
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="MacroAvg (AUC:{0:0.2f})".format(roc_auc["macro"]),
    color="aqua",
    linestyle=":",
    linewidth=3,
)

# Curve per ogni classe
colors = cycle(["black", "darkorange", "blue", "pink", "purple", "green"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1,
        label="Class {0} (AUC:{1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# MicroAvg calcola score di tutte le classi
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

_, ax = plt.subplots(figsize=(7, 8))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="MicroAvg Precision-Recall", color="red")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-Recall for class {i}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([-.02, 1.02])
ax.set_ylim([0.0, 1.02])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Random Forest Precision-Recall Curve")

plt.show()

# %%
# MicroAvg
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot()
_ = display.ax_.set_title("Random Forest Precision-Recall Curve")

# %% [markdown]
#

# %% [markdown]
# ### **<font color="#34eb89">6.5.2 Extra Trees</font>**

# %%
etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)

y_pred = etc.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'Extra Trees'
cm_etc = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_etc, model)

# %%
# Inizializziamo il classifier
etc = OneVsRestClassifier(ExtraTreesClassifier())
y_score = etc.fit(X_train, y_train_bin).predict_proba(X_test)

# Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcoliamo microaverage
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Mettiamo tutti i FPR insieme
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

# Macro
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="MacroAvg (AUC:{0:0.2f})".format(roc_auc["macro"]),
    color="aqua",
    linestyle=":",
    linewidth=3,
)

# Curve per ogni classe
colors = cycle(["black", "darkorange", "blue", "pink", "purple", "green"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1,
        label="Class {0} (AUC:{1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Extra Trees ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Extra Trees ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# MicroAvg calcola score di tutte le classi
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

_, ax = plt.subplots(figsize=(7, 8))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="MicroAvg Precision-Recall", color="red")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-Recall for class {i}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([-.02, 1.02])
ax.set_ylim([0.0, 1.02])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Extra Trees Precision-Recall Curve")

plt.show()

# %%
# MicroAvg
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot()
_ = display.ax_.set_title("Extra Trees Precision-Recall Curve")

# %% [markdown]
# ### **<font color="#34eb89">6.5.4 Bagging</font>**

# %%
bag = BaggingClassifier()
bag.fit(X_train, y_train)

y_pred = bag.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'Bagging Classifier'
cm_bag = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_bag, model)

# %%
# Inizializziamo il classifier
bag = OneVsRestClassifier(BaggingClassifier())
y_score = bag.fit(X_train, y_train_bin).predict_proba(X_test)

# Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcoliamo microaverage
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Mettiamo tutti i FPR insieme
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

# Macro
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="MacroAvg (AUC:{0:0.2f})".format(roc_auc["macro"]),
    color="aqua",
    linestyle=":",
    linewidth=3,
)

# Curve per ogni classe
colors = cycle(["black", "darkorange", "blue", "pink", "purple", "green"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1,
        label="Class {0} (AUC:{1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Bagging ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Bagging ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# MicroAvg calcola score di tutte le classi
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

_, ax = plt.subplots(figsize=(7, 8))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="MicroAvg Precision-Recall", color="red")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-Recall for class {i}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([-.02, 1.02])
ax.set_ylim([0.0, 1.02])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Bagging Precision-Recall Curve")

plt.show()

# %%
# MicroAvg
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot()
_ = display.ax_.set_title("Bagging Precision-Recall Curve")

# %% [markdown]
# ### **<font color="#34eb89">6.5.5 GradientBoost</font>**

# %%
# gbc = GradientBoostingClassifier()
# gbc.fit(X_train, y_train)

# y_pred = gbc.predict(X_test)

# print('Accuracy %s' % accuracy_score(y_test, y_pred))
# print('F1-score %s' % f1_score(y_test, y_pred, average=None))
# print(classification_report(y_test, y_pred, digits=4))

# model = 'Gradient Boosting Classifier'
# cm_gbc = confusion_matrix(y_test,y_pred)
# draw_confusion_matrix(cm_gbc, model)

# %%
# # Inizializziamo il classifier
# gbc = OneVsRestClassifier(GradientBoostingClassifier())
# y_score = gbc.fit(X_train, y_train_bin).predict_proba(X_test)

# # Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
# fpr = {}
# tpr = {}
# roc_auc = {}
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Calcoliamo microaverage
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # Mettiamo tutti i FPR insieme
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# mean_tpr /= n_classes

# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# # Plottiamo i ROC curves
# plt.figure()

# # Micro
# plt.plot(
#     fpr["micro"],
#     tpr["micro"],
#     label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
#     color="red",
#     linestyle=":",
#     linewidth=3,
# )

# # Macro
# plt.plot(
#     fpr["macro"],
#     tpr["macro"],
#     label="MacroAvg (AUC:{0:0.2f})".format(roc_auc["macro"]),
#     color="aqua",
#     linestyle=":",
#     linewidth=3,
# )

# # Curve per ogni classe
# colors = cycle(["black", "darkorange", "blue", "pink", "purple", "green"])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(
#         fpr[i],
#         tpr[i],
#         color=color,
#         lw=1,
#         label="Class {0} (AUC:{1:0.2f})".format(i, roc_auc[i]),
#     )

# plt.plot([0, 1], [0, 1], "k--", lw=1)
# plt.xlim([-.02, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("GradientBoosting ROC Curve")
# plt.legend(loc="lower right")
# plt.show()

# %%
# # Plottiamo i ROC curves
# plt.figure()

# # Micro
# plt.plot(
#     fpr["micro"],
#     tpr["micro"],
#     label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
#     color="red",
#     linestyle=":",
#     linewidth=3,
# )

# plt.plot([0, 1], [0, 1], "k--", lw=1)
# plt.xlim([-.02, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("GradientBoosting ROC Curve")
# plt.legend(loc="lower right")
# plt.show()

# %%
# # Precision-Recall Curve
# precision = dict()
# recall = dict()
# average_precision = dict()
# for i in range(n_classes):
#     precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
#     average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# # MicroAvg calcola score di tutte le classi
# precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
# average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

# colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

# _, ax = plt.subplots(figsize=(7, 8))

# display = PrecisionRecallDisplay(
#     recall=recall["micro"],
#     precision=precision["micro"],
#     average_precision=average_precision["micro"],
# )
# display.plot(ax=ax, name="MicroAvg Precision-Recall", color="red")

# for i, color in zip(range(n_classes), colors):
#     display = PrecisionRecallDisplay(
#         recall=recall[i],
#         precision=precision[i],
#         average_precision=average_precision[i],
#     )
#     display.plot(ax=ax, name=f"Precision-Recall for class {i}", color=color)

# handles, labels = display.ax_.get_legend_handles_labels()
# ax.set_xlim([-.02, 1.02])
# ax.set_ylim([0.0, 1.02])
# ax.legend(handles=handles, labels=labels, loc="best")
# ax.set_title("GradientBoosting Precision-Recall Curve")

# plt.show()

# %%
# # MicroAvg
# display = PrecisionRecallDisplay(
#     recall=recall["micro"],
#     precision=precision["micro"],
#     average_precision=average_precision["micro"],
# )
# display.plot()
# _ = display.ax_.set_title("GradientBoosting Precision-Recall Curve")

# %% [markdown]
# ### **<font color="#34eb89">6.5.6 XGBoost</font>**

# %%
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'XGBoost Classifier'
cm_xgb = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_xgb, model)

# %%
xgb = XGBClassifier(base_score=0.5, learning_rate=0.25, max_depth=2,
              n_estimators=300, objective='multi:softprob',
              tree_method='exact', random_state=0)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'XGBoost Classifier after tuning'
cm_xgb = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_xgb, model)

# %%
# Inizializziamo il classifier
xgb = OneVsRestClassifier(XGBClassifier())
y_score = xgb.fit(X_train, y_train_bin).predict_proba(X_test)

# Calcoliamo FalsePositiveRate e TruePositiveRate per ogni classe
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcoliamo microaverage
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Mettiamo tutti i FPR insieme
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

# Macro
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="MacroAvg (AUC:{0:0.2f})".format(roc_auc["macro"]),
    color="aqua",
    linestyle=":",
    linewidth=3,
)

# Curve per ogni classe
colors = cycle(["black", "darkorange", "blue", "pink", "purple", "green"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1,
        label="Class {0} (AUC:{1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Plottiamo i ROC curves
plt.figure()

# Micro
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="MicroAvg (AUC:{0:0.2f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=3,
)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([-.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %%
# Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# MicroAvg calcola score di tutte le classi
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "purple"])

_, ax = plt.subplots(figsize=(7, 8))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="MicroAvg Precision-Recall", color="red")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-Recall for class {i}", color=color)

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([-.02, 1.02])
ax.set_ylim([0.0, 1.02])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("XGBoost Precision-Recall Curve")

plt.show()

# %%
# MicroAvg
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot()
_ = display.ax_.set_title("XGBoost Precision-Recall Curve")

# %% [markdown]
# # **<font color="#42f5f5">7.0 REGRESSION TASK</font>**

# %% [markdown]
# ## **<font color="#FBBF44">7.1 Simple Linear Regression</font>**

# %%
reg = LinearRegression()
reg.fit(X_train[:,366].reshape(-1, 1), X_train[:,367].reshape(-1, 1))
y_pred = reg.predict(X_test[:,367].reshape(-1, 1)).reshape(1,-1)[0]

plt.scatter(X_test[:,366], X_test[:,367], color='skyblue',edgecolor='white', linewidth = .7, alpha = .7)
plt.plot(X_test[:,367], y_pred, color='orangered', linewidth=3, alpha = .7)
plt.title('Most correlated variables according to ANOVA F-Value')
plt.savefig('FigXX-LinearRegression1.png', dpi=600)

print('R2: %.3f' % r2_score(X_test[:,367], y_pred))
print('MSE: %.3f' % mean_squared_error(X_test[:,367], y_pred))
print('MAE: %.3f' % mean_absolute_error(X_test[:,367], y_pred))

reg_diff = pd.DataFrame({'Actual value': X_test[:,367], 'Predicted value': y_pred})
reg_diff

# %%
reg = LinearRegression()
reg.fit(X_train[:,92].reshape(-1, 1), X_train[:,93].reshape(-1, 1))
y_pred = reg.predict(X_test[:,93].reshape(-1, 1)).reshape(1,-1)[0]

plt.scatter(X_test[:,92], X_test[:,93], color='skyblue',edgecolor='white', linewidth = .7, alpha = .7)
plt.plot(X_test[:,93], y_pred, color='orangered', linewidth=3, alpha = .7)
plt.title('Most correlated variables with the target class')
plt.savefig('FigXX-LinearRegression2.png', dpi=600)
plt.show()

print('R2: %.3f' % r2_score(X_test[:,93], y_pred))
print('MSE: %.3f' % mean_squared_error(X_test[:,93], y_pred))
print('MAE: %.3f' % mean_absolute_error(X_test[:,93], y_pred))

reg_diff = pd.DataFrame({'Actual value': X_test[:,93], 'Predicted value': y_pred})
reg_diff

# %%
reg = LinearRegression()
reg.fit(X_train[:,35].reshape(-1, 1), X_train[:,93].reshape(-1, 1))
y_pred = reg.predict(X_test[:,93].reshape(-1, 1)).reshape(1,-1)[0]

plt.scatter(X_test[:,35], X_test[:,93], color='skyblue',edgecolor='white', linewidth = .7, alpha = .7)
plt.plot(X_test[:,93], y_pred, color='orangered', linewidth=3, alpha = .7)
plt.title('Most correlated variable and a random pick')
plt.savefig('FigXX-LinearRegression3.png', dpi=600)
plt.show()

print('R2: %.3f' % r2_score(X_test[:,93], y_pred))
print('MSE: %.3f' % mean_squared_error(X_test[:,93], y_pred))
print('MAE: %.3f' % mean_absolute_error(X_test[:,93], y_pred))

reg_diff = pd.DataFrame({'Actual value': X_test[:,93], 'Predicted value': y_pred})
reg_diff

# %% [markdown]
# ## **<font color="#FBBF44">7.2 Multiple Linear Regression</font>**
#

# %%
normalized_df = (data-data.min())/(data.max()-data.min())
normalized_df_test = (data_test-data_test.min())/(data_test.max()-data_test.min())

X = normalized_df.iloc[:,0:561]
y = y_label

Xtest = normalized_df_test.iloc[:,0:561]
ytest = y_label_test

sel_uni = SelectKBest(score_func=chi2, k=35)
X_train_sel_uni = sel_uni.fit(X,y)

dfscores = pd.DataFrame(X_train_sel_uni.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']

print(featureScores.nlargest(35,'Score'))

X_test_sel_uni = sel_uni.fit_transform(Xtest,ytest)

# %%
bestk = featureScores.nlargest(34,'Score')

data.columns=feature_list
data_test.columns=feature_list
data_reg = data.copy()
data_reg_test = data_test.copy()

data_reg = data_reg[bestk['Feature']]
data_reg_test = data_reg_test[bestk['Feature']]

X_train_reg = data_reg.values
X_test_reg = data_reg_test.values

# %%
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = data[list(data.columns[:-2])]

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False)

# %%
vif_infinito = vif_info.loc[vif_info['VIF'] != np.inf]
vif_cols = vif_infinito.loc[vif_infinito['VIF'] < 10 ]
print(len(vif_cols))
vif_cols.sort_values('VIF', ascending=False)

# %%
data.columns=feature_list
data_test.columns=feature_list

data_reg_vif = data.copy()
data_reg_test_vif = data_test.copy()

data_reg_vif = data_reg_vif[vif_cols['Column']]
data_reg_test_vif = data_reg_test_vif[vif_cols['Column']]

X_train_reg2 = data_reg_vif.values
X_test_reg2 = data_reg_test_vif.values

# %% [markdown]
# **<font color="#34eb89">Regular</font>**

# %%
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg2, y_train)
y_pred = lin_reg.predict(X_test_reg2)

print('R2: %.3f' % r2_score(y_test, y_pred))
print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

reg_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
reg_diff

# %%
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

print('R2: %.3f' % r2_score(y_test, y_pred))
print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

reg_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
reg_diff

# %% [markdown]
# **<font color="#34eb89">Ridge</font>**

# %%
lin_reg_ridge = Ridge()
lin_reg_ridge.fit(X_train, y_train)
y_pred = lin_reg_ridge.predict(X_test)

print('R2: %.3f' % r2_score(y_test, y_pred))
print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

reg_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
reg_diff

# %% [markdown]
# **<font color="#34eb89">Lasso</font>**

# %%
lin_reg_lasso = Lasso()
lin_reg_lasso.fit(X_train_pca, y_train)
y_pred = lin_reg_lasso.predict(X_test_pca)

print('R2: %.3f' % r2_score(y_test, y_pred))
print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

reg_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
reg_diff

# %% [markdown]
# **<font color="#34eb89">GradientBoosting</font>**

# %%
xgb = XGBClassifier()
xgb.fit(X_train_reg2, y_train)

y_pred = xgb.predict(X_test_reg2)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred, digits=4))

model = 'XGBoost Classifier'
cm_xgb = confusion_matrix(y_test,y_pred)
draw_confusion_matrix(cm_xgb, model)

# %% [markdown]
# **Gradient Boosting Regressor**

# %%
gbr = GradientBoostingRegressor()
gbr.fit(X_train_reg2, y_train)

y_pred = gbr.predict(X_test_reg2)

print('R2: %.3f' % r2_score(y_test, y_pred))
print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

# %% [markdown]
# # **<font color="#42f5f5">8.0 TIME SERIES ANALYSIS</font>**

# %% [markdown]
# ## **<font color="#FBBF44">8.x Preprocessing</font>**

# %%
data_ts0=pd.read_csv('/content/local_data/train/Inertial Signals/body_acc_x_train.txt', header=None, delim_whitespace=True)
data_ts1=pd.read_csv('/content/local_data/train/Inertial Signals/body_acc_y_train.txt', header=None, delim_whitespace=True)
data_ts2=pd.read_csv('/content/local_data/train/Inertial Signals/body_acc_z_train.txt', header=None, delim_whitespace=True)
data_ts3=pd.read_csv('/content/local_data/train/Inertial Signals/body_gyro_x_train.txt', header=None, delim_whitespace=True)
data_ts4=pd.read_csv('/content/local_data/train/Inertial Signals/body_gyro_y_train.txt', header=None, delim_whitespace=True)
data_ts5=pd.read_csv('/content/local_data/train/Inertial Signals/body_gyro_z_train.txt', header=None, delim_whitespace=True)
data_ts6=pd.read_csv('/content/local_data/train/Inertial Signals/total_acc_x_train.txt', header=None, delim_whitespace=True)
data_ts7=pd.read_csv('/content/local_data/train/Inertial Signals/total_acc_y_train.txt', header=None, delim_whitespace=True)
data_ts8=pd.read_csv('/content/local_data/train/Inertial Signals/total_acc_z_train.txt', header=None, delim_whitespace=True)

data_test_ts0=pd.read_csv('/content/local_data/test/Inertial Signals/body_acc_x_test.txt', header=None, delim_whitespace=True)
data_test_ts1=pd.read_csv('/content/local_data/test/Inertial Signals/body_acc_y_test.txt', header=None, delim_whitespace=True)
data_test_ts2=pd.read_csv('/content/local_data/test/Inertial Signals/body_acc_z_test.txt', header=None, delim_whitespace=True)
data_test_ts3=pd.read_csv('/content/local_data/test/Inertial Signals/body_gyro_x_test.txt', header=None, delim_whitespace=True)
data_test_ts4=pd.read_csv('/content/local_data/test/Inertial Signals/body_gyro_y_test.txt', header=None, delim_whitespace=True)
data_test_ts5=pd.read_csv('/content/local_data/test/Inertial Signals/body_gyro_z_test.txt', header=None, delim_whitespace=True)
data_test_ts6=pd.read_csv('/content/local_data/test/Inertial Signals/total_acc_x_test.txt', header=None, delim_whitespace=True)
data_test_ts7=pd.read_csv('/content/local_data/test/Inertial Signals/total_acc_y_test.txt', header=None, delim_whitespace=True)
data_test_ts8=pd.read_csv('/content/local_data/test/Inertial Signals/total_acc_z_test.txt', header=None, delim_whitespace=True)

# %%
ts_dict=[data_ts0.std(), data_ts1.std(), data_ts2.std(), data_ts3.std(), data_ts4.std(), data_ts5.std(), data_ts6.std(), data_ts7.std(), data_ts8.std()]

# %%
std_data_ts0=data_ts0.std()
std_data_ts1=data_ts1.std()
std_data_ts2=data_ts2.std()
std_data_ts3=data_ts3.std()
std_data_ts4=data_ts4.std()
std_data_ts5=data_ts5.std()
std_data_ts6=data_ts6.std()
std_data_ts7=data_ts7.std()
std_data_ts8=data_ts8.std()

# %%
print("standard deviation of: data_ts: " ,std_data_ts0.mean())
print("standard deviation of: data_ts1: " ,std_data_ts1.mean())
print("standard deviation of: data_ts2: " ,std_data_ts2.mean())
print("standard deviation of: data_ts3: " ,std_data_ts3.mean())
print("standard deviation of: data_ts4: " ,std_data_ts4.mean())
print("standard deviation of: data_ts5: " ,std_data_ts5.mean())
print("standard deviation of: data_ts6: " ,std_data_ts6.mean())
print("standard deviation of: data_ts7: " ,std_data_ts7.mean())
print("standard deviation of: data_ts8: " ,std_data_ts8.mean())

# %%
data_ts6

# %%
full_ts = pd.concat([data_ts0,data_ts1,data_ts2,data_ts3,data_ts4,data_ts5,data_ts6,data_ts7,data_ts8],axis=0)
full_ts

full_test_ts = pd.concat([data_test_ts0,data_test_ts1,data_test_ts2,data_test_ts3,data_test_ts4,data_test_ts5,data_test_ts6,data_test_ts7,data_test_ts8],axis=0)
full_test_ts

print(full_ts.shape)

print(full_test_ts.shape)

full_ts

# %%
y1 = y2 = y3 = y4 = y5 = y6 = y7 = y8 = y9 = y_train
yt1 = yt2 = yt3 = yt4 = yt5 = yt6 = yt7 = yt8 = yt9 = y_test

y_fulltrain = np.concatenate((y1, y2, y3, y4, y5, y6, y7, y8, y9))
y_fulltest = np.concatenate((yt1, yt2, yt3, yt4, yt5, yt6, yt7, yt8, yt9))

print(y_fulltrain.shape)
print(y_fulltest.shape)

# %%
# Calcoliamo i features

def calculate_features(values):
    features = {
        'avg': np.mean(values),
        'std': np.std(values),
        'var': np.var(values),
        'med': np.median(values),
        '10p': np.percentile(values, 10),
        '25p': np.percentile(values, 25),
        '50p': np.percentile(values, 50),
        '75p': np.percentile(values, 75),
        '90p': np.percentile(values, 90),
        'iqr': np.percentile(values, 75) - np.percentile(values, 25),
        'cov': 1.0 * np.mean(values) / np.std(values),
        'skw': stats.skew(values),
        'kur': stats.kurtosis(values)
    }
    return features

# %%
data_ts = (data_ts6-data_ts6.min())/(data_ts6.max()-data_ts6.min())
data_test_ts = (data_test_ts6-data_test_ts6.min())/(data_test_ts6.max()-data_test_ts6.min())

data_ts.shape
data_ts

# %%
X_train_ts = data_ts.values
X_test_ts = data_test_ts.values

X_fulltrain_ts = full_ts.values
X_fulltest_ts = full_test_ts.values

# %% [markdown]
# ## **<font color="#FBBF44">8.x Understanding</font>**

# %%
# from tslearn.generators import random_walks
# Xtoy = random_walks(n_ts=30, sz=20, d=1)
# print(Xtoy.shape)

# %%
# X_train_ts, y_train_ts, X_test_ts, y_test_ts = CachedDatasets().load_dataset("Trace")

# print(X_train_ts.shape)
# X_train_ts

# %%
data_ts = (data_ts6-data_ts6.min())/(data_ts6.max()-data_ts6.min())
data_test_ts = (data_test_ts6-data_test_ts6.min())/(data_test_ts6.max()-data_test_ts6.min())

# %%
data_ts.shape

# %%
ts1 = data_ts.iloc[0,:]

plt.figure(figsize=(10,2))
plt.plot(ts1)
plt.xlim([0, 127])
plt.title('total_acc_x_train[0]')
plt.savefig('FigXX-RawTS.png', dpi=600)
plt.show()

# %%
ts1 = data_ts.iloc[7351,:]

plt.figure(figsize=(10,2))
plt.plot(ts1)
plt.xlim([0, 127])
plt.title('total_acc_x_train[7351]')
plt.savefig('FigXX-RawTS2.png', dpi=600)
plt.show()

# %%
# pca = PCA(n_components=3)
# data_pca = pca.fit_transform(data_ts)

# ts_pca = data_pca
# plt.plot(ts_pca)
# plt.show()

# %%
ts1 = data_ts.iloc[0,:]
features = calculate_features(ts1)
features

# %% [markdown]
# ## **<font color="#FBBF44">8.x Transformations</font>**

# %%
# data_ts = data_ts6
# data_test = data_test_ts6

# X_train_ts = data_ts.values
# X_test_ts = data_test_ts.values

# %%
data_ts = (data_ts6-data_ts6.min())/(data_ts6.max()-data_ts6.min())
data_test_ts = (data_test_ts6-data_test_ts6.min())/(data_test_ts6.max()-data_test_ts6.min())

# %%
# Timeseries originali
ts1 = data_ts.iloc[2153,:]
ts2 = data_ts.iloc[4671,:]

plt.figure(figsize=(10,3))
plt.plot(ts1, label = 'total_acc_x_train[2153]')
plt.plot(ts2, label = 'total_acc_x_train[4671]')
plt.title('Original Time Series without transformations')
plt.xlim([0, 127])
plt.legend(loc="upper right")
plt.savefig('FigXX-TSbeforeTransform.png', dpi=600)
plt.show()

dist_euc = euclidean(ts1, ts2)
dist_man = cityblock(ts1, ts2)
print(dist_euc)
print(dist_man)

# %%
# Offset Translation
ts1 = data_ts.iloc[2153,:]
ts2 = data_ts.iloc[4671,:]
ts1 = ts1 - ts1.mean()
ts2 = ts2 - ts2.mean()

plt.figure(figsize=(10,3))
plt.plot(ts1, label = 'total_acc_x_train[2153]')
plt.plot(ts2, label = 'total_acc_x_train[4671]')
plt.title('Offset Translation')
plt.xlim([0, 127])
plt.legend(loc="upper right")
plt.savefig('FigXX-TransformOffset.png', dpi=600)
plt.show()

dist_euc = euclidean(ts1, ts2)
dist_man = cityblock(ts1, ts2)
print(dist_euc)
print(dist_man)

# %%
# Amplitude Scaling
ts1 = data_ts.iloc[2153,:]
ts2 = data_ts.iloc[4671,:]
ts1 = (ts1 - ts1.mean())/ts1.std()
ts2 = (ts2 - ts2.mean())/ts2.std()

plt.figure(figsize=(10,3))
plt.plot(ts1, label = 'total_acc_x_train[2153]')
plt.plot(ts2, label = 'total_acc_x_train[4671]')
plt.title('Amplitude Scaling')
plt.xlim([0, 127])
plt.legend(loc="upper right")
plt.savefig('FigXX-TransformAmplitude.png', dpi=600)
plt.show()

dist_euc = euclidean(ts1, ts2)
dist_man = cityblock(ts1, ts2)
print(dist_euc)
print(dist_man)

# %%
# Trend Removal
w = 3
ts1 = data_ts.iloc[2153,:]
ts2 = data_ts.iloc[4671,:]

plt.figure(figsize=(10,3))
plt.plot(ts1)
plt.plot(ts2)
plt.plot(ts1.rolling(window=w).mean(), label = 'TimeSeries 1 Trend')
plt.plot(ts2.rolling(window=w).mean(), label = 'TimeSeries 2 Trend')
plt.title('Time Series and their trends')
plt.xlim([0, 127])
plt.legend(loc="upper right")
plt.savefig('FigXX-TransformTrends.png', dpi=600)
plt.show()

# %%
ts1 = data_ts.iloc[2153,:]
ts2 = data_ts.iloc[4671,:]
ts1 = ts1 - ts1.rolling(window=w).mean()
ts2 = ts2 - ts2.rolling(window=w).mean()

ts1[np.isnan(ts1)] = 0
ts2[np.isnan(ts2)] = 0

plt.figure(figsize=(10,3))
plt.plot(ts1, label = 'total_acc_x_train[2153]')
plt.plot(ts2, label = 'total_acc_x_train[4671]')
plt.title('Removing Trends')
plt.xlim([0, 127])
plt.legend(loc="upper right")
plt.savefig('FigXX-TransformTrendsRemoved.png', dpi=600)
plt.show()

dist_euc = euclidean(ts1, ts2)
dist_man = cityblock(ts1, ts2)
print(dist_euc)
print(dist_man)

# %%
# Removing Noise
ts1 = data_ts.iloc[2153,:]
ts2 = data_ts.iloc[4671,:]
ts1 = ((ts1 - ts1.mean())/ts1.std()).rolling(window=w).mean()
ts2 = ((ts2 - ts2.mean())/ts2.std()).rolling(window=w).mean()

ts1[np.isnan(ts1)] = 0
ts2[np.isnan(ts2)] = 0

plt.figure(figsize=(10,3))
plt.plot(ts1, label = 'total_acc_x_train[2153]')
plt.plot(ts2, label = 'total_acc_x_train[4671]')
plt.title('Removing Noise')
plt.xlim([0, 127])
plt.legend(loc="upper right")
plt.savefig('FigXX-TransformNoise.png', dpi=600)
plt.show()

dist_euc = euclidean(ts1, ts2)
dist_man = cityblock(ts1, ts2)
print(dist_euc)
print(dist_man)

# %%
# Combinando transformazioni
ts1 = data_ts.iloc[2153,:]
ts2 = data_ts.iloc[4671,:]

ts1x = ts1 - ts1.mean()
ts2x = ts2 - ts2.mean()

# ts1x = (ts1x - ts1x.mean())/ts1x.std()
# ts2x = (ts2x - ts2x.mean())/ts2x.std()

ts1x = ts1x - (ts1x.rolling(window=w).mean())
ts2x = ts1x - (ts1x.rolling(window=w).mean())

# ts1x = ((ts1x - ts1x.mean())/ts1x.std()).rolling(window=w).mean()
# ts2x = ((ts2x - ts2x.mean())/ts2x.std()).rolling(window=w).mean()

ts1x[np.isnan(ts1x)] = 0
ts2x[np.isnan(ts2x)] = 0

plt.figure(figsize=(10,3))
plt.plot(ts1x, label = 'total_acc_x_train[2153]')
plt.plot(ts2x, label = 'total_acc_x_train[4671]')
plt.title('Combination of different transformations')
plt.xlim([0, 127])
plt.legend(loc="upper right")
plt.savefig('FigXX-TransformCombination.png', dpi=600)
plt.show()

dist_euc = euclidean(ts1x, ts2x)
dist_man = cityblock(ts1x, ts2x)
print(dist_euc)
print(dist_man)

# %% [markdown]
# ## **<font color="#FBBF44">8.x Dynamic Time Warping</font>**

# %%
%pylab inline
from pylab import rcParams
rcParams['figure.figsize'] = 16,4

def dtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)

    # D0 = D1 = matrix of point-to-point costs
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view (hide first column and first row)

    # Fill the point-to-point costs matrix
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])

    # C = matrix of optimal paths costs
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])

    # Infer the path from matrix C
    if len(x)==1:
        path = zeros(len(y)), range(len(y))  # special case 1
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))  # special case 2
    else:
        path = _traceback(D0)  # general case

    return D1[-1, -1], C, D1, path

# Function for inferring the optimal path (general case)
# Starts from last cell and goes backward...
def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def distance(x,y):
    return abs(x-y)

def nice_table(cost_matrix, title, first_timeseries, second_timeseries):
    df = pd.DataFrame(cost_matrix.transpose().astype(int))[::-1]
    df.columns = first_timeseries
    df.index = second_timeseries[::-1]
    mask = np.zeros_like(df)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if(np.array(df)[i][j] == -1):
                mask[i][j] = True
    sns.set_context('notebook', font_scale=2.5)
    ax = sns.heatmap(df, annot=True, fmt="d", cbar=False, mask=mask)
    ax.set_title(title)

def dtw_band(x, y, dist, band=inf):
    """
    Computes Dynamic Time Warping (DTW) of two sequences with Sakoe-Chiba band.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int band: size of Sakow-Chiba band (default=inf)

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)

    # D0 = D1 = matrix of point-to-point costs
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view (hide first column and first row)

    # Fill the point-to-point costs matrix
    # Effect of bands: cells farther than "band" from diagonal have "inf" cost
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j]) if abs(i-j)<band else inf

    # C = matrix of optimal paths costs
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])

    # Infer the path from matrix C
    if len(x)==1:
        path = zeros(len(y)), range(len(y))  # special case 1
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))  # special case 2
    else:
        path = _traceback(D0)  # general case

    return D1[-1, -1], C, D1, path

# Function for inferring the optima path (general case)
# Starts from last cell and goes backward...
def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def dtw_parallel(x, y, dist, coeff=inf):
    """
    Computes Dynamic Time Warping (DTW) of two sequences with Itakura Parallelogram constraints

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param float coeff: angular coefficient of parallelogram (default=inf)

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j]) if abs(j-i) < (min(i,j,r-i,c-j)+1)*coeff else inf
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path

# %%
data_ts = data_ts6
data_test = data_test_ts6

X_train_ts = data_ts.values
X_test_ts = data_test_ts.values

# %%
# data_ts = (data_ts6-data_ts6.min())/(data_ts6.max()-data_ts6.min())
# data_test_ts = (data_test_ts6-data_test_ts6.min())/(data_test_ts6.max()-data_test_ts6.min())

# %%
ts1 = data_ts.iloc[7345,:]
ts2 = data_ts.iloc[3574,:]

ts1 = ts1[:10]
ts2 = ts2[:10]

path, dist = dtw_path(ts1, ts2)
print(dist)
print(path)

# %%
mat = cdist(ts1.values.reshape(-1,1), ts2.values.reshape(-1,1))

plt.figure(figsize=(8,8))
plt.imshow(mat)
plt.yticks(range(10))
plt.xticks(range(10))
plt.title('Point-to-point Cost Matrix of total_acc_x_train[7345] and [3574]')
plt.autoscale(False)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        text = plt.text(j, i, '%.4f' % mat[i, j], ha="center", va="center", color="w")
plt.savefig('FigXX-CostMatrix.png', dpi=600)
plt.show()

# %%
acc = subsequence_cost_matrix(ts1.values.reshape(-1,1), ts2.values.reshape(-1,1))

plt.figure(figsize=(8,8))
plt.imshow(acc)
plt.yticks(range(10))
plt.xticks(range(10))
plt.title('Cumulative Cost Matrix of total_acc_x_train[7345] and [3574]')
plt.autoscale(False)
for i in range(acc.shape[0]):
    for j in range(acc.shape[1]):
        text = plt.text(j, i, '%.4f' % np.sqrt(acc[i, j]), ha="center", va="center", color="w")
plt.savefig('FigXX-AccCostMatrix.png', dpi=600)
plt.show()

# %%
path, dist = dtw_path(ts1, ts2)
print(dist)
print(path)

plt.imshow(mat)
plt.axis("off")
plt.autoscale(False)
plt.plot([j for (i, j) in path], [i for (i, j) in path], "w", linewidth=3.)
plt.savefig('FigXX-matpath.png', dpi=600)
plt.show()

# %%
path, dist = dtw_path(ts1, ts2)
print(dist)
print(path)

plt.imshow(acc)
plt.axis("off")
plt.autoscale(False)
plt.plot([j for (i, j) in path], [i for (i, j) in path], "w", linewidth=3.)
plt.savefig('FigXX-accpath.png', dpi=600)
plt.show()

# %%
path, dist = dtw_path(ts1, ts2, global_constraint="sakoe_chiba", sakoe_chiba_radius=2)
print(dist)
print(path)

plt.imshow(mat)
plt.axis("off")
plt.autoscale(False)
plt.plot([j for (i, j) in path], [i for (i, j) in path], "w-", linewidth=3.)
plt.show()

# %%
path, dist = dtw_path(ts1, ts2, global_constraint="itakura", itakura_max_slope=2.)
print(dist)
print(path)

plt.imshow(mat)
plt.axis("off")
plt.autoscale(False)
plt.plot([j for (i, j) in path], [i for (i, j) in path], "w-", linewidth=3.)
plt.show()

# %%
ts1 = data_ts.iloc[7345,:]
ts2 = data_ts.iloc[3574,:]

mat = cdist(ts1.values.reshape(-1,1), ts2.values.reshape(-1,1))
acc = subsequence_cost_matrix(ts1.values.reshape(-1,1), ts2.values.reshape(-1,1))

# %%
path, dist = dtw_path(ts1, ts2)
print(dist)
print(path)

# Optimal Path w.r.t point-to-point costs
plt.imshow(mat)
plt.axis("off")
plt.autoscale(False)
plt.plot([j for (i, j) in path], [i for (i, j) in path], "w", linewidth=3.)
plt.savefig('FigXX-CostMatrixPath.png', dpi=600)
plt.show()

# Optimal Path w.r.t cumulative costs
plt.imshow(acc)
plt.axis("off")
plt.autoscale(False)
plt.plot([j for (i, j) in path], [i for (i, j) in path], "w", linewidth=3.)
plt.show()

# %%
path, dist = dtw_path(ts1, ts2, global_constraint="sakoe_chiba", sakoe_chiba_radius=2)
print(dist)
print(path)

# Optimal Path w.r.t point-to-point costs
plt.imshow(mat)
plt.axis("off")
plt.autoscale(False)
plt.plot([j for (i, j) in path], [i for (i, j) in path], "w", linewidth=3.)
plt.savefig('FigXX-CostMatrixPathSakoe.png', dpi=600)
plt.show()

# Optimal Path w.r.t cumulative costs
plt.imshow(acc)
plt.axis("off")
plt.autoscale(False)
plt.plot([j for (i, j) in path], [i for (i, j) in path], "w", linewidth=3.)
plt.show()

# %%
path, dist = dtw_path(ts1, ts2, global_constraint="itakura", itakura_max_slope=2.)
print(dist)
print(path)

# Optimal Path w.r.t point-to-point costs
plt.imshow(mat)
plt.axis("off")
plt.autoscale(False)
plt.plot([j for (i, j) in path], [i for (i, j) in path], "w", linewidth=3.)
plt.savefig('FigXX-CostMatrixItakura.png', dpi=600)
plt.show()

# Optimal Path w.r.t cumulative costs
plt.imshow(acc)
plt.axis("off")
plt.autoscale(False)
plt.plot([j for (i, j) in path], [i for (i, j) in path], "w", linewidth=3.)
plt.show()

# %%
(dist, cost, acc, path) = dtw(ts1, ts2, distance)

yshift = .05
plt.figure(figsize=(10,2))
for (i,j) in zip(path[0],path[1]):
    col = 'r-' if i == j else 'y-'
    plt.plot([ i, j ] , [ ts1[i], ts2[j]+yshift ], col)
plt.xlim(-1,max(len(ts1),len(ts2)))
plt.plot(ts2+yshift, label = 'total_acc_x_train[7345]')
plt.plot(ts1, label = 'total_acc_x_train[3574]')
plt.title('Dynamic Time Warping Alignment')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.savefig('FigXX-DTWAlignment.png', dpi=600, bbox_inches = 'tight')
plt.show()

# %%
d, aa, mat, path = dtw_band(ts1,ts2,distance,band=40)

plt.imshow(mat.T)
plt.axis("off")
plt.plot(path[0], path[1], 'w')
plt.plot(path[0], path[0], 'r-')
plt.title('Optimal Path with Band=40')
plt.xlim(-0.5,mat.shape[0]-0.5)
plt.ylim(-0.5,mat.shape[1]-0.5)
plt.savefig('FigXX-PathBand40.png', dpi=600)
plt.show()

# %%
d, aa, mat, path = dtw_band(ts1,ts2,distance,band=20)

plt.imshow(mat.T)
plt.axis("off")
plt.plot(path[0], path[1], 'w')
plt.plot(path[0], path[0], 'r-')
plt.title('Optimal Path with Band=20')
plt.xlim(-0.5,mat.shape[0]-0.5)
plt.ylim(-0.5,mat.shape[1]-0.5)
plt.savefig('FigXX-PathBand20.png', dpi=600)
plt.show()

# %%
d, aa, mat, path = dtw_band(ts1,ts2,distance,band=10)

plt.imshow(mat.T)
plt.axis("off")
plt.plot(path[0], path[1], 'w')
plt.plot(path[0], path[0], 'r-')
plt.title('Optimal Path with Band=10')
plt.xlim(-0.5,mat.shape[0]-0.5)
plt.ylim(-0.5,mat.shape[1]-0.5)
plt.savefig('FigXX-PathBand10.png', dpi=600)
plt.show()

# %%
b = 20
d, aa, mat, path = dtw_band(ts1,ts2,distance,band=b)

yshift = .05
plt.figure(figsize=(10,2))
for (i,j) in zip(path[0],path[1]):
    col = 'r-' if i == j else 'y-'
    plt.plot([ i, j ] , [ ts1[i], ts2[j]+yshift ], col)
plt.xlim(-1,max(len(ts1),len(ts2)))
plt.plot(ts2+yshift)
plt.plot(ts1)
plt.title('DTW Alignment with Band = {}'.format(b))
plt.savefig('FigXX-DTWAlignment20.png', dpi=600, bbox_inches = 'tight')
plt.show()

# %%
b = 2
d, aa, mat, path = dtw_band(ts1,ts2,distance,band=b)

yshift = .05
plt.figure(figsize=(10,2))
for (i,j) in zip(path[0],path[1]):
    col = 'r-' if i == j else 'y-'
    plt.plot([ i, j ] , [ ts1[i], ts2[j]+yshift ], col)
plt.xlim(-1,max(len(ts1),len(ts2)))
plt.plot(ts2+yshift)
plt.plot(ts1)
plt.title('DTW Alignment with Band = {}'.format(b))
plt.savefig('FigXX-DTWAlignment2.png', dpi=600, bbox_inches = 'tight')
plt.show()

# %%
b = np.inf
d, aa, mat, path = dtw_band(ts1,ts2,distance,band=b)

yshift = .05
plt.figure(figsize=(10,2))
for (i,j) in zip(path[0],path[1]):
    col = 'r-' if i == j else 'y-'
    plt.plot([ i, j ] , [ ts1[i], ts2[j]+yshift ], col)
plt.xlim(-1,max(len(ts1),len(ts2)))
plt.plot(ts2+yshift)
plt.plot(ts1)
plt.title('DTW Alignment with Band = {}'.format(b))
plt.savefig('FigXX-DTWAlignmentinf.png', dpi=600, bbox_inches = 'tight')
plt.show()

# %%
b = 0
d, aa, mat, path = dtw_band(ts1,ts2,distance,band=b)

yshift = .05
plt.figure(figsize=(10,2))
for (i,j) in zip(path[0],path[1]):
    col = 'r-' if i == j else 'y-'
    plt.plot([ i, j ] , [ ts1[i], ts2[j]+yshift ], col)
plt.xlim(-1,max(len(ts1),len(ts2)))
plt.plot(ts2+yshift)
plt.plot(ts1)
plt.title('DTW Alignment with Band = {}'.format(b))
plt.savefig('FigXX-DTWAlignment0.png', dpi=600, bbox_inches = 'tight')
plt.show()

# %%
bvals = range(0,60)
plt.figure(figsize=(5,3))
d_list = [ dtw_band(ts1,ts2,distance,band=b)[0] for b in bvals ]
best_d = dtw_band(ts1,ts2,distance,band=np.inf)[0]
plt.plot(bvals, d_list, label = 'DTW-band(b)' )
plt.plot([bvals[0],bvals[-1]],[best_d, best_d], '--', label = 'band = inf')
plt.legend(loc='upper right')
plt.xlabel('Values of b')
plt.xlim(0, 60)
plt.title('DTW-band(b) for increasing values of b')
plt.savefig('FigXX-DTWAlignmentGraph.png', dpi=600,bbox_inches = 'tight')
plt.show()

# %%
b = 15
d, aa, mat, path = dtw_band(ts1,ts2,distance,band=b)

yshift = .05
plt.figure(figsize=(10,2))
for (i,j) in zip(path[0],path[1]):
    col = 'r-' if i == j else 'y-'
    plt.plot([ i, j ] , [ ts1[i], ts2[j]+yshift ], col)
plt.xlim(-1,max(len(ts1),len(ts2)))
plt.plot(ts2+yshift)
plt.plot(ts1)
plt.title('DTW Alignment with Band = {}'.format(b))
plt.show()

# %%
d, aa, mat, path = dtw_parallel(ts1,ts2,distance,coeff=.5)

plt.imshow(mat.T)
plt.axis("off")
plt.plot(path[0], path[1], 'w')
plt.plot(path[0], path[0], 'r-')
plt.title('Optimal Path with Parallel coefficient=.5')
plt.xlim(-0.5,mat.shape[0]-0.5)
plt.ylim(-0.5,mat.shape[1]-0.5)
plt.savefig('FigXX-ParallelpathCoeff05.png', dpi=600)
plt.show()

# %%
d, aa, mat, path = dtw_parallel(ts1,ts2,distance,coeff=1.0)

plt.imshow(mat.T)
plt.axis("off")
plt.plot(path[0], path[1], 'w')
plt.plot(path[0], path[0], 'r-')
plt.xlim(-0.5,mat.shape[0]-0.5)
plt.ylim(-0.5,mat.shape[1]-0.5)
plt.title('Optimal Path with Parallel coefficient=1')
plt.savefig('FigXX-ParallelpathCoeff1.png', dpi=600)
plt.show()

# %%
d, aa, mat, path = dtw_parallel(ts1,ts2,distance,coeff=5.0)

plt.imshow(mat.T)
plt.axis("off")
plt.plot(path[0], path[1], 'w')
plt.plot(path[0], path[0], 'r-')
plt.xlim(-0.5,mat.shape[0]-0.5)
plt.ylim(-0.5,mat.shape[1]-0.5)
plt.title('Optimal Path with Parallel coefficient=5')
plt.savefig('FigXX-ParallelpathCoeff5.png', dpi=600)
plt.show()

# %%
c = .5
d, aa, mat, path = dtw_parallel(ts1,ts2,distance,coeff=c)

yshift = .05
plt.figure(figsize=(10,2))
for (i,j) in zip(path[0],path[1]):
    col = 'r-' if i == j else 'y-'
    plt.plot([ i, j ] , [ ts1[i], ts2[j]+yshift ], col)
plt.xlim(-1,max(len(ts1),len(ts2)))
plt.plot(ts2+yshift)
plt.plot(ts1)
plt.title('DTW Alignment for Parallel with Coefficient = {}'.format(c))
plt.savefig('FigXX-ParallelCoeff05.png', dpi=600)
plt.show()

# %%
c = 1
d, aa, mat, path = dtw_parallel(ts1,ts2,distance,coeff=c)

yshift = .05
plt.figure(figsize=(10,2))
for (i,j) in zip(path[0],path[1]):
    col = 'r-' if i == j else 'y-'
    plt.plot([ i, j ] , [ ts1[i], ts2[j]+yshift ], col)
plt.xlim(-1,max(len(ts1),len(ts2)))
plt.plot(ts2+yshift)
plt.plot(ts1)
plt.title('DTW Alignment for Parallel with Coefficient = {}'.format(c))
plt.savefig('FigXX-ParallelCoeff1.png', dpi=600)
plt.show()

# %%
c = 5
d, aa, mat, path = dtw_parallel(ts1,ts2,distance,coeff=c)

yshift = .05
plt.figure(figsize=(10,2))
for (i,j) in zip(path[0],path[1]):
    col = 'r-' if i == j else 'y-'
    plt.plot([ i, j ] , [ ts1[i], ts2[j]+yshift ], col)
plt.xlim(-1,max(len(ts1),len(ts2)))
plt.plot(ts2+yshift)
plt.plot(ts1)
plt.title('DTW Alignment for Parallel with Coefficient = {}'.format(c))
plt.savefig('FigXX-ParallelCoeff5.png', dpi=600)
plt.show()

# %%
c = np.inf
d, aa, mat, path = dtw_parallel(ts1,ts2,distance,coeff=c)

yshift = .05
plt.figure(figsize=(10,2))
for (i,j) in zip(path[0],path[1]):
    col = 'r-' if i == j else 'y-'
    plt.plot([ i, j ] , [ ts1[i], ts2[j]+yshift ], col)
plt.xlim(-1,max(len(ts1),len(ts2)))
plt.plot(ts2+yshift)
plt.plot(ts1)
plt.title('DTW Alignment for Parallel with Coefficient = {}'.format(c))
plt.savefig('FigXX-ParallelCoeffing.png', dpi=600)
plt.show()

# %%
c = 0
d, aa, mat, path = dtw_parallel(ts1,ts2,distance,coeff=c)

yshift = .05
plt.figure(figsize=(10,2))
for (i,j) in zip(path[0],path[1]):
    col = 'r-' if i == j else 'y-'
    plt.plot([ i, j ] , [ ts1[i], ts2[j]+yshift ], col)
plt.xlim(-1,max(len(ts1),len(ts2)))
plt.plot(ts2+yshift)
plt.plot(ts1)
plt.title('DTW Alignment for Parallel with Coefficient = {}'.format(c))
plt.savefig('FigXX-ParallelCoeff0.png', dpi=600)
plt.show()

# %%
cvals = np.linspace(0.1,20.0,20)
plt.figure(figsize=(5,3))
d_list = [ dtw_parallel(ts1,ts2,distance,coeff=c)[0] for c in cvals ]
best_d = dtw_parallel(ts1,ts2,distance)[0]
plt.plot(cvals,d_list, label = 'DTW-Parallel(c)')
plt.plot([cvals[0],cvals[-1]],[best_d, best_d], '--', label='c = inf')
plt.legend(loc='upper right')
plt.xlabel('Values of c')
plt.xlim(0, 20)
plt.title('DTW-parallel(c) for increasing values of c')
plt.savefig('FigXX-DTWAlignmentParallelGraph.png', dpi=600,bbox_inches = 'tight')
plt.show()

# %%
c = 2.5
d, aa, mat, path = dtw_parallel(ts1,ts2,distance,coeff=c)

yshift = .05
plt.figure(figsize=(10,2))
for (i,j) in zip(path[0],path[1]):
    col = 'r-' if i == j else 'y-'
    plt.plot([ i, j ] , [ ts1[i], ts2[j]+yshift ], col)
plt.xlim(-1,max(len(ts1),len(ts2)))
plt.plot(ts2+yshift)
plt.plot(ts1)
plt.title('DTW Alignment for Parallel with Coefficient = {}'.format(c))
# plt.savefig('FigXX-ParallelCoeff15.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">8.x TS Approximation</font>**

# %%
data_ts = (data_ts6-data_ts6.min())/(data_ts6.max()-data_ts6.min())
data_test_ts = (data_test_ts6-data_test_ts6.min())/(data_test_ts6.max()-data_test_ts6.min())

# %%
ts = data_ts.iloc[6927,:]

scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
ts = scaler.fit_transform(ts.values.reshape(1,-1))

plt.figure(figsize=(10,2))
plt.title("Raw total_acc_x_train[6927]")
plt.xlim([0, 127])
plt.plot(ts[0].ravel())
plt.savefig('FigXX-ApproxRaw.png', dpi=600)
plt.show()

# %%
def dft_inverse_trasform(X_dft, n_coefs, n_timestamps):
    # Compute the inverse transformation
    n_samples = X_dft.shape[0]
    if n_coefs % 2 == 0:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs, 2)
        X_dft_new = np.c_[
            X_dft[:, :1],
            X_dft[:, real_idx] + 1j * np.c_[X_dft[:, imag_idx],
                                            np.zeros((n_samples, ))]
        ]
    else:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs + 1, 2)
        X_dft_new = np.c_[
            X_dft[:, :1],
            X_dft[:, real_idx] + 1j * X_dft[:, imag_idx]
        ]
    X_irfft = np.fft.irfft(X_dft_new, n_timestamps)
    return X_irfft

# %%
from pyts.approximation import DiscreteFourierTransform
ts = data_ts.iloc[6927,:]
n_coefs = 16

dft = DiscreteFourierTransform(n_coefs=n_coefs)
ts_dft = dft.fit_transform(ts.values.reshape(1, -1))
ts_dft.shape

# %%
ts_dft_inv = dft_inverse_trasform(ts_dft, n_coefs=n_coefs, n_timestamps=len(ts.values))

plt.figure(figsize=(10,2))
plt.title("Discrete Fourier Transform of total_acc_x_train[6927]")
plt.xlim([0, 127])
plt.plot(ts.ravel(), alpha=0.6, label = 'Raw TS')
plt.plot(ts_dft_inv.ravel(),label = 'DFT')
plt.legend(loc='upper right')
plt.savefig('FigXX-ApproxDFT.png', dpi=600)
plt.show()

# %%
ts = data_ts.iloc[6927,:]
ts = scaler.fit_transform(ts.values.reshape(1,-1))

# PAA transform (and inverse transform) of the data
n_paa_segments = 10
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
ts_paa = paa.fit_transform(ts)
paa_dataset_inv = paa.inverse_transform(ts_paa)

plt.figure(figsize=(10,2))
plt.plot(ts[0].ravel(), alpha=0.6, label="Raw TS")
plt.plot(paa_dataset_inv[0].ravel(), label="PAA")
plt.legend(loc='upper right')
plt.title("PAA approximation of total_acc_x_train[6927]")
plt.xlim([0, 127])
plt.savefig('FigXX-ApproxPAA.png', dpi=600)
plt.show()

# %%
# SAX transform
n_sax_symbols = 8
sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
ts_sax = sax.fit_transform(ts)
sax_dataset_inv = sax.inverse_transform(ts_sax)

plt.figure(figsize=(10,2))
plt.plot(ts[0].ravel(), alpha=0.6, label="Raw TS")
plt.plot(sax_dataset_inv[0].ravel(), label="SAX")
plt.legend(loc='upper right')
plt.title("SAX approximation of total_acc_x_train[6927], %d symbols" % n_sax_symbols)
plt.xlim([0, 127])
plt.savefig('FigXX-ApproxSAX.png', dpi=600)
plt.show()

# %%
# 1d-SAX transform
n_sax_symbols_avg = 8
n_sax_symbols_slope = 4
one_d_sax = OneD_SymbolicAggregateApproximation(
    n_segments=n_paa_segments,
    alphabet_size_avg=n_sax_symbols_avg,
    alphabet_size_slope=n_sax_symbols_slope)

ts_sax1d = one_d_sax.fit_transform(ts)
one_d_sax_dataset_inv = one_d_sax.inverse_transform(ts_sax1d)

plt.figure(figsize=(10,2))
plt.plot(ts[0].ravel(), alpha=0.6, label="Raw TS")
plt.plot(one_d_sax_dataset_inv[0].ravel(), label="1-d SAX")
plt.legend(loc='upper right')
plt.title("1d-SAX approximation of total_acc_x_train[6927], %d symbols" "(%dx%d)" % (n_sax_symbols_avg * n_sax_symbols_slope, n_sax_symbols_avg, n_sax_symbols_slope))
plt.xlim([0, 127])
plt.savefig('FigXX-Approx1dSAX.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">8.x TS Clustering</font>**

# %%
# from tslearn.generators import random_walks
# Xtoy = random_walks(n_ts=50, sz=32, d=1)
# print(Xtoy.shape)

# plt.plot(np.squeeze(Xtoy).T)
# plt.show()

# km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5, random_state=0)
# km.fit(Xtoy)

# print(km.cluster_centers_.shape)
# plt.plot(np.squeeze(km.cluster_centers_).T)
# plt.show()

# for i in range(3):
#     plt.plot(np.mean(Xtoy[np.where(km.labels_ == i)[0]], axis=0))
# plt.show()

# %%
data_ts = data_ts3
data_test = data_test_ts3

ts = data_ts

plt.figure(figsize=(8,4))
plt.plot(ts.T)
plt.title("Time Series of body_gyro_y_train")
plt.xlim([0, 127])
plt.savefig('FigXX-ClusterTS.png', dpi=600)
plt.show()

data_ts.shape

# %% [markdown]
# ### **<font color="#34eb89">8.x.x Shape-Based Clustering</font>**

# %%
km = TimeSeriesKMeans(n_clusters=4, metric="euclidean", max_iter=100, random_state=0)
km.fit(ts)

print(km.cluster_centers_.shape)
plt.figure(figsize=(8,4))
plt.plot(np.squeeze(km.cluster_centers_).T)
plt.title("Shape-based Clusters found by KMeans")
plt.xlim([0, 127])
plt.savefig('FigXX-ClusterShape.png', dpi=600)
plt.show()

# %%
# km_dtw = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5, random_state=0)
# km_dtw.fit(ts)

# plt.plot(np.squeeze(km_dtw.cluster_centers_).T)
# plt.show()

# %%
# km = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5, random_state=0)
# km.fit(ts.T)

# plt.plot(np.squeeze(km.cluster_centers_).T)
# plt.show()

# %% [markdown]
# ### **<font color="#34eb89">8.x.x Feature-Based Clustering</font>**

# %%
ts = data_ts.T
F = []
for i in ts:
  F.append(list(calculate_features(ts[i]).values()))

np.shape(F)

# %%
kmeans = KMeans(n_clusters=3)
kmeans.fit(F)
kmeans.labels_

# %%
plt.figure(figsize=(10,3))
for i in range(3):
    plt.plot(np.mean(ts[np.where(kmeans.labels_ == i)[0]].T),label=i)
plt.title("Feature-based Clusters found by KMeans")
plt.legend(loc="best")
plt.xlim([0, 127])
plt.savefig('FigXX-ClusterFeature.png', dpi=600)
plt.show()

# %% [markdown]
# ### **<font color="#34eb89">8.x.x Compression-Based Clustering</font>**

# %%
# import zlib
# import string

# %%
# def cdm_dist(x, y):
#     x_str = (' '.join([str(v) for v in x.ravel()])).encode('utf-8')
#     y_str = (' '.join([str(v) for v in y.ravel()])).encode('utf-8')
#     return len(zlib.compress(x_str + y_str)) / (len(zlib.compress(x_str)) + len(zlib.compress(y_str)))

# %%
# M = pairwise_distances(ts, metric=cdm_dist)
# plt.plot(sorted(M.ravel()))
# plt.show()

# %% [markdown]
# ### **<font color="#34eb89">8.x.x Approximate Clustering</font>**

# %%
data_ts.shape

# %%
n_paa_segments = 10
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
ts_paa = paa.fit_transform(ts)

plt.figure(figsize=(8,4))
plt.plot(ts_paa.reshape(ts_paa.shape[1], ts_paa.shape[0]))
plt.xlim([0, 9])
plt.title("Timeseries Approximation of body_gyro_y_train with {} segments".format(n_paa_segments))
plt.savefig('FigXX-ClusterApproxTS.png', dpi=600)
plt.show()

ts_paa.shape

# %%
km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5, random_state=0)
km.fit(ts_paa)

clusters = ['0', '1', '2']
plt.figure(figsize=(8,4))
plt.plot(km.cluster_centers_.reshape(ts_paa.shape[1], 3))
plt.title("Cluster Centers in approximated body_gyro_y_train")
plt.legend(labels=clusters, loc='best')
plt.xlim([0, 9])
plt.show()

# %%
plt.figure(figsize=(10,3))
for i in range(3):
    plt.plot(np.mean(ts_paa[np.where(km.labels_ == i)[0]] , axis=0), label = i)
plt.title("Clusters found by KMeans in approximated body_gyro_y_train")
plt.xlim([0, 9])
plt.legend(loc="best")
plt.savefig('FigXX-ClusterApprox.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">8.x Shapelet Discovery</font>**

# %%
data_ts = data_ts3
data_test = data_test_ts3

# %%
n_ts, ts_sz = data_ts.shape
n_classes = len(set(y_train))

shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts, ts_sz=ts_sz, n_classes=n_classes, l=0.1, r=1)

print('Number of time series:', n_ts)
print('Time series size:', ts_sz)
print('n_classes:', n_classes)
print('shapelet_sizes:', shapelet_sizes)

# %%
shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                        optimizer="sgd",
                        weight_regularizer=.01,
                        max_iter=200,
                        verbose=0,
                        random_state=0)

shp_clf.fit(data_ts, y_train)

# %%
predicted_locations = shp_clf.locate(data_ts)

ts_id = 4637
plt.figure(figsize=(10,2))
n_shapelets = sum(shapelet_sizes.values())
plt.title("Example locations of shapelet matches in body_gyro_y_train[4637]")
plt.plot(data_ts.iloc[ts_id,:])
plt.xlim([0,127])
for idx_shp, shp in enumerate(shp_clf.shapelets_):
    t0 = predicted_locations[ts_id, idx_shp]
    plt.plot(np.arange(t0, t0 + len(shp)), shp, linewidth=2, label = 'Shapelet {}'.format(idx_shp))
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.savefig('FigXX-Shapelets.png', dpi=600,bbox_inches = 'tight')
plt.show()

# %%
pred_labels = shp_clf.predict(X_test_ts)
print("Correct classification rate:", accuracy_score(y_test, pred_labels))

shapelets = ['Shapelet 0', 'Shapelet 1', 'Shapelet 2', 'Shapelet 3', 'Shapelet 4', 'Shapelet 5']
plt.figure(figsize=(10,3))
for i, sz in enumerate(shapelet_sizes.keys()):
  plt.title("%d shapelets of size %d" % (shapelet_sizes[sz], sz))
  for shp in shp_clf.shapelets_:
    if ts_size(shp) == sz:
      plt.plot(shp.ravel())
  plt.xlim([0, max(shapelet_sizes.keys()) - 1])
  plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5, labels=shapelets)
  plt.show()

  plt.figure(figsize=(8,4))
  plt.plot(np.arange(1, shp_clf.n_iter_ + 1), shp_clf.history_["loss"])
  plt.title("Evolution of cross-entropy loss during training")
  plt.xlabel("Epochs")
  plt.xlim([0,200])
  plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">8.x Motif Discovery</font>**

# %%
!pip install matrixprofile-ts
from matrixprofile import *

# %%
# data_ts = data_ts6
# data_test = data_test_ts6

# X_train_ts = data_ts.values
# X_test_ts = data_test_ts.values

# %%
data_ts = (data_ts6-data_ts6.min())/(data_ts6.max()-data_ts6.min())
data_test_ts = (data_test_ts6-data_test_ts6.min())/(data_test_ts6.max()-data_test_ts6.min())

# %%
ts = data_ts.iloc[3245,:]

plt.figure(figsize=(10,2))
plt.plot(ts)
plt.xlim([0,127])
plt.show()

# %%
w =12
mp, mpi= matrixProfile.stomp(ts.values, w)

plt.figure(figsize=(10,2))
plt.plot(mp)
plt.xlim([0,120])
plt.show()

# %%
ts_log_mov_diff = pd.Series(np.log(ts) - np.log(ts).rolling(w, center=False).mean(), index=ts.index)

plt.figure(figsize=(10,2))
plt.plot(ts_log_mov_diff)
plt.xlim([0,127])
plt.show()

# %%
w = 3
mp, mpi = matrixProfile.stomp(ts.values, w)

plt.figure(figsize=(10,2))
plt.plot(mp)
plt.xlim([0,127])
plt.show()

# %%
ts = data_ts6.iloc[3245,:]
w = 3

mo, mod  = motifs.motifs(ts.values, (mp, mpi), max_motifs=5)

print(mo)
print(mod)

# %%
plt.figure(figsize=(10,2))
plt.plot(ts, alpha=.7)
plt.xlim([0,127])
colors = ['seagreen', 'darkorange', 'blueviolet', 'orchid', 'maroon'][:len(mo)]
for m, d, c in zip(mo, mod, colors):
    plt.title('Motifs found in total_acc_x_train[3245]')
    for i in m:
        m_shape = ts.values[i:i+w]
        plt.plot(range(i,i+w), m_shape, lw=3)
plt.savefig('FigXX-Motifs.png', dpi=600)
plt.show()

# %%
for m, d, c in zip(mo, mod, colors):
    plt.figure(figsize=(10,2))
    for i in m:
        m_shape = ts.values[i:i+w]
        plt.plot(range(i,i+w), m_shape, lw=2)
    plt.savefig('FigXX-'+str(i)+'Motif.png', dpi=600)
    plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">8.x Anomaly Discovery</font>**

# %%
from matrixprofile.discords import discords

# %%
anoms = discords(mp, ex_zone=3, k=5)
anoms

# %%
plt.figure(figsize=(10,2))
plt.plot(ts, alpha=.7)
plt.xlim([0,127])
plt.title('Anomalies found in total_acc_x_train[3245]')
colors = ['seagreen', 'darkorange', 'blueviolet', 'orchid', 'maroon'][:len(mo)]
for a, c in zip(anoms, colors):
    a_shape = ts.values[a:a+w]
    plt.plot(range(a, a+w), a_shape, color=c, lw=3)
plt.savefig('FigXX-TSAnomalies.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">8.x TS Classification</font>**

# %%
data_ts = (data_ts6-data_ts6.min())/(data_ts6.max()-data_ts6.min())
data_test_ts = (data_test_ts6-data_test_ts6.min())/(data_test_ts6.max()-data_test_ts6.min())

X_train_ts = data_ts.values
X_test_ts = data_test_ts.values

# %% [markdown]
# ### **<font color="#34eb89">8.x.x TimeSeries Classification</font>**

# %%
clf = KNeighborsClassifier()
clf.fit(X_train_ts, y_train)

y_pred = clf.predict(X_test_ts)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

# %%
clf = DecisionTreeClassifier()
clf.fit(X_train_ts, y_train)

y_pred = clf.predict(X_test_ts)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

# %%
from pyts.classification import KNeighborsClassifier

clf = KNeighborsClassifier(metric='dtw_sakoechiba')
clf.fit(X_train_ts, y_train)

y_pred = clf.predict(X_test_ts)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### **<font color="#34eb89">8.x.x Shapelet-Based Classification</font>**

# %%
# st = ShapeletTransform(window_sizes=[4, 12], random_state=42, sort=True)

# X_new = st.fit_transform(X_train_ts, y_train)
# X_test_new = st.transform(X_test_ts)

# %%
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(max_depth=8, random_state=42)
# clf.fit(X_new, y_train)

# y_pred = clf.predict(X_test_new)

# print('Accuracy %s' % accuracy_score(y_test, y_pred))
# print('F1-score %s' % f1_score(y_test, y_pred, average=None))
# print(classification_report(y_test, y_pred))

# %%
n_ts, ts_sz = data_ts.shape
n_classes = len(set(y_train))

shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts, ts_sz=ts_sz, n_classes=n_classes, l=0.1, r=1)

print('Number of time series:', n_ts)
print('Time series size:', ts_sz)
print('n_classes:', n_classes)
print('shapelet_sizes:', shapelet_sizes)

# %%
shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                        optimizer="sgd",
                        weight_regularizer=.01,
                        max_iter=200,
                        verbose=0,
                        random_state=0)

# %%
shp_clf.fit(X_train_ts, y_train)
y_pred = shp_clf.predict(X_test_ts)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

# %%
X_train2_ts = shp_clf.transform(X_train_ts)
X_test2_ts = shp_clf.transform(X_test_ts)

X_train2_ts.shape

# %%
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train2_ts, y_train)

y_pred = clf.predict(X_test2_ts)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

# %%
clf = DecisionTreeClassifier(max_depth=8, random_state=42)
clf.fit(X_train2_ts, y_train)

y_pred = clf.predict(X_test2_ts)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### **<font color="#34eb89">8.x.x Feature-Based Classification</font>**

# %%
train = data_ts.T
test = data_test_ts.T

X_trainF = []
for i in train:
  X_trainF.append(list(calculate_features(train[i]).values()))

X_testF = []
for i in test:
  X_testF.append(list(calculate_features(test[i]).values()))

np.shape(X_trainF)

# %%
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_trainF, y_train)

y_pred = clf.predict(X_testF)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

# %%
clf = DecisionTreeClassifier(max_depth=8, random_state=42)
clf.fit(X_trainF, y_train)

y_pred = clf.predict(X_testF)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### **<font color="#34eb89">8.x.x CNN Classification</font>**

# %%
def build_simple_cnn(n_timesteps, n_outputs):
    model = Sequential()

    model.add(Conv1D(filters=16, kernel_size=8, activation='relu', input_shape=(n_timesteps, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.3))

    model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.3))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.3))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# %%
X_train_cnn = X_train_ts.reshape((X_train_ts.shape[0], X_train_ts.shape[1], 1))
X_test_cnn = X_test_ts.reshape((X_test_ts.shape[0], X_test_ts.shape[1], 1))

X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(X_train_cnn, y_train, test_size=0.2, stratify=y_train)

n_timesteps, n_outputs, n_features = X_train_cnn.shape[1], len(np.unique(y_train_cnn)), 1
print("TIMESTEPS: ", n_timesteps)
print("N. LABELS: ", n_outputs)

# %%
cnn = build_simple_cnn(n_timesteps, n_outputs)
cnn.summary()

# %%
rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
mc = ModelCheckpoint('best_model_cnn.h5', monitor='val_loss', save_best_only=True)

callbacks = [rlr, mc]

batch_size = 16
mini_batch_size = int(min(X_train_cnn.shape[0]/10, batch_size))

history_cnn = cnn.fit(X_train_cnn, y_train_cnn, epochs=5, batch_size=mini_batch_size, callbacks=callbacks,
                      validation_data=(X_val_cnn, y_val_cnn)).history

# %%
y_pred = np.argmax(cnn.predict(X_test_cnn), axis=1)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

cnn.evaluate(X_test_cnn, y_test)

# %% [markdown]
# ### **<font color="#34eb89">8.x.x LSTM Classification</font>**

# %%
def build_lstm(n_timesteps, n_outputs):
    model = Sequential()
    model.add(LSTM(256, input_shape=(n_timesteps, 1)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_outputs, activation='sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# %%
lstm = build_lstm(n_timesteps, n_outputs)
lstm.summary()

# %%
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0)
mc = ModelCheckpoint('best_model_cnn.h5', monitor='val_loss', save_best_only=True)

callbacks = [rlr, mc]

batch_size = 16
mini_batch_size = int(min(X_train_cnn.shape[0]/10, batch_size))

history_lstm = cnn.fit(X_train_cnn, y_train_cnn, epochs=100, batch_size=mini_batch_size, callbacks=callbacks,
                       validation_data=(X_val_cnn, y_val_cnn)).history

# %%
y_pred = np.argmax(lstm.predict(X_test_cnn), axis=1)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

lstm.evaluate(X_test_cnn, y_test)

# %% [markdown]
# ### **<font color="#34eb89">8.x.x Advanced Classification - Crossvalidation</font>**

# %%
# Lista dei modeli a utilizzare
model_knn = KNeighborsClassifier()
model_gaussnb = GaussianNB()
model_svc = SVC()
model_etc = ExtraTreesClassifier()
model_bag = BaggingClassifier()
model_xgb = XGBClassifier()

# Dizionario a percorrere per chiamare ogni modello
models = {
    'K Neighbors': model_knn,
    'Gaussian Naive Bayes': model_gaussnb,
    'SVM': model_svc,
    'ExtraTrees': model_etc,
    'Bagging': model_bag,
    'XGBoost': model_xgb
    }

# Lista vuota dove mettere i valori accuracy di ogni metodo con cross-validation
validation_scores = {}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# %%
# Prima "vista" del performance di ogni modello, inizializzandoli senza alcun parametri,
for name, model in models.items():
    print(f"{name}'s KFold starting")
    score = cross_val_score(model, X_train_ts, y_train, scoring='accuracy', cv=kf, n_jobs=-1, verbose=0).mean()
    print(f"{name}'s cross validation score: {score:.6f}\n")
    validation_scores[name] = score

# %%
plt.figure(figsize=(8,4), tight_layout=True)
colors = sns.color_palette('Set3')

plt.barh(list(validation_scores.keys()), list(validation_scores.values()), color=colors[1:16])
plt.title("Cross-validation Scores")
plt.xlabel('Performance')
# plt.savefig('FigXXCrossValidationScore.png', dpi=600)
plt.show()

# %%
# # Lista dei modeli a utilizzare con hypertuning
# model_dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_leaf_nodes=50,
#                        min_samples_leaf=2, random_state=0)
# model_knn = KNeighborsClassifier(algorithm='brute', n_neighbors=1, weights='distance')
# model_gaussnb = GaussianNB(var_smoothing=0.0001)
# model_bernb = BernoulliNB(alpha=0)
# model_logreg = LogisticRegression(penalty='l1',  random_state=0, solver='liblinear')
# model_linsvc = LinearSVC(C=1, dual=False, fit_intercept=False, random_state=0)
# model_svc = SVC(C=5, random_state=0)
# model_mlp = MLPClassifier(alpha=0.05, hidden_layer_sizes=(555,), learning_rate='adaptive',
#               max_iter=10000, solver='sgd')
# model_rfc = RandomForestClassifier(criterion='entropy', n_estimators=300, oob_score=True, random_state=0)
# model_etc = ExtraTreesClassifier(n_estimators=300, random_state=0)
# model_bag = BaggingClassifier(max_samples=0.5, n_estimators=300, random_state=0)
# model_gbc = GradientBoostingClassifier(n_estimators=300, random_state=0)
# model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
#               gamma=0, gpu_id=-1, importance_type=None, predictor='auto',
#               interaction_constraints='', learning_rate=0.25, max_delta_step=0,
#               max_depth=2, min_child_weight=1, monotone_constraints='()',
#               n_estimators=300, n_jobs=8, num_parallel_tree=1, objective='multi:softprob',
#               random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
#               seed=0, subsample=1, tree_method='exact', validate_parameters=1,
#               verbosity=None)

# # Dizionario a percorrere per chiamare ogni modello
# models = {
#     'Gaussian Naive Bayes': model_gaussnb,
#     'Bernoulli Naive Bayes': model_bernb,
#     'MLP': model_mlp,
#     'Random Forest': model_rfc,
#     'ExtraTrees': model_etc,
#     'Bagging': model_bag,
#     'GradientBoost': model_gbc,
#     'XGBoost': model_xgb,
#     'SVC': model_svc,
#     'Linear SVC': model_linsvc,
#     'Logistic Regression': model_logreg,
#     'Decision Tree': model_dt,
#     'K Neighbors': model_knn
#     }

# # Lista vuota dove mettere i valori accuracy di ogni metodo con cross-validation
# tuned_validation_scores = {}

# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# %%
# # Crossvalidation con parametri tunnati
# import math
# for name, model in models.items():
#     print(f"{name}'s KFold starting")
#     score = cross_val_score(model, X_train_ts, y_train, scoring='accuracy', cv=kf, n_jobs=-1, verbose=0).mean()
#     prev_score = round(validation_scores[name],6)
#     print(f"{name}'s cross validation score before tunning: ", prev_score)
#     print(f"{name}'s cross validation score after tunning: {score:.6f}")
#     tuned_validation_scores[name] = score
#     print("Improvement over base model: ", round(((score-prev_score)/prev_score)*100,4), "%\n")

# %%
# plt.figure(figsize=(8,4), tight_layout=True)
# colors = sns.color_palette('Set3')

# plt.barh(list(tuned_validation_scores.keys()), list(tuned_validation_scores.values()), color=colors)
# plt.title("Cross-validation Scores after tuning")
# plt.xlabel('Performance')
# plt.savefig('FigXX-TunedCrossValidationScore.png', dpi=600)
# plt.show()

# %% [markdown]
# ### **<font color="#34eb89">8.x.x Advanced Classification - RandomSearch</font>**

# %%
knn_param_grid = {
    #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
    'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21,23], #default: 5
    'weights': ['uniform', 'distance'], #default = ‘uniform’
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

gaussnb_param_grid = {
    'priors': [None],
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    }

svc_param_grid = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
    #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [.1,.5,1], #default=1.0
    'gamma': [.25, .5, 1.0], #edfault: auto
    'decision_function_shape': ['ovo', 'ovr'], #default:ovr
    'random_state': [0]
    }

etc_param_grid = {
    #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
    'n_estimators': [10, 50, 100, 300], #default=10
    'criterion': ['gini', 'entropy'], #default=”gini”
    'max_depth': [2, 4, 6, 8, 10, None], #default=None
    'random_state': [0]
    }

bag_param_grid = {
    #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
    'n_estimators': [10, 50, 100, 300], #default=10
    'max_samples': [.5, 1], #default=1.0
    'random_state': [0]
    }

xgb_param_grid = {
    #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
    'learning_rate': [.01, .03, .05, .1, .25], #default: .3
    'max_depth': [2,4,6,8,10], #default 2
    'n_estimators': [10, 50, 100, 300],
    'seed': [0]
    }

models_params = {
    'K Neighbors': [model_knn, knn_param_grid],
    'Gaussian Naive Bayes': [model_gaussnb, gaussnb_param_grid],
    'SVM': [model_svc, svc_param_grid],
    'ExtraTrees': [model_etc, etc_param_grid],
    'Bagging': [model_bag, bag_param_grid],
    'XGBoost': [model_xgb, xgb_param_grid]
    }

# %%
# random_models = {}
# random_validation_scores = {}

# for name, [model, param] in models_params.items():
#     print(f'{name} Random search starting')
#     search = RandomizedSearchCV(estimator = model,
#                                 param_distributions = param,
#                                 n_iter = 10,
#                                 cv = kf,
#                                 verbose=2,
#                                 random_state=42,
#                                 n_jobs = -1).fit(X_train_ts, y_train)
#     random_models[name] = search.best_estimator_
#     random_validation_scores[name] = search.best_score_
#     print(f'Best score: {search.best_score_}')
#     print("Best parameters: ", random_models[name], '\n')

# %%
# final_models = {}
# final_validation_scores = {}

# for name, [model, param] in models_params.items():
#     print(f'{name} Grid search starting')
#     search = GridSearchCV(model,
#                           param,
#                           cv=kf,
#                           n_jobs=-1,
#                           verbose=1,
#                           scoring='accuracy').fit(X_train_ts, y_train)
#     final_models[name] = search.best_estimator_
#     final_validation_scores[name] = search.best_score_
#     print(f'Best score: {search.best_score_}')
#     print("Best parameters: ", final_models[name], '\n')

# %% [markdown]
# ### **<font color="#34eb89">8.x.x Advanced Classification - ROC</font>**

# %%
# # Lista dei modeli a utilizzare
# gauss_nb = OneVsRestClassifier(GaussianNB(var_smoothing=0.0001))
# etc = OneVsRestClassifier(ExtraTreesClassifier(n_estimators=300, random_state=0))
# bag = OneVsRestClassifier(BaggingClassifier(max_samples=0.5, n_estimators=300, random_state=0))
# xgb = OneVsRestClassifier(XGBClassifier(XGBClassifier(base_score=0.5, booster='gbtree', learning_rate=0.25))
# svc = OneVsRestClassifier(SVC(C=5, random_state=0))
# knn = OneVsRestClassifier(KNeighborsClassifier(algorithm='brute', n_neighbors=1, weights='distance'))

# # Dizionario a percorrere per chiamare ogni modello
# models = {
#     'Gaussian Naive Bayes': gauss_nb,
#     'ExtraTrees': etc,
#     'Bagging': bag,
#     'XGBoost': xgb,
#     'SVC': svc,
#     'K Nearest Neighbor': knn
#     }

# %%
# # FARE ATTENZIONE A NON ESEGUIRLA DUE VOLTE SE NON SI VUOLE PERDERE UN SACCO DI TEMPO
# for name, model in models.items():
#     model.fit(X_train_ts, y_train_bin)
#     print(f"{name} fitted")

# %%
# y_score1 = gauss_nb.predict_proba(X_test)
# y_score5 = etc.predict_proba(X_test)
# y_score6 = bag.predict_proba(X_test)
# y_score7 = xgb.predict_proba(X_test)
# y_score9 = svc.decision_function(X_test)
# y_score12 = knn.predict_proba(X_test)

# %%
# plt.figure(figsize=(10,7))
# lw = 2

# #################### GAUSSIAN NAIVE BAYES ####################
# fpr1 = {}
# tpr1 = {}
# roc_auc1 = {}
# for i in range(n_classes):
#     fpr1[i], tpr1[i], _ = roc_curve(y_test_bin[:, i], y_score1[:, i])
#     roc_auc1[i] = auc(fpr1[i], tpr1[i])

# fpr1["micro"], tpr1["micro"], _ = roc_curve(y_test_bin.ravel(), y_score1.ravel())
# roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])

# plt.plot(
#     fpr1["micro"],
#     tpr1["micro"],
#     label="Gaussian Naïve Bayes (AUC:{0:0.2f})".format(roc_auc1["micro"]),
#     color="navy",
#     linestyle=":",
#     linewidth=lw,
# )

# #################### EXTRA TREES ####################
# fpr5 = {}
# tpr5 = {}
# roc_auc5 = {}
# for i in range(n_classes):
#     fpr5[i], tpr5[i], _ = roc_curve(y_test_bin[:, i], y_score5[:, i])
#     roc_auc5[i] = auc(fpr5[i], tpr5[i])

# fpr5["micro"], tpr5["micro"], _ = roc_curve(y_test_bin.ravel(), y_score5.ravel())
# roc_auc5["micro"] = auc(fpr5["micro"], tpr5["micro"])

# plt.plot(
#     fpr5["micro"],
#     tpr5["micro"],
#     label="Extra Trees (AUC:{0:0.2f})".format(roc_auc5["micro"]),
#     color="darkred",
#     linestyle=":",
#     linewidth=lw,
# )

# #################### BAGGING ####################
# fpr6 = {}
# tpr6 = {}
# roc_auc6 = {}
# for i in range(n_classes):
#     fpr6[i], tpr6[i], _ = roc_curve(y_test_bin[:, i], y_score6[:, i])
#     roc_auc6[i] = auc(fpr6[i], tpr6[i])

# fpr6["micro"], tpr6["micro"], _ = roc_curve(y_test_bin.ravel(), y_score6.ravel())
# roc_auc6["micro"] = auc(fpr6["micro"], tpr6["micro"])

# plt.plot(
#     fpr6["micro"],
#     tpr6["micro"],
#     label="Bagging (AUC:{0:0.2f})".format(roc_auc6["micro"]),
#     color="purple",
#     linestyle=":",
#     linewidth=lw,
# )

# #################### XGBOOST ####################
# fpr7 = {}
# tpr7 = {}
# roc_auc7 = {}
# for i in range(n_classes):
#     fpr7[i], tpr7[i], _ = roc_curve(y_test_bin[:, i], y_score7[:, i])
#     roc_auc7[i] = auc(fpr7[i], tpr7[i])

# fpr7["micro"], tpr7["micro"], _ = roc_curve(y_test_bin.ravel(), y_score7.ravel())
# roc_auc7["micro"] = auc(fpr7["micro"], tpr7["micro"])

# plt.plot(
#     fpr7["micro"],
#     tpr7["micro"],
#     label="XGBoost (AUC:{0:0.2f})".format(roc_auc7["micro"]),
#     color="olivedrab",
#     linestyle=":",
#     linewidth=lw,
# )

# #################### SVC ####################
# fpr9 = {}
# tpr9 = {}
# roc_auc9 = {}
# for i in range(n_classes):
#     fpr9[i], tpr9[i], _ = roc_curve(y_test_bin[:, i], y_score9[:, i])
#     roc_auc9[i] = auc(fpr9[i], tpr9[i])

# fpr9["micro"], tpr9["micro"], _ = roc_curve(y_test_bin.ravel(), y_score9.ravel())
# roc_auc9["micro"] = auc(fpr9["micro"], tpr9["micro"])

# plt.plot(
#     fpr9["micro"],
#     tpr9["micro"],
#     label="SVC (AUC:{0:0.2f})".format(roc_auc9["micro"]),
#     color="lime",
#     linestyle=":",
#     linewidth=lw,
# )

# #################### K NEAREST NEIGHBOR ####################
# fpr12 = {}
# tpr12 = {}
# roc_auc12 = {}
# for i in range(n_classes):
#     fpr12[i], tpr12[i], _ = roc_curve(y_test_bin[:, i], y_score12[:, i])
#     roc_auc12[i] = auc(fpr12[i], tpr12[i])

# fpr12["micro"], tpr12["micro"], _ = roc_curve(y_test_bin.ravel(), y_score12.ravel())
# roc_auc12["micro"] = auc(fpr12["micro"], tpr12["micro"])

# plt.plot(
#     fpr12["micro"],
#     tpr12["micro"],
#     label="K Nearest Neighbor (AUC:{0:0.2f})".format(roc_auc12["micro"]),
#     color="pink",
#     linestyle=":",
#     linewidth=lw,
# )

# ###############################################################

# plt.plot([0, 1], [0, 1], "k--")
# plt.xlim([-.02, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve Comparison")
# plt.legend(loc="lower right")
# plt.savefig('FigXX-ROCcomparison.png', dpi=600)
# plt.show()

# %% [markdown]
# # **<font color="#42f5f5">9.0 SEQUENTIAL PATTERN MINING</font>**

# %% [markdown]
# https://pypi.org/project/prefixspan/

# %%
!pip install prefixspan
from prefixspan import PrefixSpan

# %%
data_ts0=pd.read_csv('/content/local_data/train/Inertial Signals/body_acc_x_train.txt', header=None, delim_whitespace=True)
data_ts1=pd.read_csv('/content/local_data/train/Inertial Signals/body_acc_y_train.txt', header=None, delim_whitespace=True)
data_ts2=pd.read_csv('/content/local_data/train/Inertial Signals/body_acc_z_train.txt', header=None, delim_whitespace=True)
data_ts3=pd.read_csv('/content/local_data/train/Inertial Signals/body_gyro_x_train.txt', header=None, delim_whitespace=True)
data_ts4=pd.read_csv('/content/local_data/train/Inertial Signals/body_gyro_y_train.txt', header=None, delim_whitespace=True)
data_ts5=pd.read_csv('/content/local_data/train/Inertial Signals/body_gyro_z_train.txt', header=None, delim_whitespace=True)
data_ts6=pd.read_csv('/content/local_data/train/Inertial Signals/total_acc_x_train.txt', header=None, delim_whitespace=True)
data_ts7=pd.read_csv('/content/local_data/train/Inertial Signals/total_acc_y_train.txt', header=None, delim_whitespace=True)
data_ts8=pd.read_csv('/content/local_data/train/Inertial Signals/total_acc_z_train.txt', header=None, delim_whitespace=True)

data_test_ts0=pd.read_csv('/content/local_data/test/Inertial Signals/body_acc_x_test.txt', header=None, delim_whitespace=True)
data_test_ts1=pd.read_csv('/content/local_data/test/Inertial Signals/body_acc_y_test.txt', header=None, delim_whitespace=True)
data_test_ts2=pd.read_csv('/content/local_data/test/Inertial Signals/body_acc_z_test.txt', header=None, delim_whitespace=True)
data_test_ts3=pd.read_csv('/content/local_data/test/Inertial Signals/body_gyro_x_test.txt', header=None, delim_whitespace=True)
data_test_ts4=pd.read_csv('/content/local_data/test/Inertial Signals/body_gyro_y_test.txt', header=None, delim_whitespace=True)
data_test_ts5=pd.read_csv('/content/local_data/test/Inertial Signals/body_gyro_z_test.txt', header=None, delim_whitespace=True)
data_test_ts6=pd.read_csv('/content/local_data/test/Inertial Signals/total_acc_x_test.txt', header=None, delim_whitespace=True)
data_test_ts7=pd.read_csv('/content/local_data/test/Inertial Signals/total_acc_y_test.txt', header=None, delim_whitespace=True)
data_test_ts8=pd.read_csv('/content/local_data/test/Inertial Signals/total_acc_z_test.txt', header=None, delim_whitespace=True)

# %%
data_ts = (data_ts6-data_ts6.min())/(data_ts6.max()-data_ts6.min())
data_test_ts = (data_test_ts6-data_test_ts6.min())/(data_test_ts6.max()-data_test_ts6.min())
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series

# %%
n_paa_segments = 12
n_sax_symbols = 12
sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)


# %%
list_sax=[]
for i in range(0,7352):
  ts=data_ts.iloc[i,:].values
  ts=scaler.fit_transform(ts.reshape(1,-1))
  ts=sax.fit_transform(ts).ravel()
  list_sax.append(ts)

# %%
len(list_sax)

# %%
list_sax[7351]

# %%
plt.plot(list_sax[2])
plt.show()

# %%
ps1 = PrefixSpan(list_sax)
ps1.minlen = 10

# %%
psx1 = ps1.frequent(10)
psx1

# %%
res_list1 = [x[0] for x in psx1]
print(sum(res_list1))

# %%
!pip install pyspark

# %%
from pyspark.ml.fpm import PrefixSpan

# %%
from pyspark import SparkContext
sc =SparkContext()

# %%
from pyspark.sql import Row
df = sc.parallelize([Row(sequence=[db])]).toDF()

# %%
prefixspan = PrefixSpan()

# %%
prefixspan.findFrequentSequentialPatterns(db).sort("sequence").show(truncate=False)

# %%
!pip install spmf

# %%
spmf = Spmf("GSP", spmf_bin_location_dir='/content/local_data/train/',
            input_direct=db,
            output_filename="output.txt",
            arguments=[0.5])

sp.run()
output = spmf.to_pandas_dataframe(pickle=True)
output

# %% [markdown]
# # **<font color="#42f5f5">10.0 ADVANCED CLUSTERING</font>**

# %%
!pip install pyclustering

from pyclustering.cluster.silhouette import silhouette
from sklearn.metrics import silhouette_score

# %%
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

i = 0
j = 1

# %%
data_cluster = data.copy()
data_cluster['activity'] = y_label[0].values

# Seleziono le classi che voglio droppare. Lasciamo solo 2 e 3, 6 era troppo facile da predictare-->
classe2 = np.array(y_label[y_label[0]==2].index)
classe3 = np.array(y_label[y_label[0]==3].index)
classe4 = np.array(y_label[y_label[0]==4].index)
classe5 = np.array(y_label[y_label[0]==5].index)
classes2remove = np.concatenate((classe2,classe3,classe4,classe5))

data_cluster.drop(data_cluster.index[classes2remove], inplace=True)

X_train_cluster = data_cluster.iloc[:, 0:561].values
y_train_cluster = data_cluster['activity'].values
print(data_cluster['activity'].value_counts(), "\n")

print(X_train_cluster.shape)
print(X_train_cluster.shape)

# %% [markdown]
# ## **<font color="#FBBF44">10.x Gaussian Mixture Model</font>**

# %%
from sklearn.mixture import GaussianMixture

# %%
n_components = np.arange(1, 10)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X_train_pca) for n in n_components]

# %%
plt.plot(n_components, [m.aic(X_train_pca) for m in models], label='Akaike information criterion (AIC)')
plt.legend(loc='best')
plt.xlabel('n_components');

# %%
gmm = GaussianMixture(n_components=6)
gmm.fit(X_train_pca)

# %%
labels = gmm.predict(X_train_pca)
plt.scatter(X_train_pca[:, i], X_train_pca[:, j], c=labels, cmap=plt.cm.Set3);
plt.title('Gaussian Mixture Clustering of raw dataset')
plt.savefig('FigXX-ClusterGMM1.png', dpi=600)
plt.show()

# %%
silhouette_score(X_train_pca, labels)

# %% [markdown]
# ## **<font color="#FBBF44">10.x X-Means</font>**

# %%
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.elbow import elbow

# %% [markdown]
# https://pyclustering.github.io/docs/0.9.0/html/dd/db4/classpyclustering_1_1cluster_1_1xmeans_1_1xmeans.html
#

# %%
kmin, kmax = 1, 10
elbow_instance = elbow(X_train_pca, kmin, kmax)
elbow_instance.process()
#amount_clusters = elbow_instance.get_amount()
#amount_clusters

# %%
wce = elbow_instance.get_wce()

plt.plot(wce, label='WCE Within Cluster Error')
plt.legend(loc='best')
plt.xlabel('k');
plt.ylabel('WCE');

# %%
amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(X_train_pca, amount_initial_centers).initialize()
xm = xmeans(X_train_pca, initial_centers=initial_centers, kmax=6)
xm.process()

clusters = xm.get_clusters()
centers = xm.get_centers()

len(clusters)

# %%
for clust in clusters:
    plt.scatter(X_train_pca[clust,i], X_train_pca[clust,j], alpha=0.6)
for cent in centers:
    plt.scatter(cent[i], cent[j], s=60, edgecolors='k', marker='o')
plt.title('X-means Clustering of raw data after PCA')
plt.savefig('FigXX-ClusterXmeans2.png', dpi=600)
plt.show()

# %%
score = silhouette(X_train_pca, clusters).process().get_score()
np.mean(score)

# %% [markdown]
# ## **<font color="#FBBF44">10.x OPTICS</font>**

# %%
from sklearn.cluster import OPTICS

# %%
# Calcoliamo i valori di distanza tra ogni record e il suo nearest neighbor
nbr = NearestNeighbors(n_neighbors=2)
nbrs = nbr.fit(X_train)
distances, indices = nbrs.kneighbors(X_train)

# Plottiamo la distanza dentro i valori del df e cerchiamo il "gomito" per vedere il punto di massima curvatura e quindi il valore ottimo di Eps
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances, color = 'darkcyan')
plt.title('K-distance Graph',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Eps',fontsize=14)
plt.savefig('FigXX-ElbowMethodDBSCAN.png', dpi=600)
plt.show()

# %%
optics = OPTICS(min_samples=5, eps=3)
optics.fit(X_train)

# %%
silhouette_score(X_train[optics.labels_ != -1], optics.labels_[optics.labels_ != -1])

# %%
for cluster_id in np.unique(optics.labels_)[:10]:
    indexes = np.where(optics.labels_==cluster_id)
    plt.scatter(X_train[indexes,i], X_train[indexes,j], alpha=0.8)

# %% [markdown]
# ## **<font color="#FBBF44">10.x K-Mode</font>**

# %%


# %% [markdown]
# ## **<font color="#FBBF44">10.x ROCK</font>**

# %%
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.rock import rock

rock_instance = rock(X_train, 1.0, 7)
# Run cluster analysis.
rock_instance.process()
# Obtain results of clustering.
clusters = rock_instance.get_clusters()

# %%


# %% [markdown]
# ## **<font color="#FBBF44">10.x Transactional Clustering</font>**

# %% [markdown]
# # **<font color="#42f5f5">10.0 ADVANCED CLUSTERING'</font>**

# %%
!pip install pyclustering

from pyclustering.cluster.silhouette import silhouette
from sklearn.metrics import silhouette_score

# %%
data_cluster = data.copy()
data_cluster['activity'] = y_label[0].values

# Seleziono le classi che voglio droppare. Lasciamo solo 2 e 3, 6 era troppo facile da predictare-->
classe1 = np.array(y_label[y_label[0]==1].index)
classe2 = np.array(y_label[y_label[0]==2].index)
classe3 = np.array(y_label[y_label[0]==3].index)
classe4 = np.array(y_label[y_label[0]==4].index)
classe5 = np.array(y_label[y_label[0]==5].index)
classe6 = np.array(y_label[y_label[0]==6].index)
classes2remove = np.concatenate((classe2, classe3, classe4, classe5))

data_cluster.drop(data_cluster.index[classes2remove], inplace=True)

X_train_cluster = data_cluster.iloc[:, 0:561].values
y_train_cluster = data_cluster['activity'].values
print(data_cluster['activity'].value_counts(), "\n")

print(X_train_cluster.shape)
print(X_train_cluster.shape)

# %%
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_cluster)

i = 0
j = 1

# %% [markdown]
# ## **<font color="#FBBF44">10.x Gaussian Mixture Model</font>**

# %%
from sklearn.mixture import GaussianMixture

# %%
n_components = np.arange(1, 10)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X_train_pca) for n in n_components]

# %%
plt.plot(n_components, [m.aic(X_train_pca) for m in models], label='Akaike information criterion (AIC)')
plt.legend(loc='best')
plt.xlabel('n_components');
plt.title('Akaike Information Criterion')
plt.savefig('FigXX-AICGauss.png', dpi=600)
plt.show()

# %%
gmm = GaussianMixture(n_components=2)
gmm.fit(X_train_pca)

# %%
from matplotlib import cm
from matplotlib.colors import ListedColormap

# make the color map:
cmp = ListedColormap(['tab:blue', 'tab:orange'])

labels = gmm.predict(X_train_pca)
plt.scatter(X_train_pca[:, i], X_train_pca[:, j], c=labels, cmap=cmp);
plt.title('Gaussian Mixture Model Clustering')
plt.savefig('FigXX-ClusterGMM.png', dpi=600)
plt.show()

# %%
silhouette_score(X_train_pca, labels)

# %% [markdown]
# ## **<font color="#FBBF44">10.x X-Means</font>**

# %%
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.elbow import elbow

# %% [markdown]
# https://pyclustering.github.io/docs/0.9.0/html/dd/db4/classpyclustering_1_1cluster_1_1xmeans_1_1xmeans.html
#

# %%
kmin, kmax = 1, 10
elbow_instance = elbow(X_train_pca, kmin, kmax)
elbow_instance.process()
#amount_clusters = elbow_instance.get_amount()
#amount_clusters

# %%
wce = elbow_instance.get_wce()

plt.plot(wce, label='WCE Within Cluster Error')
plt.legend(loc='best')
plt.xlabel('k');
plt.ylabel('WCE');
plt.title('Within Cluster Error')
plt.savefig('FigXX-ClusterWCE.png', dpi=600)
plt.show()

# %%
amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(X_train_pca, amount_initial_centers).initialize()
xm = xmeans(X_train_pca, initial_centers=initial_centers, kmax=2)
xm.process()

clusters = xm.get_clusters()
centers = xm.get_centers()

len(clusters)

# %%
for clust in clusters:
    plt.scatter(X_train_pca[clust,i], X_train_pca[clust,j], alpha=0.6)
for cent in centers:
    plt.scatter(cent[i], cent[j], s=60, edgecolors='k', marker='o')
plt.title('X-means Clustering')
plt.savefig('FigXX-ClusterXmeans.png', dpi=600)
plt.show()

# %%
score = silhouette(X_train_pca, clusters).process().get_score()
np.mean(score)

# %% [markdown]
# ## **<font color="#FBBF44">10.x OPTICS</font>**

# %%
from sklearn.cluster import OPTICS

# %%
# Calcoliamo i valori di distanza tra ogni record e il suo nearest neighbor
nbr = NearestNeighbors(n_neighbors=2)
nbrs = nbr.fit(X_train_pca)
distances, indices = nbrs.kneighbors(X_train_pca)

# Plottiamo la distanza dentro i valori del df e cerchiamo il "gomito" per vedere il punto di massima curvatura e quindi il valore ottimo di Eps
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances, color = 'darkcyan')
plt.title('K-distance Graph',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Eps',fontsize=14)
plt.savefig('FigXX-ElbowMethodDBSCAN.png', dpi=600)
plt.show()

# %%
optics = OPTICS(min_samples=30, eps=.2)
optics.fit(X_train_pca)

# %%
silhouette_score(X_train_pca[optics.labels_ != -1], optics.labels_[optics.labels_ != -1])

# %%
for cluster_id in np.unique(optics.labels_)[:10]:
    indexes = np.where(optics.labels_==cluster_id)
    plt.scatter(X_train_pca[indexes,i], X_train_pca[indexes,j], alpha=0.8)
plt.title('OPTICS Clustering')
plt.savefig('FigXX-ClusterOptics.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">10.x K-Mode</font>**

# %%


# %% [markdown]
# ## **<font color="#FBBF44">10.x ROCK</font>**

# %%
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.rock import rock

rock_instance = rock(X_train, 1.0, 7)
# Run cluster analysis.
rock_instance.process()
# Obtain results of clustering.
clusters = rock_instance.get_clusters()

# %%


# %% [markdown]
# ## **<font color="#FBBF44">10.x Transactional Clustering</font>**

# %%
import os, sys, importlib
from os.path import expanduser
from pathlib import Path

!cp -r /content/drive/MyDrive/"TX-Means-master" /content/local_data

home = str(Path.home())
home

Folder_Cloned_In = '/content/local_data/TX-Means-master'
path_to_lib = home + Folder_Cloned_In

if os.path.isdir(path_to_lib + 'TXMeans'):
    print(f'My Home is: {home}')
    print(f'I cloned in: {path_to_lib}')
    # Add dirs to Python Path
    sys.path.insert(0, path_to_lib + 'TXMeans/code')
    sys.path.insert(0, path_to_lib + 'TXMeans/code/algorithms')
else:
    print("Can't find Directory.")
    print('For example: you are in')
    print(str(os.getcwd()))

# %%
# sys.path.append('/local_data/TX-Means-master/code/algorithms')
# # sys.path.append('/content/local_data/TX-Means-master/code/generators')
# # sys.path.append('/content/local_data/TX-Means-master/code/validators')
# # sys.path.append('/content/local_data/TX-Means-master/code/validators/algorithms/util.py')

# import algorithms.txmeans

# %%
# import algorithms.txmeans

# %% [markdown]
# # **<font color="#42f5f5">11.0 EXPLAINABILITY</font>**

# %%
import sys

sys.path.append('/content/drive/MyDrive/Colab Notebooks/DM2/lore')

# %%
import pydotplus
from sklearn import tree
from IPython.display import Image

from util import record2str, neuclidean
from datamanager import prepare_adult_dataset, prepare_dataset, one_hot_encoding

# %%
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_list,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# %% [markdown]
# ## **<font color="#FBBF44">11.x Skater</font>**

# %%
# !pip install skater

# from skater.model import InMemoryModel
# from skater.core.explanations import Interpretation

# from skater.core.global_interpretation.partial_dependence import PartialDependence

# %%
# interpreter = Interpretation()
# interpreter.load_data(X_train, feature_names=feature_list)

# %% [markdown]
# ## **<font color="#FBBF44">11.x Local</font>**

# %%
numeric_columns = list(data._get_numeric_data().columns)

# %%
bb = DecisionTreeClassifier(random_state=0)
bb.fit(X_train, y_train)

def bb_predict(X):
    return bb.predict(X)

def bb_predict_proba(X):
    return bb.predict_proba(X)

# %%
i2e = 235
x = X_test[i2e]

record2str(x, feature_list, numeric_columns)

bb_outcome = bb_predict(x.reshape(1, -1))[0]
bb_outcome_str = y_train[bb_outcome]

print('bb(x) = { %s }' % bb_outcome_str)
print('')

# %% [markdown]
# ## **<font color="#FBBF44">11.x Lime</font>**

# %%
!pip install lime

# %%
import lime
from lime import lime_tabular

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))

# %%
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict_proba(X_test)

# %%
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_list,
    class_names=['1', '2','3','4','5','6'],
    mode='classification'
)

# %%
exp = explainer.explain_instance(
    data_row=X_test[1,:],
    predict_fn=rfc.predict_proba
)

# Su jupyter va bene questo
exp.show_in_notebook(show_table=True)


