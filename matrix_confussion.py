from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_pred = 'firefighter-float16-size-optimized.csv'
path_df = 'training\obj\obj\dataset.csv'

df_pred = pd.read_csv(path_pred)
df = pd.read_csv(path_df)

df_temp = []
for i in df['actual'].values.tolist():
    if i == 'Padam':
        df_temp.append(0)
    else:
        df_temp.append(1)

df_pred_temp = []
for i in df_pred['predict'].values.tolist():
    if i == 'Padam':
        df_pred_temp.append(0)
    else:
        df_pred_temp.append(1)

confusion_matrix = metrics.confusion_matrix(df_temp, df_pred_temp)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=['Padam', 'Nyala'])
cm_display.plot()

name_fig = 'firefighter-float16-size-optimized.png'
plt.savefig(name_fig)
print(f'Accuracy : {accuracy_score(df_temp, df_pred_temp)}')
print(f'Recall : {recall_score(df_temp, df_pred_temp)}')
print(f'F1 Score : {f1_score(df_temp, df_pred_temp)}')
