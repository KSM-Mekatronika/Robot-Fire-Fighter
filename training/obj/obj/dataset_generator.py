import glob
import os
import pandas as pd

dataframe = {
    "img": [],
    "actual": []
}

images = glob.glob('*.jpg')
texts = glob.glob('*.txt')

for i in images:
    text = os.path.splitext(i)[0] + '.txt'
    df = pd.read_csv(text, delim_whitespace=True)
    if df.columns[0] == "0":
        dataframe["actual"].append("Padam")
    else:
        dataframe["actual"].append("Nyala")
    dataframe["img"].append(i)

df = pd.DataFrame(dataframe)
df.to_csv('dataset.csv')
