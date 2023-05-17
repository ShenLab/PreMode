import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

ICC = pd.read_csv('figs/IonChannel.csv')
plt.figure(figsize=(6,6))
p = sns.scatterplot(data=ICC, x="energy", y="pLDDT", hue="LABEL", alpha=0.2)
g = sns.kdeplot(x="energy", y="pLDDT", hue="LABEL", data=ICC, levels=5, thresh=0.05, alpha=1, ax=p)
g.figure.savefig('figs/IonChannel.contour.pdf')

ICC = pd.read_csv('figs/IonChannel.ddg.csv')
plt.figure(figsize=(6,6))
p = sns.scatterplot(data=ICC, x="ddg", y="pLDDT", hue="LABEL", alpha=0.2)
g = sns.kdeplot(x="ddg", y="pLDDT", hue="LABEL", data=ICC, levels=5, thresh=0.01, alpha=1, ax=p)
g.figure.savefig('figs/IonChannel.ddg.contour.pdf')
