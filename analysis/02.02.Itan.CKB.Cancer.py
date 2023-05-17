import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

ICC = pd.read_csv('figs/ICC.csv')
plt.figure(figsize=(6,6))
p = sns.scatterplot(data=ICC, x="energy", y="pLDDT", hue="LABEL", alpha=0.2)
g = sns.kdeplot(x="energy", y="pLDDT", hue="LABEL", data=ICC, levels=5, thresh=0.01, alpha=1, ax=p)
g.figure.savefig('figs/Itan.CKB.Cancer.contour.pdf')

ICC = pd.read_csv('figs/Itan.CKB.Cancer.ddg.csv')
plt.figure(figsize=(6,6))
p = sns.scatterplot(data=ICC, x="ddg", y="pLDDT", hue="LABEL", alpha=0.2)
g = sns.kdeplot(x="ddg", y="pLDDT", hue="LABEL", data=ICC, levels=5, thresh=0.01, alpha=1, ax=p)
g.figure.savefig('figs/Itan.CKB.Cancer.ddg.contour.pdf')

pfams = ["PF07714", "PF00454", "PF00069", "PF07679", "PF00047", "PF00028", "PF00520", "PF06512", "PF11933"]
for i in pfams:
  pfam = pd.read_csv(f'figs/{i}.csv')
  plt.figure(figsize=(6,6))
  p = sns.scatterplot(data=pfam, x="energy", y="pLDDT", hue="LABEL", alpha=0.2)
  g = sns.kdeplot(x="energy", y="pLDDT", hue="LABEL", data=pfam, levels=5, thresh=0.01, alpha=1, ax=p)
  g.figure.savefig(f'figs/{i}.contour.pdf')
  pfam = pd.read_csv(f'figs/{i}.ddg.csv')
  plt.figure(figsize=(6,6))
  p = sns.scatterplot(data=pfam, x="ddg", y="pLDDT", hue="LABEL", alpha=0.2)
  g = sns.kdeplot(x="ddg", y="pLDDT", hue="LABEL", data=pfam, levels=5, thresh=0.01, alpha=1, ax=p)
  g.figure.savefig(f'figs/{i}.ddg.contour.pdf')
  
  

  
