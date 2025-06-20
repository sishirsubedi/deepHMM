import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns 


model = 'HMM'

df_tnloss = pd.read_csv('output/'+model+'model_train_loss.csv.gz')
df_tnloss['model'] = model+'_train'
df_tnloss.columns = ['index','loss','model']
print(df_tnloss.shape)
df_ttloss = pd.read_csv('output/'+model+'model_test_loss.csv.gz')
df_ttloss['model'] = model+'_test'
df_ttloss.columns = ['index','loss','model']
print(df_ttloss.shape)

df = pd.concat([df_tnloss,df_ttloss])
df = df.drop(columns='index').reset_index()


sns.lineplot(data=df, x='index', y='loss', hue='model')

plt.savefig('output/'+model+'_eval_plot.png')
plt.close()

model = 'DHMM'

df_tnloss2 = pd.read_csv('output/'+model+'model_train_loss.csv.gz')
df_tnloss2['model'] = model+'_train'
df_tnloss2.columns = ['index','loss','model']
print(df_tnloss2.shape)
df_ttloss2 = pd.read_csv('output/'+model+'model_test_loss.csv.gz')
df_ttloss2['model'] = model+'_test'
df_ttloss2.columns = ['index','loss','model']
print(df_ttloss2.shape)

df = pd.concat([df_tnloss2,df_ttloss2])
df = df.drop(columns='index').reset_index()


sns.lineplot(data=df, x='index', y='loss', hue='model')

plt.savefig('output/'+model+'_eval_plot.png')
plt.close()