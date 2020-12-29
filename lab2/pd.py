import pandas as pd
import numpy as np


df=pd.read_csv('lab02Exercise01.csv', names=['Channel1','Channel2','Channel3','Channel4','Channel5'])
df.loc[(df['Channel1']+df['Channel5'])/2 <(df['Channel2']+df['Channel3']+df['Channel4'])/3,'class']=1
df.loc[(df['Channel1']+df['Channel5'])/2>=(df['Channel2']+df['Channel3']+df['Channel4'])/3,'class']=0
print(df)
