from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules 
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np


#(a) Import the given dataset ’groceries.csv’ 
data=pd.read_csv('groceries.csv')


#(b) Explore the dataset and create the frequent item dataset. 
'''
print(dataset.shape)
(9834, 32)

print(dataset.head())
 citrus fruit semi-finished bread       margarine               ready soups        Unnamed: 4  ... Unnamed: 27 Unnamed: 28 Unnamed: 29 Unnamed: 30 Unnamed: 31
0    tropical fruit              yogurt          coffee                       NaN               NaN  ...         NaN         NaN         NaN         NaN         NaN
1        whole milk                 NaN             NaN                       NaN               NaN  ...         NaN         NaN         NaN         NaN         NaN
2         pip fruit              yogurt    cream cheese              meat spreads               NaN  ...         NaN         NaN         NaN         NaN         NaN
3  other vegetables          whole milk  condensed milk  long life bakery product               NaN  ...         NaN         NaN         NaN         NaN         NaN
4        whole milk              butter          yogurt                      rice  abrasive cleaner  ...         NaN         NaN         NaN         NaN         NaN

[5 rows x 32 columns]


'''
data.fillna(0,inplace=True)
trans = []
for i in range(0,len(data)):
    trans.append([str(data.values[i,j]) for j in range(0,32) if str(data.values[i,j])!='0'])


te = TransactionEncoder() 
te_ary = te.fit(trans).transform(trans) 
df = pd.DataFrame(te_ary, columns=te.columns_)


#(c) Apply Apriori algorithm and identify the itemsets

frequent_itemsets= apriori(df, min_support = 0.05, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

'''
print(frequent_itemsets)

    support                        itemsets  length
0   0.052471                          (beef)       1
1   0.080537                  (bottled beer)       1
2   0.110535                 (bottled water)       1
3   0.064877                   (brown bread)       1
4   0.055420                        (butter)       1
5   0.077690                   (canned beer)       1
6   0.082672                  (citrus fruit)       1
7   0.058064                        (coffee)       1
8   0.053285                          (curd)       1
9   0.063453                 (domestic eggs)       1
10  0.058979                   (frankfurter)       1
11  0.072300         (fruit/vegetable juice)       1
12  0.058471                     (margarine)       1
13  0.052369                       (napkins)       1
14  0.079825                    (newspapers)       1
15  0.193512              (other vegetables)       1
16  0.088977                        (pastry)       1
17  0.075656                     (pip fruit)       1
18  0.057657                          (pork)       1
19  0.183954                    (rolls/buns)       1
20  0.109010               (root vegetables)       1
21  0.093960                       (sausage)       1
22  0.098536                 (shopping bags)       1
23  0.174395                          (soda)       1
24  0.104942                (tropical fruit)       1
25  0.071690            (whipped/sour cream)       1
26  0.255542                    (whole milk)       1
27  0.139516                        (yogurt)       1
28  0.074842  (whole milk, other vegetables)       2
29  0.056640        (rolls/buns, whole milk)       2
30  0.056030            (yogurt, whole milk)       2

'''

#(d) Find the set of Association rules using the metric ’lift’ 

rules= association_rules(frequent_itemsets, metric="lift", min_threshold=1)
'''
print(rules)
 antecedents         consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction
0        (whole milk)  (other vegetables)            0.255542            0.193512  0.074842    0.292877  1.513480  0.025392    1.140520
1  (other vegetables)        (whole milk)            0.193512            0.255542  0.074842    0.386758  1.513480  0.025392    1.213971
2        (rolls/buns)        (whole milk)            0.183954            0.255542  0.056640    0.307905  1.204909  0.009632    1.075659
3        (whole milk)        (rolls/buns)            0.255542            0.183954  0.056640    0.221647  1.204909  0.009632    1.048428
4            (yogurt)        (whole milk)            0.139516            0.255542  0.056030    0.401603  1.571575  0.020378    1.244088
5        (whole milk)            (yogurt)            0.255542            0.139516  0.056030    0.219260  1.571575  0.020378    1.102139
'''

'''
#(e) Select any Association rule from the set and describe the selected rule.
confidence rule:
The confidence of an association rule is a percentage value that shows how frequently the rule head occurs among all the groups containing the rule body. 
The confidence value indicates how reliable this rule is. 
The higher the value, the more likely the head items occur in a group if it is known that all body items are contained in that group.
'''


#(f) How many rules are there when the ’lift’ is greater than 4 and the ’conﬁdence’ is greater than 0.8.

new_rules=rules[(rules['lift']>4) & (rules['confidence']>0.8)]
'''
print(new_rules)
Empty DataFrame
Columns: [antecedents, consequents, antecedent support, consequent support, support, confidence, lift, leverage, conviction]
Index: []

when the ’lift’ is greater than 4 and the ’conﬁdence’ is greater than 0.8,number of rules are 0.

