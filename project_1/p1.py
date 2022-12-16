#! /home/mny-1026/miniconda3/envs/fdd2/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

purchase_carts = pd.read_csv('purchase_carts1.csv', header=None)
col_names= ['cart_id', 'user_id','book_id', 'purchase_date', 'percentile']
purchase_carts = purchase_carts.rename(columns=lambda x: col_names[x])

ids = purchase_carts['user_id'].max()

mean_p_user = []

delta = []

for id in range(1,ids+1):
	mean_p = purchase_carts.loc[purchase_carts['user_id'] == id, 'percentile'].mean()
	mean_p_user.append(mean_p)

	delta.append(np.random.exponential(scale=mean_p*10**(1+mean_p)))


	purchase_carts_2 = purchase_carts.loc[purchase_carts['user_id'] == id]
	purchase_carts_2.reset_index(inplace = True, drop = True)
	date = purchase_carts_2.loc[0]['purchase_date']
	purchase_carts_2.loc[:,'cart_id'] = 2
	purchase_carts_2['purchase_date'] = pd.to_datetime(purchase_carts_2['purchase_date'])

	purchase_carts_2.loc[:, 'purchase_date'] += pd.Timedelta(days = delta[id-1])

	date_2 = purchase_carts_2.loc[0]['purchase_date']
	frames = [purchase_carts, purchase_carts_2]
	purchase_carts = pd.concat(frames, ignore_index=True)


for i in range(len(delta)):
	if delta[i] < 1:
		delta[i] += 1 

result = {"mean percentile":mean_p_user, "delta":delta}
df = pd.DataFrame(result)

plt.scatter(mean_p_user, delta)
plt.xlabel("Mean percentile")
plt.ylabel("Days between purchases")
plt.show()

print(df["mean percentile"].corr(df["delta"]))
