import pandas as pd

df = pd.read_csv('data/HCP_s1200.csv')
with open("data/prediction_variables.txt") as f:
    pred_vars = f.read().splitlines()
df = df[pred_vars]
mean = df.mean()
std_error = df.std() / df.mean()
result = pd.DataFrame({'mean': mean, 'n_std_error': std_error})
print(result)