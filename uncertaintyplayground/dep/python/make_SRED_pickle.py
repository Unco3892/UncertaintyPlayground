import pandas as pd

df_train = pd.read_csv ('data/metadata/train_data.csv')
df_test = pd.read_csv ('data/metadata/train_data.csv')
print (df_train)
print (df_test)

# take the log of the file in R? Or make the transformation directly in python?
# better to make the transformation directly in python (colab) for interpretability
df_train.to_pickle('data/metadata/train_data.pkl')
df_test.to_pickle('data/metadata/test_data.pkl')
