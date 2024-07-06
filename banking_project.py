import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"C:\Users\diwak\OneDrive\Desktop\banking_data.csv")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_row",None)
# print(df.info())
unique_count=df["y"].nunique()
# print(unique_count)
value_counts=df["campaign"].value_counts()
# print(value_counts)
# print(value_counts)
# print(df["campaign"].max())


# print(df["contact"].head(150))

# Q1
print(df["age"].describe())
# plt.hist(df["age"],bins=10,color="green",edgecolor="red")
# sns.countplot(x="age",data=df)
plt.show()

# Q2
# unique_job=df["job"].nunique()
# print(unique_job)
# unique_value_job=df["job"].value_counts()
# print(unique_value_job)
# sns.countplot(x="job",hue="job",data=df)
# plt.title("Job Distribution of Customers",color="red")
# plt.show()
# print((df["job"]=="student").sum())


# Q3
# sns.countplot(x="marital",data=df)
# plt.title("Marital Status of Customers",color="red")
# plt.show()

# Q4
# sns.countplot(x="education",data=df)
# plt.title("educationalqualification of Customers",color="red")
# plt.show()
