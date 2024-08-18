# Predictive Modeling for Term Deposit Subscription in Bank Marketing

Author : Arpit Singh

---


# Introduction
**Table of Content**:

- Introduction
- Load Dataset
- Data Cleaning
- Exploratory Data Analysis
- Data Preprocessing
- Modeling
- Evaluation
- Business Recomendation
**Background**


  
# Library and Load Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
import math
import scipy.stats as ss
from scipy.stats import pointbiserialr
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
sns.set(style='whitegrid')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
!pip install catboost
!pip install bayesian-optimization
!pip install shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
import joblib

# Model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Evaluation
from bayes_opt import BayesianOptimization
import shap
!git clone https://github.com/Theofilusarifin/Predictive-Modeling-for-Term-Deposit-Subscription-in-Bank-Marketing.git
## Read Dataset

---


df = pd.read_csv('/content/Predictive-Modeling-for-Term-Deposit-Subscription-in-Bank-Marketing/dataset/dataset.csv')
df.shape
df.sample(5)
df.info()
## Dataset Overview


---


### Feature Segmentation
#### Client Data

| **Feature** | **Description**                                                                                        |
|------------|--------------------------------------------------------------------------------------------------------|
| `ID`       | ID of client                                                                                           |
| `age`      | Age of client                                                                                          |
| `job`      | Type of job                                                                                            |
| `marital`  | Marital status                                                                                         |
| `education`| Education level                                                                                        |
| `default`  | Has credit in default?                                                                                |
| `balance`  | Balance in account                                                                                     |
| `housing`  | Has housing loan?                                                                                      |
| `loan`     | Has personal loan?                                                                                     |
#### Latest Campaign Interaction Data

| **Feature** | **Description**                                                                                        |
|------------|--------------------------------------------------------------------------------------------------------|
| `contact`  | Contact communication type                                                                             |
| `day`      | Date digit of last contact                                                                             |
| `month`    | Last contact month of year                                                                             |
| `duration` | Last contact duration, in seconds
#### Historical Campaign Interaction Data

| **Feature** | **Description**                                                                                        |
|------------|--------------------------------------------------------------------------------------------------------|
| `campaign` | Number of contacts performed during this campaign and for this client                                   |
| `pdays`    | Number of days since the client was last contacted from a previous campaign                              |
| `previous` | Number of contacts performed before this campaign and for this client                                   |
| `poutcome` | Outcome of the previous marketing campaign                                                             |
#### Output Target

| **Feature** | **Description**                                                                                        |
|------------|--------------------------------------------------------------------------------------------------------|
| `subscribed`| Has the client subscribed a term deposit?                                                              |

### Overview
numerical_feats = df.dtypes[df.dtypes != "object"].index
categorical_feats = df.dtypes[df.dtypes == "object"].index

print("Jumlah fitur numerik:", len(numerical_feats))
print("Jumlah fitur kategorikal:", len(categorical_feats))
list_item = []
for col in df.columns:
    list_item.append([col, df[col].dtype, df[col].isna().sum(), round(100*df[col].isna().sum()/len(df[col]), 2), df[col].nunique(), df[col].unique()[:5]])
desc_df = pd.DataFrame(data=list_item, columns='feature, data_type, null_values, null_percentage, unique_values, unique_sample'.split(","))
desc_df
for feature in categorical_feats:
    print("{} have {} unique values".format(feature, df[feature].nunique()))
    print("{} values: {}".format(feature, df[feature].unique()))
    print('-' * 100)
df.isna().sum().sum()
df['ID'].duplicated().sum()
**Key Takeaways** :

- Dataset memeiliki 31647 records dan 18 fitur
- 8 fitur bertipe numerik dan 10 fitur bertipe kategorikal
- Tipe data pada feature yang ada berupa: object dan int64
- Tidak ada fitur yang memiliki missing value
- Tidak ada data duplikat
# Data Cleaning
Menghapus kolom `ID` dari DataFrame. Ini dilakukan karena kolom `ID` tidak diperlukan dalam analisis atau pemodelan selanjutnya, karena hanya merupakan identifikasi unik untuk setiap entri dan tidak memberikan informasi yang relevan untuk tujuan analisis.
df = df.drop(columns=['ID'], axis=1)
numerical_feats = numerical_feats.drop('ID')
# Exploratory Data Analysis
df_eda = df.copy()
## Univariate Analysis
Analisis statistik yang memeriksa satu variabel tunggal dalam sebuah dataset untuk memahami karakteristik dasarnya, seperti distribusi dan tendensi sentral
### Function
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)

def num_plot(df, col, figsize=(8, 5)):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize,gridspec_kw={"height_ratios": (.2, .8)})
    ax[0].set_title(col + ' Distribution', fontsize=18)
    sns.boxplot(x=col, data=df, ax=ax[0],color = "#1D8EF5")
    ax[0].set(yticks=[])
    sns.histplot(x=col, data=df, ax=ax[1],color = "#33AAFF", edgecolor="#1D1EA2")
    ax[1].set_xlabel(col, fontsize=16)
    plt.axvline(df[col].mean(), color='darkgreen', linestyle='--',linewidth=2.2, label='mean=' + str(np.round(df[col].mean(),1)))
    plt.axvline(df[col].median(), color='red', linestyle='--',linewidth=2.2, label='median='+ str(np.round(df[col].median(),1)))
    plt.axvline(df[col].mode()[0], color='purple', linestyle='--',linewidth=2.2, label='mode='+ str(np.round(df[col].mode()[0],1)))

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    plt.axvline(q1, color='orange', linestyle='--',linewidth=2.2, label='Q1=' + str(np.round(q1,1)))
    plt.axvline(q3, color='blue', linestyle='--',linewidth=2.2, label='Q3='+ str(np.round(q3,1)))

    plt.legend(bbox_to_anchor=(1, 1.03), ncol=1, fontsize=17, fancybox=True, shadow=True, frameon=True)
    plt.tight_layout()
    plt.show()
def top_freq_percentage(dataset, feats):
  top_freq_percentages = {}
  top_categories = {}

  for feature in feats:
      value_counts = dataset[feature].value_counts()
      top_category = value_counts.idxmax()

      top_freq_percentage = (value_counts[top_category] / len(dataset)) * 100
      top_freq_percentages[feature] = top_freq_percentage
      top_categories[feature] = top_category

  top_freq_percentages_df = pd.DataFrame.from_dict(top_freq_percentages, orient='index', columns=['top_frequency_percentage']).sort_values('top_frequency_percentage', ascending=False)
  top_categories_df = pd.DataFrame.from_dict(top_categories, orient='index', columns=['top_category'])

  result_df = pd.concat([top_freq_percentages_df, top_categories_df], axis=1)
  return result_df
def calculate_outliers(column):
  # Calculate the IQR
  Q1 = column.quantile(0.25)
  Q3 = column.quantile(0.75)
  IQR = Q3 - Q1

  # Define lower and upper bounds
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Filter the column to count outliers
  outliers = column[(column < lower_bound) | (column > upper_bound)]

  # Calculate percentage of outliers
  percentage_outliers = (outliers.shape[0] / column.shape[0]) * 100

  return outliers.shape[0], percentage_outliers
### Numerical Feature
df_eda[numerical_feats].describe()
result_df = top_freq_percentage(df_eda, numerical_feats)
result_df
player_not_contacted_count = len(df_eda[df_eda['pdays'] == -1])
player_not_contacted_percentage = round(player_not_contacted_count / df_eda.shape[0] * 100, 2)
print("Client yang tidak dikontak pada campaign sebelumnya:")
print(f'{player_not_contacted_count} ({player_not_contacted_percentage}%)')
print()

negative_saldo_count = len(df_eda[df_eda['balance'] < 0])
negative_saldo_percentage = round(negative_saldo_count / df_eda.shape[0] * 100, 2)
print("Client yang memiliki saldo negatif:")
print(f'{negative_saldo_count} ({negative_saldo_percentage}%)')
#### Outliers

Outlier diidentifikasi menggunakn perhitung IQR. IQR dihitung sebagai selisih antara kuartil atas (Q3) dan kuartil bawah (Q1). Pada proses ini, data dianggap sebagai outlier jika nilainya di bawah Q1 - 1.5 * IQR atau di atas Q3 + 1.5 * IQR.
outliers_info = []

for feature in numerical_feats:
  num_outliers, percentage_outliers = calculate_outliers(df_eda[feature])

  if percentage_outliers > 0:
    outliers_info.append({
        'feature': feature,
        'outlier_count': num_outliers,
        'percentage_outliers': percentage_outliers
    })

outliers_df = pd.DataFrame(outliers_info).sort_values(by='percentage_outliers', ascending=False)
outliers_df
#### Skewness and Kurtosis

- **Skewness**: ukuran seberapa simetris atau tidak simetris distribusi data
- **Kurtosis**: adalah ukuran tingkat kecuraman atau tumpukan data di sekitar ekor distribusi.
results = []
for col in numerical_feats:
    skewness = df_eda[col].skew()
    kurtosis = df_eda[col].kurt()

    # Determine skewness type
    if skewness > 0:
        skew_type = 'positive'
    elif skewness < 0:
        skew_type = 'negative'
    else:
        skew_type = 'no skew'
    results.append({'Column': col, 'Skewness': skewness, 'Kurtosis': kurtosis, 'Skew_Type': skew_type})

results_df = pd.DataFrame(results)
results_df = results_df.reindex(results_df['Skewness'].abs().sort_values(ascending=False).index)
results_df
#### Variation and Central Tendencies

- **Variation**: Menggambarkan seberapa jauh data tersebar dari nilai pusatnya, memberikan informasi tentang keragaman atau variasi nilai dalam dataset.
- **Central Tendencies**: Merupakan ukuran yang digunakan untuk merepresentasikan nilai pusat dalam dataset, yang membantu dalam pemahaman karakteristik utama dari distribusi data.
num_plot(df_eda, 'age')
num_plot(df_eda, 'balance')
num_plot(df_eda, 'day')
num_plot(df_eda, 'duration')
num_plot(df_eda, 'campaign')
num_plot(df_eda, 'pdays')
num_plot(df_eda, 'previous')
**Key Takeaways** :

1. Segmentasi Pelanggan
- Rata-rata usia client adalah 41 tahun dengan saldo rata-rata 1364. Terdapat variasi besar dalam saldo client, yang mengindikasikan adanya perbedaan signifikan dalam kepemilikan keuangan.

2. Saldo Negatif
- Terdapat 8.42% client dengan saldo negatif, menandakan risiko kredit yang perlu dipantau oleh bank untuk memastikan kestabilan keuangan. Diperlukan manajemen risiko yang efektif untuk mengelola potensi risiko kredit ini.

3. Kontak Sebelumnya
- Terdapat 81.92% client tidak memiliki kontak sebelumnya (pdays = -1), menunjukkan potensi untuk meningkatkan interaksi dengan pelanggan.

4. Durasi panggilan
- Rata-rata waktu panggilan adalah 4 menit, dengan kebanyakan panggilan singkat. Sebagian besar klien, sekitar 75%, telah dihubungi sebanyak 2-3 kali selama kampanye.
### Categorical Feature
df[categorical_feats].describe()
result_df = top_freq_percentage(df_eda, categorical_feats)
result_df
**Key Takeaways** :

1. Mayoritas Pelanggan Tidak Berlangganan
- Mayoritas pelanggan (88.26%) tidak berlangganan. Hal ini menunjukan adanya imbalanced data yang perlu ditangani lebih lanjut.

2. Dominasi Unique Feature
- Selain variabel target, ada beberapa fitur yang memiliki jumlah nilai unik yang sangat mendominasi, yaitu default (98.151%), pinjaman (83.787%), dan poutcome (81.932%). Penting untuk menginvestigasi lebih lanjut apakah keberadaan fitur-fitur ini berdampak pada label target karena kurangnya variasi dalam fitur tersebut.

3. Pembayaran Lain
- Lebih banyak client bank yang melakukan subscribe pada Housing Loan (55.56%), daripada Credit Default (1.84%) atau Personal Loan (16.21%)
## Bivariate Analysis
Analisa hubungan antara dua variabel dalam dataset untuk memahami korelasi atau asosiasi di antara keduanya. Variabel kedua akan berfokus pada target.
### Numerical Feature
plt.figure(figsize=(14, 10))
for i in range(0, len(numerical_feats)):
    plt.subplot(3, 3, i+1)
    sns.kdeplot(df_eda, x=df_eda[numerical_feats[i]], hue=df_eda['subscribed'], fill=True).set(title=f'{numerical_feats[i]} Distribution Based Subscribed')
    plt.tight_layout()
**Key Takeaways** :

- Distribusi fitur numerik tidak dapat secara jelas membedakan apakah pelanggan akan berlangganan atau tidak, seperti yang terlihat dari hasil KDE Plot, distribusi "subscribe yes" dan "subscribe no" saling tumpang tindih.
### Categorical Feature
def plot_subscription_counts(feature, df, figsize=(7, 5), custom_order=False):
    plt.figure(figsize=figsize)

    # Calculate counts for each unique value of the feature
    feature_counts = df.groupby(feature)['subscribed'].value_counts().unstack(fill_value=0)

    # Sort by total counts
    sorted_index = feature_counts.sum(axis=1).sort_values(ascending=False).index
    feature_counts = feature_counts.loc[sorted_index]

    # Calculate percentages
    total_counts = feature_counts.sum(axis=1)
    yes_percentages = feature_counts['yes'] / total_counts * 100
    no_percentages = feature_counts['no'] / total_counts * 100

    # Define colors with transparency
    yes_color = sns.color_palette()[0] + (0.5,)
    no_color = sns.color_palette()[1] + (0.5,)
    edge_color = 'lightgray'  # Change the color here

    if custom_order:
        custom_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        # Reindex feature_counts based on custom_order
        feature_counts = feature_counts.reindex(custom_order)

    # Plot yes percentages with outline stroke
    plt.bar(feature_counts.index, yes_percentages, color=yes_color, label='Yes', width=0.4, edgecolor=edge_color)

    # Plot no percentages with outline stroke
    plt.bar(feature_counts.index, no_percentages, bottom=yes_percentages, color=no_color, label='No', width=0.4, edgecolor=edge_color)

    plt.title(f"Subscription Counts by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Percentage")
    plt.legend(title='Subscribed', labels=['Yes', 'No'])
    plt.xticks(rotation=45)

    # Display percentages at the center of each bar
    for idx, (value, yes_percentage, no_percentage) in enumerate(zip(feature_counts.index, yes_percentages, no_percentages)):
        plt.text(idx, yes_percentage / 2, f'{yes_percentage:.1f}%', ha='center', va='center')
        plt.text(idx, yes_percentage + no_percentage / 2, f'{no_percentage:.1f}%', ha='center', va='center')

    # Print total counts of each column after reindexing
    for idx, (index, count) in enumerate(zip(sorted_index, total_counts), start=1):
        print(f"{idx}. {index}: {count}")
    print()

    plt.grid(False)  # Remove grid lines
    plt.show()
plot_subscription_counts('default', df_eda)
plot_subscription_counts('loan', df_eda)
plot_subscription_counts('housing', df_eda)
**Key Takeaways** :

- Berdasarkan data persentase, terlihat tren menarik dimana lebih banyak klien yang berlangganan term deposit saat mereka tidak memiliki langganan lain seperti kredit, pinjaman rumah, atau pinjaman pribadi di bank. Kemungkinan, klien tanpa langganan lain cenderung lebih tertarik pada investasi jangka panjang seperti term deposit karena tidak ada tanggungan lain yang mengikat mereka.





plot_subscription_counts('poutcome', df_eda)
**Key Takeaways** :

- Berdasarkan data persentase, terdapat perbedaan signifikan dalam respons klien terhadap kampanye sebelumnya. Status kesuksesan memiliki persentase langganan 'yes' yang jauh lebih tinggi dibandingkan dengan status hasil lainnya, dengan nilai mencapai 65%.







plot_subscription_counts('contact', df_eda)
**Key Takeaways** :

- Komunikasi melalui perangkat seluler memiliki tingkat langganan tertinggi sebesar 15%, menekankan pentingnya mengutamakan platform seluler dalam strategi pemasaran bank.





plot_subscription_counts('marital', df_eda)
**Key Takeaways** :

- Marital status 'single' menunjukkan tingkat langganan tertinggi sebesar 15.1%, kemungkinan karena individu single cenderung memiliki tanggung jawab keuangan yang lebih sedikit dibandingkan dengan mereka yang sudah menikah atau memiliki tanggungan keluarga.


plot_subscription_counts('education', df_eda)
plot_subscription_counts('month', df_eda, (22,5), True)
**Key Takeaways** :

- Langganan term deposit mencapai persentase tertinggi pada kuartal terakhir tahun, melebihi 40%. Hal ini mungkin terjadi karena akhir tahun sering kali menjadi waktu di mana individu membuat keputusan keuangan besar, seperti perencanaan untuk tahun depan atau alokasi bonus tahunan, yang dapat mendorong minat mereka dalam investasi jangka panjang seperti term deposit.
plot_subscription_counts('job', df_eda, (22,5))
**Key Takeaways** :

- Secara persentase, langganan term deposit mencapai tingkat tertinggi di antara pekerjaan mahasiswa atau pensiunan, melampaui 20%. Hal ini terjadi mungkin karena mereka memiliki waktu dan kecenderungan untuk mengelola keuangan dengan lebih hati-hati atau memiliki kebutuhan investasi jangka panjang.
## Multivariate Analysis
# Create a dictionary to map 'no' to 0 and 'yes' to 1
subscribed_mapping = {'no': 0, 'yes': 1}

# Apply the mapping to the 'subscribed' column and store it in a new variable
subscribed_encoded = df_eda['subscribed'].map(subscribed_mapping)

# Create a new DataFrame with numerical features and the encoded 'subscribed'
combined_df = pd.concat([df_eda[numerical_feats], subscribed_encoded], axis=1)
sns.pairplot(combined_df,hue='subscribed',corner=True)
**Key Takeaways** :

- Tidak terdapat pola khusus yang dapat diambil karena kdeplot antara masing masing feature dengan target sebagai hue saling tumpang tindih dan tidak terpisah.
# Calculate the correlation matrix
correlation_matrix = combined_df.corr()

# Create a mask to hide the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# Plot the heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, cmap='Blues', annot=True, fmt='.2f', mask=mask)
plt.title('Correlation Matrix of Numerical Features')
plt.grid(False)  # Remove grid lines
plt.show()
**Key Takeaways** :

- Tidak ada feature yang memiliki multicolinearity
- Fitur dengan kolerasi linear yang cukup tinggi dengan target adalah duration dan pdays. Namun karena fitur hanya sedikit, penggunaan semua fitur bisa dipertimbangkan meskipun memiliki korelasi linear yang rendah
# Feature Selection
Feature selection adalah proses pemilihan subset fitur yang paling relevan dari sekumpulan fitur yang tersedia dalam data untuk digunakan dalam model machine learning
### Numerical Feature
Relevansi suatu fitur numerical terhadap target akan dikalkulasi menggunakn point biserial. Point biserial adalah metode statistik yang digunakan untuk mengukur hubungan antara variabel biner dan variabel numerik.
# Drop the 'subscribed' column from the DataFrame
numerical_features = combined_df.drop(columns=['subscribed'])

# Create an empty DataFrame to store the correlation coefficients
correlation_matrix = pd.DataFrame(index=numerical_features.columns, columns=['Point_Biserial_Correlation'])

# Iterate over each numerical feature
for feature in numerical_features.columns:
    # Compute the point biserial correlation coefficient with respect to the target variable
    pb_corr, _ = pointbiserialr(numerical_features[feature], combined_df['subscribed'])
    # Store the correlation coefficient in the DataFrame
    correlation_matrix.loc[feature, 'Point_Biserial_Correlation'] = pb_corr

# Convert the dtype of the correlation coefficients to float64
correlation_matrix = correlation_matrix.astype('float64')

# Sort the correlation matrix by correlation values
correlation_matrix = correlation_matrix.sort_values(by='Point_Biserial_Correlation', ascending=False)

# Plot the heatmap
plt.figure(figsize=(5, 6))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Point Biserial Correlation Heatmap')
plt.xlabel('Target Variable: subscribed')
plt.ylabel('Numerical Features')
plt.show()
**Key Takeaways** :

- Feature numerik dengan korelasi yang cukup tinggi adalah duration (39%) dan pdays (11%). Namun penggunaan seluruh fitur dapat dipertambangkan karena fitur yang tidak terlalu banyak.
### Categorical Feature
Relevansi suatu fitur kategorikal terhadap target akan dikalkulasi menggunakn Cramér's V dan P-value

- Cramér's V adalah metrik yang mengukur kekuatan hubungan antara dua variabel kategorikal dengan nilai berkisar antara 0 hingga 1. Nilai yang lebih tinggi menunjukkan hubungan yang lebih kuat antara variabel tersebut.
- P-value adalah ukuran statistik yang menentukan signifikansi dari hasil uji hipotesis. Nilai P yang rendah menunjukkan bahwa ada bukti yang kuat menentang hipotesis nol, sementara nilai P yang tinggi menunjukkan tidak cukup bukti untuk menolak hipotesis nol.
def categorical_stats(confusion_matrix):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    if isinstance(confusion_matrix, pd.DataFrame):
        confusion_matrix = confusion_matrix.to_numpy()  # Convert DataFrame to numpy array

    if confusion_matrix.shape[0] == 1 or confusion_matrix.shape[1] == 1:
        # Handle edge case where one of the dimensions is 1
        return np.nan, np.nan

    chi2, p_value, _, _ = ss.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))), p_value
results = []
for col in categorical_feats:
  if col != 'subscribed':
    confusion_matrix = pd.crosstab(df_eda[col], df_eda['subscribed'])
    cramers_v, p_value = categorical_stats(confusion_matrix)
    results.append({'Column': col, 'Cramers_V': cramers_v, 'P_value': p_value})

# Define thresholds for significance
p_value_threshold = 0.05
cramers_v_threshold = 0.1

# Function to determine significance based on thresholds
def is_significant(p_value, cramers_v):
    return (p_value < p_value_threshold) and (cramers_v > cramers_v_threshold)

results_df = pd.DataFrame(results)
# Add a new column to indicate significance
results_df['Significance & Correlated'] = results_df.apply(lambda row: is_significant(row['P_value'], row['Cramers_V']), axis=1)
results_df = results_df.reindex(results_df['Cramers_V'].sort_values(ascending=False).index)
results_df
Based on Dai et al. (2021):

- Weak: Cramér's V > 0.05
- Moderate: Cramér's V > 0.10
- Strong: Cramér's V > 0.15
- Very Strong: Cramér's V > 0.25

Ref:
Dai, J., Teng, L., Zhao, L., & Zou, H. (2021). The combined analgesic effect of pregabalin and morphine in the treatment of pancreatic cancer pain, a retrospective study. Cancer Medicine, 10(5), 1738–1744. https://doi.org/10.1002/cam4.3779
# Sort the correlation matrix by correlation values
correlation_matrix = results_df[['Column', 'Cramers_V']].sort_values(by='Cramers_V', ascending=False)

# Set the "Column" column as the index and drop it
correlation_matrix.set_index('Column', drop=True, inplace=True)

# Plot the heatmap
plt.figure(figsize=(5, 7))
sns.heatmap(correlation_matrix[['Cramers_V']], cmap='coolwarm', annot=True, fmt=".3f", linewidths=0.5)
plt.title('Cramér\'s V Heatmap')
plt.xlabel('Categorical Features')
plt.ylabel('Categorical Features')

plt.show()
**Key Takeaways** :

- Feature numerik dengan korelasi yang cukup tinggi adalah poutcome (31%), month (25%), contact (15%), housing (14%), dan job (13%). Namun penggunaan seluruh fitur dapat dipertambangkan karena fitur yang tidak terlalu banyak.
# Data Preprocessing
df_prep = df_eda.copy()
## Label Encoding

---


Proses konversi nilai-nilai dalam suatu fitur kategokal ordinal menjadi nilai numerik yang terurut secara bertingkat.
# Education
education_mapping = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3}

df_prep['education'] = df_prep['education'].map(education_mapping)
df_prep[['education']].sample(3)
# Default, Housing, Loan, Subscribed
binary_mapping = {'no': 0, 'yes': 1}

df_prep['default'] = df_prep['default'].map(binary_mapping)
df_prep['housing'] = df_prep['housing'].map(binary_mapping)
df_prep['loan'] = df_prep['loan'].map(binary_mapping)
df_prep['subscribed'] = df_prep['subscribed'].map(binary_mapping)

df_prep[['default', 'housing', 'loan', 'subscribed']].sample(3)
# Month
month_mapping = {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11}

df_prep['month'] = df_prep['month'].map(month_mapping)
df_prep[['month']].sample(3)
# Poutcome

poutcome_mapping = {"failure": 0, "unknown": 1, "other": 2, "success": 3}
df_prep['poutcome'] = df_prep['poutcome'].map(poutcome_mapping)
df_prep[['poutcome']].sample(3)
## One-Hot Encoding

---


Proses konversi nilai-nilai dalam suatu fitur kategorikal nominal menjadi vektor biner yang dapat diinterpretasikan oleh model machine learning
# Contact, Marital, Job

# List of features to one-hot encode
features_to_encode = ['contact', 'marital', 'job']

# Perform one-hot encoding for each feature
for feature in features_to_encode:
    # Perform one-hot encoding
    df_prep = pd.get_dummies(df_prep, columns=[feature])
df_prep.head()
df_prep.info()
df_prep.to_csv('./preprocessed_data.csv', header=False)
df_prep = pd.read_csv('/content/Predictive-Modeling-for-Term-Deposit-Subscription-in-Bank-Marketing/dataset/preprocessed_data.csv')
# Modeling
## Preparation

---


X = df_prep.drop(columns=['subscribed'])
y = df_prep[['subscribed']]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
def eval_classification(model, X_train, X_test, y_train, y_test, n_splits=5):
    # Evaluate on the test set
    y_pred_test = model.predict(X_test)

    # StratifiedKFold for cross-validation with stratified sampling
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_test_results = cross_validate(model, X_test, y_test, scoring=['roc_auc'],
                                     cv=cv, return_train_score=False)
    cv_train_results = cross_validate(model, X_train, y_train, scoring=['roc_auc'],
                                      cv=cv, return_train_score=False)


    # Calculate metrics for the test set
    precision_test = round(precision_score(y_test, y_pred_test),2)
    recall_test = round(recall_score(y_test, y_pred_test),2)
    f1_test = round(f1_score(y_test, y_pred_test),2)

    # Display metrics for the training set
    print("Metrics for the Test Set:")
    print("Precision: %.2f" % precision_test)
    print("Recall: %.2f" % recall_test)
    print("F1-Score: %.2f" % f1_test)
    print()

    # Display cross-validation results
    print("Metrics Using Cross Validation:")
    print(f"Mean ROC-AUC (Test): {cv_test_results['test_roc_auc'].mean():.2f}")
    print(f"Std ROC-AUC (Test): {cv_test_results['test_roc_auc'].std():.2f}")
    print()
    print(f"Mean ROC-AUC (Train): {cv_train_results['test_roc_auc'].mean():.2f}")
    print(f"Std ROC-AUC (Train): {cv_train_results['test_roc_auc'].std():.2f}")

    return precision_test, recall_test, f1_test
# Save the trained model to a specific path
MODEL_PATH = '/content/base_model/'
## Logistic Regression
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
log_pred = log_clf.predict(X_test)
joblib.dump(log_clf, MODEL_PATH + 'log.pkl')

precision_log, recall_log, f1_log = eval_classification(log_clf, X_train, X_test, y_train, y_test)
## KNN

---


knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test)
joblib.dump(knn_clf, MODEL_PATH + 'knn.pkl')

precision_knn, recall_knn, f1_knn = eval_classification(knn_clf, X_train, X_test, y_train, y_test)
## SVM

---


svc_clf = SVC()
svc_clf.fit(X_train, y_train)
svc_pred = svc_clf.predict(X_test)
joblib.dump(svc_clf, MODEL_PATH + 'svc.pkl')

precision_svc, recall_svc, f1_svc = eval_classification(svc_clf, X_train, X_test, y_train, y_test)
## Decision Tree

---


dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
joblib.dump(dt_clf, MODEL_PATH + 'dt.pkl')

precision_dt, recall_dt, f1_dt = eval_classification(dt_clf, X_train, X_test, y_train, y_test)
## Random Forest

---


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
joblib.dump(rf_clf, MODEL_PATH + 'rf.pkl')

precision_rf, recall_rf, f1_rf = eval_classification(rf_clf, X_train, X_test, y_train, y_test)
## Gaussian Naive Bayes

---


gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)
gnb_pred = gnb_clf.predict(X_test)
joblib.dump(gnb_clf, MODEL_PATH + 'gnb.pkl')

precision_gnb, recall_gnb, f1_gnb = eval_classification(gnb_clf, X_train, X_test, y_train, y_test)
## XGBoost

---


xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)
xgb_pred = xgb_clf.predict(X_test)
joblib.dump(xgb_clf, MODEL_PATH + 'xgb.pkl')

precision_xgb, recall_xgb, f1_xgb = eval_classification(xgb_clf, X_train, X_test, y_train, y_test)
## Gradient Boosting

---


gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
joblib.dump(gb_clf, MODEL_PATH + 'gb.pkl')

precision_gb, recall_gb, f1_gb = eval_classification(gb_clf, X_train, X_test, y_train, y_test)
## LightGBM

---


lgbm_clf = LGBMClassifier()
lgbm_clf.fit(X_train, y_train)
lgbm_pred = lgbm_clf.predict(X_test)
joblib.dump(lgbm_clf, MODEL_PATH + 'lgbm.pkl')

precision_lgbm, recall_lgbm, f1_lgbm = eval_classification(lgbm_clf, X_train, X_test, y_train, y_test)
## CatBoost

---


cb_clf = CatBoostClassifier()
cb_clf.fit(X_train, y_train)
cb_pred = cb_clf.predict(X_test)
joblib.dump(cb_clf, MODEL_PATH + 'cb.pkl')

precision_cb, recall_cb, f1_cb = eval_classification(cb_clf, X_train, X_test, y_train, y_test)
## Adaboost

---


ab_clf = AdaBoostClassifier()
ab_clf.fit(X_train, y_train)
ab_pred = ab_clf.predict(X_test)
joblib.dump(ab_clf, MODEL_PATH + 'ab.pkl')

precision_ab, recall_ab, f1_ab = eval_classification(ab_clf, X_train, X_test, y_train, y_test)
# Evaluation
## Model Comparison

---


# Define model names and corresponding metrics
models = ['Logistic Regression', 'KNN', 'SVC', 'Decision Tree', 'Random Forest',
          'Gaussian NB', 'XGBoost', 'Gradient Boosting', 'LGBM', 'CatBoost', 'AdaBoost']
precisions = [precision_log, precision_knn, precision_svc, precision_dt, precision_rf,
              precision_gnb, precision_xgb, precision_gb, precision_lgbm, precision_cb, precision_ab]
recalls = [recall_log, recall_knn, recall_svc, recall_dt, recall_rf,
           recall_gnb, recall_xgb, recall_gb, recall_lgbm, recall_cb, recall_ab]
f1_scores = [f1_log, f1_knn, f1_svc, f1_dt, f1_rf, f1_gnb, f1_xgb, f1_gb, f1_lgbm, f1_cb, f1_ab]

# Create DataFrame
metrics_df = pd.DataFrame({'Model': models, 'Precision': precisions, 'Recall': recalls, 'F1-Score': f1_scores}).sort_values('F1-Score', ascending=False)

# Display the DataFrame
metrics_df
LightGBM (LGBM) akan dipilih sebagai model yang digunakan didasarkan pada nilai F1-score tertinggi dibandingkan dengan model-model lainnya dalam daftar. F1-score adalah ukuran gabungan dari precision dan recall, yang berarti nilai F1-score yang tinggi menunjukkan keseimbangan antara kedua metrik tersebut.
## Optimization

---


# Define the LightGBM model
def create_lgbm_model(n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree):
    model = LGBMClassifier(n_estimators=int(n_estimators),
                           learning_rate=learning_rate,
                           max_depth=int(max_depth),
                           min_child_weight=min_child_weight,
                           subsample=subsample,
                           colsample_bytree=colsample_bytree,
                           random_state=42)
    return model
# Define the objective function to optimize
def objective(n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree):
    model = create_lgbm_model(n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='binary')
    return f1
# Define the search space for hyperparameters
pbounds = {'n_estimators': (50, 500),
          'learning_rate': (0.01, 0.3),
          'max_depth': (3, 15),
          'min_child_weight': (1, 10),
          'subsample': (0.5, 1),
          'colsample_bytree': (0.5, 1)}
# Perform Bayesian optimization
optimizer = BayesianOptimization(f=objective, pbounds=pbounds, verbose=2)
optimizer.maximize(init_points=5, n_iter=100)
# Print the optimized hyperparameters and F1 Score
print('Optimized hyperparameters:')
print(optimizer.max['params'])
print('F1 Score: {:.2f}%'.format(optimizer.max['target'] * 100))
## Model Result

---


MODEL_PATH = '/content/best_model/'
# Define the optimized hyperparameters
optimized_params = {
    'colsample_bytree': 0.9189486441541788,
    'learning_rate': 0.05313304063365639,
    'max_depth': int(14.708706776984497),
    'min_child_weight': 6.64071495254778,
    'n_estimators': int(350.33109950160144),
    'subsample': 0.7208687584446896,
    'num_leaves': 31
}
best_lgbm_clf = LGBMClassifier(**optimized_params)
best_lgbm_clf.fit(X_train, y_train)
lgbm_pred = best_lgbm_clf.predict(X_test)
joblib.dump(best_lgbm_clf, MODEL_PATH + 'lgbm.pkl')
y_pred = best_lgbm_clf.predict(X_test)
# Calculate precision, recall, and F1-score
precision_best_lgbm = precision_score(y_test, y_pred)
recall_best_lgbm = recall_score(y_test, y_pred)
f1_best_lgbm = f1_score(y_test, y_pred)

print("Precision (LightGBM): {:.2f}".format(precision_lgbm))
print("Recall (LightGBM): {:.2f}".format(recall_lgbm))
print("F1 Score (LightGBM): {:.2f}".format(f1_lgbm))
**Key Takeaways** :

- Optimisasi parameter pada model LightGBM tidak menghasilkan peningkatan,dengan nilai Precision, Recall, dan F1 Score tidak mengalami perubahan yang signifikan. Namun penggunaan model setelah optimization akan tetap dilakukan
## Feature Importance

---


Feature importance akan menggunakan SHAP untuk memberikan wawasan tentang pengaruh setiap fitur terhadap prediksi model, dengan nilai antara -1 hingga 1. Nilai positif menunjukkan kontribusi positif terhadap prediksi, sedangkan nilai negatif menunjukkan kontribusi negatif. Semakin besar nilai absolut, semakin besar pengaruhnya terhadap prediksi.
# Remove the "subscribed" column from the DataFrame
df_feat = df_prep.drop(columns=["subscribed"])
# Calculate SHAP values
explainer = shap.TreeExplainer(best_lgbm_clf)
shap_values = explainer.shap_values(X_test)
# Summarize the effects of features
shap.summary_plot(shap_values, X_test, feature_names=list(df_feat.columns))
**Key Takeaways** :

Dari hasil feature importance, top 5 fitur yang paling memengaruhi prediksi model adalah
- duration (Positive)
- month (Negative)
- contact_unknown (Negative)
- contact_celular (Positive)
- housing (Negative)
## Error Analysis

---


subscribed = df_prep.pop('subscribed')
df_prep['subscribed'] = subscribed

X_columns = df_prep.drop('subscribed', axis=1).columns
# Convert X_train and X_test to DataFrames
df_X_train = pd.DataFrame(X_train, columns=X_columns)
df_X_test = pd.DataFrame(X_test, columns=X_columns)

# Create DataFrames for y_train, y_test, and y_pred
df_y_train = pd.DataFrame(y_train, columns=['subscribed'])
df_y_test = pd.DataFrame(y_test, columns=['subscribed'])

# Reset indices
df_X_train.reset_index(drop=True, inplace=True)
df_X_test.reset_index(drop=True, inplace=True)
df_y_train.reset_index(drop=True, inplace=True)
df_y_test.reset_index(drop=True, inplace=True)

# Combine all DataFrames into one DataFrame for training and testing
df_mod_train = pd.concat([df_X_train, df_y_train], axis=1)
df_mod_test = pd.concat([df_X_test, df_y_test], axis=1)

# Combine training and testing DataFrames into one
df_mod = pd.concat([df_mod_train, df_mod_test], axis=0)

df_mod.head()
# Make predictions on all data in df_mod without 'subscribed' column
y_pred_all = best_lgbm_clf.predict(df_mod.drop('subscribed', axis=1))

# Add the predicted values as a new column to df_mod
df_mod['predicted_label'] = y_pred_all
df_mod.head()
df_err = df_mod[df_mod['subscribed'] != df_mod['predicted_label']]
df_err.head()
# Calculate the number of rows and columns needed
num_features = len(df_err.columns[:-2])  # Exclude target and predicted label
num_cols = 3
num_rows = math.ceil(num_features / num_cols)

# Create a grid of subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot KDE plots for each feature
for i, feature in enumerate(df_err.columns[:-2]):  # Exclude target and predicted label
    ax = axes[i]
    sns.kdeplot(data=df_err[feature], color='blue', label='Misclassified', shade=True, ax=ax)
    ax.set_title(f'KDE Plot of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')

# Hide empty subplots
for j in range(num_features, num_rows * num_cols):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
**Key Takeaways** :

Pada Error Analysis, model gagal mengklasifikasikan data tertentu, yaitu:
- Pengguna yang tidak memiliki kredit bermasalah (default credit).
- Pengguna yang tidak berlangganan pinjaman pribadi (personal loan).
- Pengguna yang dihubungi melalui ponsel seluler (contact celular).
# Recommendation
## Technical

---


- Mencari dataset tambahan yang lebih bervariasi untuk menambah variasi dan representasi yang lebih baik dari populasi pengguna.
- Lakukan peninjauan mendalam terhadap fitur-fitur yang digunakan dalam model untuk memastikan relevansi dan keterkaitannya dengan target yang diinginkan.
- Eksplorasi lebih lanjut terhadap teknik pemrosesan data yang dapat meningkatkan kinerja model.
## Business

---


- Manfaatkan informasi dari model untuk menyusun kampanye pemasaran yang lebih terarah, dengan menargetkan pelanggan yang memiliki kemungkinan tinggi untuk berlangganan.
- Tingkatkan strategi retensi pelanggan dengan memanfaatkan wawasan dari model untuk memahami faktor-faktor yang memengaruhi keputusan pelanggan dalam berlangganan.
