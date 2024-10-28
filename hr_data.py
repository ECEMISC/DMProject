import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import warnings
import missingno as msno
from datetime import date
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import recall_score, precision_score, f1_score
from catboost import CatBoostClassifier, Pool, cv
from xgboost import XGBClassifier
import missingno as mo
from sklearn.preprocessing import LabelEncoder, RobustScaler
from tabulate import tabulate
from scipy.stats import randint
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

warnings.simplefilter("ignore")
pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)

df = pd.read_csv("/Users/ecemzeynepiscanli/PycharmProjects/DMProject/IBM_Attrition.csv")
df.head()

Y=df[['Attrition']]
X=df.drop(['Attrition'],axis=1)

constant_columns = [col for col in X.columns if X[col].nunique() == 1]
print("Sabit sütunlar:", constant_columns)

df.drop(['EmployeeCount','Over18', 'StandardHours' ], axis=1, inplace=True)

df.drop(['EmployeeNumber'], axis=1, inplace=True)

####################################
def check_df(dataframe, head=5):
    # Shape
    shape_table = [["Shape", dataframe.shape]]

    # Types
    types_table = dataframe.dtypes.reset_index()
    types_table.columns = ['Column', 'Type']


    # NA
    na_table = dataframe.isnull().sum().reset_index()
    na_table.columns = ['Column', 'NA Count']

    # Quantiles
    quantiles_table = dataframe.describe([0.01, 0.05, 0.50, 0.95, 0.99]).T.reset_index()
    quantiles_table.columns = ['Column', 'Count', 'Mean', 'Std', 'Min', '1%', '5%', '50%', '95%', '99%', 'Max']

    # Print tables
    print("\n##################### Shape #####################\n")
    print(tabulate(shape_table, headers=['Metric', 'Value'], tablefmt='psql'))
    print("\n##################### Types #####################\n")
    print(tabulate(types_table, headers='keys', tablefmt='psql', showindex=False))
    print("\n##################### NA #####################\n")
    print(tabulate(na_table, headers='keys', tablefmt='psql', showindex=False))
    print("\n##################### Quantiles #####################\n")
    print(tabulate(quantiles_table, headers='keys', tablefmt='psql', showindex=False))

check_df(df)

##################################################

def grab_col_names(dataframe, cat_th=15, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

grab_col_names(df)

df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)


# 1. Dummy değişkenler oluştur (One-Hot Encoding)
catag_dum = pd.get_dummies(df[cat_cols], drop_first=True)

# 2. Hedef değişkeni (Attrition) zaten sayısal hale getirildiği için ayrıca dönüştürmeye gerek yok
Y = df['Attrition']  # Zaten sayısal

# 3. SelectKBest ile en iyi 20 özelliği seçme
selector = SelectKBest(chi2, k=20)
selector.fit_transform(catag_dum, Y)

# 4. Seçilen sütunları geri al
cols = selector.get_support(indices=True)
X_ca = catag_dum.iloc[:, cols]

# 5. Sayısal sütunlar ile seçilen kategorik sütunları birleştirme (Hedef değişken dahil edilmeden)
X_all = pd.concat([df[num_cols], X_ca], axis=1, join='inner')

# Sonuçların şekline bakalım
print("Birleştirilmiş veri setinin şekli: ", X_all.shape)
print(X_all.head())

##################
# Hedef değişkeni (Y) X_all'a ekle
X_all['Attrition'] = Y

# Majority ve minority sınıflarını ayıralım
df_majority = X_all[X_all['Attrition'] == 0]  # 'No' sınıfı (Attrition == 0)
df_minority = X_all[X_all['Attrition'] == 1]  # 'Yes' sınıfı (Attrition == 1)

# Azınlık sınıfını (Yes) aşırı örnekleme ile çoğaltalım
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # Örnekleri tekrarlı al
                                 n_samples=len(df_majority),  # Çoğunluk sınıfı kadar örnek al
                                 random_state=17)  # Tekrar edilebilirlik için random_state

# Çoğunluk ve azınlık sınıflarını birleştir
X_all_balanced = pd.concat([df_majority, df_minority_upsampled])

# X_all_balanced'ı X_all'a eşitle
X_all = X_all_balanced

# Yeni dengeli veri setini kontrol et
print("Dengelenmiş X_all şekli: ", X_all.shape)

# Y'nin dengeli dağılımını kontrol edelim
print("Y sınıf dağılımı: \n", X_all['Attrition'].value_counts())

X_all.shape

# Boolean değişkenleri 0 ve 1'e çevir
X_all = X_all.astype({col: int for col in X_all.select_dtypes(include='bool').columns})

# Yeni veri setini kontrol edelim
print(X_all.dtypes)

X = X_all  # Sadece num_cols'teki sayısal kolonları kullanıyoruz
y = X_all['Attrition']  # Hedef değişken olarak attrition'ı tanımlıyoruz

#scaling
from sklearn.preprocessing import StandardScaler

# Attrition sütununu çıkartıyoruz çünkü bu hedef değişken
X = X_all.drop('Attrition', axis=1)

# Standart ölçekleyici
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# NumPy dizisini DataFrame'e geri dönüştür
X_all = pd.DataFrame(X_scaled, columns=X.columns)

# Ölçeklenmiş veriyi kontrol et
print(X_all.head())

# İşlenmiş veriyi CSV dosyasına kaydetme
X_all.to_csv('processed_data.csv', index=False)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

print("Shape of Training Data",X_train.shape)
print("Shape of Testing Data",X_test.shape)
print("Attrition Rate in Training Data",y_train.mean())
print("Attrition Rate in Testing Data",y_test.mean())

models = [('KNN',  KNeighborsClassifier()),
          ('Logistic Regression', LogisticRegression(max_iter=1000)),
          ('CART', DecisionTreeClassifier()),
          ('Random Forest', RandomForestClassifier()),
          ('CatBoost', CatBoostClassifier(verbose=False)),
          ('XGBoost', XGBClassifier())]

# For each model, a dictionary is created to store the performance
performances = {}

for name, classifier in models:
    # Train the model
    classifier.fit(X_train.values, y_train)

    #  Obtaining predicted values
    y_pred = classifier.predict(X_test.values)

    # Calculation of metric values
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Adding performance values to the dictionary
    performances[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'ROC AUC': roc_auc}

# Display of the obtained performance values
for name, performance in performances.items():
    print(f"{name}:")
    for metric_name, value in performance.items():
        print(f"  {metric_name}: {round(value, 16)}")
    print()


#train the model
Cat_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
Cat_model.save_model('catboost_model.cbm')
#model making prediction
y_pred = Cat_model.predict(X_test)

model_features = Cat_model.feature_names_
print(model_features)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Feature importance plot fonksiyonu
def plot_importance(cat_model, features, num=len(X_train.columns)):
    # Feature importance'ı CatBoost modelinden al
    feature_imp = pd.DataFrame({
        'Value': cat_model.get_feature_importance(),
        'Feature': features.columns
    })

    # Feature importance'ı görselleştirme
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

# Örnek kullanım:
plot_importance(Cat_model, X_train)
