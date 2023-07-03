
from tkinter import font
from turtle import color
from urllib.request import proxy_bypass
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from IPython import display
from PIL import Image
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import BaggingRegressor, VotingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# Function to load dataset
def load_dataset(dataset_name, date_column=None):
    if dataset_name == "deforestation":
        data_path = input("Enter the file path for data.csv: ")
        mun_path = input("Enter the file path for Counties.csv: ")
        states_path = input("Enter the file path for states.csv: ")

        df = pd.read_csv(data_path)
        mun = pd.read_csv(mun_path, sep=';')
        states = pd.read_csv(states_path)

        return df, mun, states
    
    file_path = input(f"Enter the file path for the {dataset_name} dataset: ")
    if date_column:
        parse_dates = [date_column]
        data = pd.read_csv(file_path, parse_dates=parse_dates)
    else:
        data = pd.read_csv(file_path)
    return data



choice = int(input(
    "Enter your choice (1 for Water Quality, 2 for Air Quality, 3 for Deforestation, 4 for Climate Pattern): "))

# Water Quality Prediction


if choice == 1:
    # Load the dataset
    water_quality_data = load_dataset("water quality")
    # ... (code for water quality prediction using 'water_quality_data')     
    print(water_quality_data.head())
    print(water_quality_data.shape)
    print(water_quality_data.isnull().sum())     

    water_quality_data.fillna(water_quality_data.mean(), inplace=True)     

    print(water_quality_data.isnull().sum())
    print(water_quality_data.describe())     

    # Exploring the data
    fig, axes = plt.subplots(figsize=(10, 7))
    sns.heatmap(water_quality_data.corr(), annot=True,
                cmap='gist_rainbow', ax=axes)
    axes.set_title("Correlation Heatmap")
    plt.show()     

    fig, axes = plt.subplots(figsize=(20, 7))
    water_quality_data.boxplot(ax=axes)
    axes.set_title("Boxplot")
    plt.show()     

    print(water_quality_data['Solids'].describe())     

    print(water_quality_data.head())     

    print(water_quality_data.shape)     

    print(water_quality_data['Potability'].value_counts())     

    fig, axes = plt.subplots()    

    # Plot the countplot of Potability
    sns.countplot(water_quality_data['Potability'], ax=axes)
    axes.set_title("Countplot of Potability")
    plt.show()    

    fig, axes = plt.subplots(figsize=(20, 15))    

    # Plot the histogram of variables
    water_quality_data.hist(ax=axes)    

    # Set title and axis labels
    axes.set_title("Histogram of Variables", color="black")
    axes.set_xlabel("Variable")
    axes.set_ylabel("Frequency")    

    plt.tight_layout()  # Ensures proper spacing between subplots
    plt.show()



    fig = sns.pairplot(water_quality_data, hue="Potability")
    fig.fig.suptitle("Pairplot of Variables")
    plt.show()     

    fig, axes = plt.subplots()
    sns.scatterplot(x=water_quality_data['ph'],
                    y=water_quality_data['Potability'], ax=axes)
    axes.set_title("Scatterplot of pH vs. Potability")
    plt.show()     

    # ...     

    x = water_quality_data.drop('Potability', axis=1)  # input data
    y = water_quality_data['Potability']  # target variable     

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=404)     

    print(x_train)
    print(y_train)     

    AB = DecisionTreeClassifier(
        criterion='entropy', min_samples_split=9, splitter='best')
    AB.fit(x_train, y_train)  # Fit the model with training data     

    y_prediction = AB.predict(x_test)
    print(accuracy_score(y_prediction, y_test) * 100)
    print(confusion_matrix(y_test, y_prediction))
    print(y_test.shape)     

    AB = DecisionTreeClassifier()     

    criterion = ["gini", "entropy"]
    splitter = ["best", "random"]
    min_samples_split = range(2, 10)     

    parameters = dict(criterion=criterion, splitter=splitter,
                      min_samples_split=min_samples_split)
    cv = RepeatedStratifiedKFold(n_splits=5, random_state=250)     

    grid_search_cv_AB = GridSearchCV(
        estimator=AB, param_grid=parameters, scoring='accuracy', cv=cv)     

    print(grid_search_cv_AB.fit(x_train, y_train))
    print(grid_search_cv_AB.best_params_)     

    prediction_grid = grid_search_cv_AB.predict(x_test)
    print(accuracy_score(prediction_grid, y_test) * 100)
    print(confusion_matrix(y_test, prediction_grid))     

    knn = KNeighborsClassifier(metric='manhattan', n_neighbors=22)
    knn.fit(x_train, y_train)     

    prediction_knn = knn.predict(x_test)
    accuracy_knn = accuracy_score(y_test, prediction_knn) * 100
    print('accuracy_score score:', accuracy_knn, '%')
    print(confusion_matrix(prediction_grid, y_test))



   

# Air Quality Prediction
elif choice == 2:
    # Load the dataset
    # Load the dataset
    air_quality_data = load_dataset("air quality", date_column="Date")

    # Now you can use the air_quality_data DataFrame
    print(air_quality_data.head())

    # ... (code for air quality prediction using 'air_quality_data')
    
    print(air_quality_data.shape)
    fig, axes = plt.subplots(figsize=(10, 7))

# Plot the heatmap with missing values
    sns.heatmap(air_quality_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')    

    axes.set_title("Missing Values Heatmap")
    plt.show()

    plt.show()
    print(air_quality_data .isnull().sum())
    print((air_quality_data.isnull().sum() /
         air_quality_data.shape[0] * 100).sort_values(ascending=False))

    # # 
    print(air_quality_data.describe())
    air_quality_data['Date'] = air_quality_data['Date'].apply(pd.to_datetime)
    air_quality_data .set_index('Date', inplace=True)
    print(air_quality_data.columns)
    valid_columns = air_quality_data.columns[1:13]
    air_quality_data.loc[:, valid_columns] = air_quality_data.groupby(
    "City")[valid_columns].transform(lambda x: x.fillna(x.mean()))


    print(air_quality_data)
    fig, axes = plt.subplots(figsize=(10, 7))
    
    # Plot the heatmap with missing values
    sns.heatmap(air_quality_data.isnull(), yticklabels=False,
                cbar=False, cmap='viridis')
    
    axes.set_title("Missing Values Heatmap")
    plt.show()
    
   
  
    air_quality_data[valid_columns] = air_quality_data[valid_columns].fillna(
        air_quality_data[valid_columns].mean())
    print(air_quality_data)

     # PM10 Sub-Index calculation

 # PM10 Sub-Index calculation


    def get_PM10_subindex(x):
        if x <= 50:
            return x
        elif x > 50 and x <= 100:
            return x
        elif x > 100 and x <= 250:
            return 100 + (x - 100) * 100 / 150
        elif x > 250 and x <= 350:
            return 200 + (x - 250)
        elif x > 350 and x <= 430:
            return 300 + (x - 350) * 100 / 80
        elif x > 430:
            return 400 + (x - 430) * 100 / 80
        else:
            return 0     
    

    air_quality_data["PM10_SubIndex"] = air_quality_data["PM10"].astype(int).apply(get_PM10_subindex)     

    # PM2.5 Sub-Index calculation     
    

    def get_PM25_subindex(x):
        if x <= 30:
            return x * 50 / 30
        elif x > 30 and x <= 60:
            return 50 + (x - 30) * 50 / 30
        elif x > 60 and x <= 90:
            return 100 + (x - 60) * 100 / 30
        elif x > 90 and x <= 120:
            return 200 + (x - 90) * 100 / 30
        elif x > 120 and x <= 250:
            return 300 + (x - 120) * 100 / 130
        elif x > 250:
            return 400 + (x - 250) * 100 / 130
        else:
            return 0     
    

    air_quality_data["PM2.5_SubIndex"] = pd.to_numeric(
        air_quality_data["PM2.5"], errors="coerce").apply(get_PM25_subindex)     

    # SO2 Sub-Index calculation     
    

    def get_SO2_subindex(x):
        if x <= 40:
            return x * 50 / 40
        elif x > 40 and x <= 80:
            return 50 + (x - 40) * 50 / 40
        elif x > 80 and x <= 380:
            return 100 + (x - 80) * 100 / 300
        elif x > 380 and x <= 800:
            return 200 + (x - 380) * 100 / 420
        elif x > 800 and x <= 1600:
            return 300 + (x - 800) * 100 / 800
        elif x > 1600:
            return 400 + (x - 1600) * 100 / 800
        else:
            return 0     
    
    air_quality_data["SO2_SubIndex"] = air_quality_data["SO2"].astype(int).apply(get_SO2_subindex)     

    # NOx Sub-Index calculation     
    

    def get_NOx_subindex(x):
        if x <= 40:
            return x * 50 / 40
        elif x > 40 and x <= 80:
            return 50 + (x - 40) * 50 / 40
        elif x > 80 and x <= 180:
            return 100 + (x - 80) * 100 / 100
        elif x > 180 and x <= 280:
            return 200 + (x - 180) * 100 / 100
        elif x > 280 and x <= 400:
            return 300 + (x - 280) * 100 / 120
        elif x > 400:
            return 400 + (x - 400) * 100 / 120
        else:
            return 0     
    

    air_quality_data["NOx_SubIndex"] = air_quality_data["NOx"].astype(int).apply(get_NOx_subindex)     

    # NH3 Sub-Index calculation     
    

    def get_NH3_subindex(x):
        if x <= 200:
            return x * 50 / 200
        elif x > 200 and x <= 400:
            return 50 + (x - 200) * 50 / 200
        elif x > 400 and x <= 800:
            return 100 + (x - 400) * 100 / 400
        elif x > 800 and x <= 1200:
            return 200 + (x - 800) * 100 / 400
        elif x > 1200 and x <= 1800:
            return 300 + (x - 1200) * 100 / 600
        elif x > 1800:
            return 400 + (x - 1800) * 100 / 600
        else:
            return 0     
    

    air_quality_data["NH3_SubIndex"] = air_quality_data["NH3"].astype(int).apply(get_NH3_subindex)     

    # CO Sub-Index calculation     
    

    def get_CO_subindex(x):
        if x <= 1:
            return x * 50 / 1
        elif x > 1 and x <= 2:
            return 50 + (x - 1) * 50 / 1
        elif x > 2 and x <= 10:
            return 100 + (x - 2) * 100 / 8
        elif x > 10 and x <= 17:
            return 200 + (x - 10) * 100 / 7
        elif x > 17 and x <= 34:
            return 300 + (x - 17) * 100 / 17
        elif x > 34:
            return 400 + (x - 34) * 100 / 17
        else:
            return 0     
    

    air_quality_data["CO_SubIndex"] = air_quality_data["CO"].astype(int).apply(get_CO_subindex)     

    # O3 Sub-Index calculation     
    

    def get_O3_subindex(x):
        if x <= 50:
            return x * 50 / 50
        elif x > 50 and x <= 100:
            return 50 + (x - 50) * 50 / 50
        elif x > 100 and x <= 168:
            return 100 + (x - 100) * 100 / 68
        elif x > 168 and x <= 208:
            return 200 + (x - 168) * 100 / 40
        elif x > 208 and x <= 748:
            return 300 + (x - 208) * 100 / 540
        elif x > 748:
            return 400 + (x - 748) * 100 / 540
        else:
            return 0     
    

    air_quality_data["O3_SubIndex"] = air_quality_data["O3"].astype(int).apply(get_O3_subindex)

    # Filling the Nan values of AQI column by taking maximum values out of sub-Indexes
    air_quality_data["AQI"] = air_quality_data[["PM2.5_SubIndex", "PM10_SubIndex", "SO2_SubIndex",
                                                "NOx_SubIndex", "NH3_SubIndex", "CO_SubIndex", "O3_SubIndex"]].max(axis=1)
    print(air_quality_data)

    fig, axes = plt.subplots(figsize=(10, 7))    

    # Plot the heatmap with missing values
    sns.heatmap(air_quality_data.isnull(), yticklabels=False,
                cbar=False, cmap='viridis')    

    axes.set_title("Missing Values Heatmap")
    plt.show()

     
    
    #AQI Bucket
    
    display.Image("notebooks\__results___16_0.png", width=400, height=200)

    

    # Load the image
    image_path = "notebooks\__results___16_0.png"
    image = Image.open(image_path)    

    # Display the image
    image.show()
    # calculating AQI bucket and filling the NAN value present
    # AQI bucketing
    def get_AQI_bucket(x):
        if x <= 50:
            return "Good"
        elif x > 50 and x <= 100:
            return "Satisfactory"
        elif x > 100 and x <= 200:
            return "Moderate"
        elif x > 200 and x <= 300:
            return "Poor"
        elif x > 300 and x <= 400:
            return "Very Poor"
        elif x > 400:
            return "Severe"
        else:
            return '0'

    air_quality_data["AQI_Bucket"] = air_quality_data["AQI_Bucket"].fillna(
        air_quality_data["AQI"].apply(lambda x: get_AQI_bucket(x)))
    print(air_quality_data)
    fig, axes = plt.subplots(figsize=(10, 7))    

    # Plot the heatmap with missing values
    sns.heatmap(air_quality_data.isnull(), yticklabels=False,
                cbar=False, cmap='viridis')    

    axes.set_title("Missing Values Heatmap")
    plt.show()

    
    air_quality_data_city_day = air_quality_data .copy()
    print(air_quality_data.columns)

 

    
    # Filter out non-numeric columns
    numeric_columns = air_quality_data.select_dtypes(include=np.number)
    
    # Plot the correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_columns.corr(), cmap='coolwarm', annot=True)
    plt.show()
    
    # Select the pollutants
    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                  'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    air_quality_data_city_day = air_quality_data_city_day[pollutants]
    
    # Plot the distribution of pollutants
    print('Distribution of different pollutants in the last 5 years')
    air_quality_data_city_day.plot(kind='line', figsize=(
        18, 18), cmap='coolwarm', subplots=True, fontsize=10)
    plt.show()
    

  




    air_quality_data[['City', 'AQI']].groupby('City').mean().sort_values(
        'AQI').plot(kind='bar', cmap='Blues_r',    figsize=(8, 8))
    plt.title('Average AQI in last 5 years')
    plt.show()
    final_air_quality_data  = air_quality_data[['AQI', 'AQI_Bucket']].copy()
    print(final_air_quality_data) 

    print(final_air_quality_data ['AQI_Bucket'].unique())

    final_air_quality_data ['AQI_Bucket'] = final_air_quality_data ['AQI_Bucket'].map(
        {'Good': 0, 'Satisfactory': 1, 'Moderate': 2, 'Poor': 3, 'Very Poor': 4, 'Severe': 5}).astype(int)  # mapping numbers
    print(final_air_quality_data.head())
    # Predicting the values of AQI_Bucket w.r.t values of AQI using Random Forest Classifier

    X = final_air_quality_data [['AQI']]
    y = final_air_quality_data [['AQI_Bucket']]

 

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Enter the value of AQI:")
    AQI = float(input("AQI : "))
    output = clf.predict([[AQI]])
    print(output)    

    # 0-->Good
    # 1-->Satisfactory
    # 2-->moderate
    # 3-->poor
    # 4-->Very poor
    # 5-->Severe

 
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# Deforestation Prediction
elif choice == 3:
    # Load the dataset
    df, mun, states = load_dataset("deforestation")
    print(df.head())
    # ... (code for deforestation prediction using 'deforestation_data')

    def call(number):
        """This fuction will use the first two characters of a number
        and return all the matches from the states["estados_id"] dataframe"""
        num = str(number)[0:2]
        num = int(num)
        return states[states["estados_id"] == num]

    def transform(air_quality_data):
        """It creates two lists with the name of the county and its state"""
        munic = []
        esta = []
        for i in range(len(air_quality_data["id_municipio"])):
            ind = mun[mun["Código Município Completo"] ==
                      air_quality_data["id_municipio"][i]]["Nome_Microrregião"].index[0]
            nome_mun = mun[mun["Código Município Completo"] ==
                           air_quality_data["id_municipio"][i]]["Nome_Microrregião"][ind]
    
            ind_es = mun[mun["Código Município Completo"] ==
                         air_quality_data["id_municipio"][i]]["Código Município Completo"][ind]
            m = call(ind_es)["Estados"].index[0]
            nome_est = call(ind_es)["Estados"][m]
    
            munic.append(nome_mun)
            esta.append(nome_est)
        return munic, esta

    def stats_year(air_quality_data, nome, Mean):
          """it returns the total sum of the nome column grouped
          by the ano column"""
          sum = air_quality_data[["ano", nome]].groupby(['ano']).sum()
          media = sum[nome].mean()
          vals = []
          if Mean == False:
              for k in sum[nome]:
                  vals.append(k)
          else:
              for k in sum[nome]:
                  vals.append(k/media)
          return np.array(vals), sum.index

    lista_mun, lista_est = transform(df)  # lets add the new columns to our dataset
    df["municipios"] = lista_mun
    df["estados"] = lista_est

    selected_columns = ["ano", "area", "desmatado", "floresta", "nuvem",
                        "nao_observado", "nao_floresta", "hidrografia", "estados", "municipios"]
    new = df[selected_columns]    

    # Exclude non-numeric columns from the 'new' DataFrame
    numeric_columns = new.select_dtypes(include=np.number).columns
    new = new[numeric_columns]    

    corr = round(new.corr(), 2)
    mask = np.triu(np.ones_like(corr, dtype=bool))    

    f, ax = plt.subplots(figsize=(9, 5))    

    # Plot the heatmap
    sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True)    

    ax.set_title("Correlation Heatmap")
    plt.show()

    lista1, anos1 = stats_year(df, "desmatado", True)
    lista2, anos2 = stats_year(df, "floresta", True)
    plt.figure(figsize=(9, 5))
    plt.plot(list(anos1), lista1, "r", label="Deforestation Rate of Change")
    plt.plot(list(anos2), lista2, "g", label="Forest Area Rate of Change")
    plt.ylabel("Percentage %")
    plt.legend()

    anos = list(set(df["ano"]))
    year = []
    desmat = []
    munic = []
    esta = []
    for ano in anos:

        new = df[df["ano"] == ano]
        new = new.sort_values(by=['desmatado'], ascending=False)
      
        year.append(np.array(list(new.copy().iloc[0:10]["ano"])[:]))
        desmat.append(np.array(list(new.copy().iloc[0:10]["desmatado"])[:]))
        munic.append(np.array(list(new.copy().iloc[0:10]["municipios"])[:]))
        esta.append(np.array(list(new.copy().iloc[0:10]["estados"])[:]))

    dic = dict()
    dic["ano"] = np.array(year).reshape(1, -1)[0]
    dic["estado"] = np.array(esta).reshape(1, -1)[0]
    dic["desmatado"] = np.array(desmat).reshape(1, -1)[0]
    dic["municipio"] = np.array(munic).reshape(1, -1)[0]
    novo = pd.DataFrame(dic)
    print(novo)

    muns = list(Counter(novo["municipio"]))
    mat = []
    for i in range(len(muns)):    

      data = df[df["municipios"] == muns[i]]
      mat.append(stats_year(data, "desmatado", False)[0])

    
    plt.figure(figsize=(15, 8))
    for i in range(len(muns)):    

      if muns[i] == "São Félix do Xingu":
        plt.plot(list(set(novo.ano)), mat[i])
        plt.text(list(set(novo.ano))[-1]-4, mat[i]
                 [-1]-1000, str(muns[i]), fontsize=10)
      elif muns[i] == "Arinos":
        plt.plot(list(set(novo.ano)), mat[i])
        plt.text(list(set(novo.ano))[-1]-4, mat[i][-1], str(muns[i]), fontsize=10)
      else:
        plt.plot(list(set(novo.ano)), mat[i])
        plt.text(list(set(novo.ano))[-1], mat[i][-1], str(muns[i]), fontsize=10)
    plt.ylabel("Deforestation (Km^2)", fontsize=20)
    plt.show()

    estd = list(Counter(df["estados"]))
    mat = []
    for i in range(len(estd)):    

       data = df[df["estados"] == estd[i]]
       mat.append(stats_year(data, "desmatado", False)[0])    

    ano = []
    est = []
    for estado in estd:
      for i in range(2000, 2022):
        ano.append(i)
        est.append(estado)

    d = {}
    d["ano"] = ano
    d["estado"] = est
    d["desmatado"] = np.array(mat).reshape(1, -1)[0]    

    X = pd.DataFrame(d)
    labels = list(Counter(X["estado"]))    

    X["estado"] = LabelEncoder().fit_transform(X["estado"])
    Y = X.pop("desmatado")
    labels_encod = list(Counter(X["estado"]))

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)


    KN = KNeighborsRegressor()
    bag = BaggingRegressor()    

    mod = GridSearchCV(estimator=KN,param_grid= {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]},cv=2)    

    mod2 = GridSearchCV(estimator=bag,param_grid= {'n_estimators':[100,120,130,150,180]},cv=2)    

    mod3 = GridSearchCV(estimator=SGDRegressor(max_iter=1200,early_stopping=True),param_grid={'penalty':    ["l1","l2"]} ,cv=2)    

    vot = VotingRegressor(estimators=[("kn",mod),("bag",mod2),("est",mod3)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=True)
    vot.fit(X_train, y_train)
    print(metrics.r2_score(y_test, vot.predict(X_test)))
    
    m = []
    for i in labels_encod:
      for year in range(2022, 2024):
          m.append({'ano': year, 'estado': i})
    m_air_quality_data = pd.DataFrame(m)    

    pred = scaler.transform(m_air_quality_data[['ano', 'estado']])

    predic = vot.predict(pred)
    air_quality_data = pd.DataFrame(d)
    ano = [i for i in range(2000, 2024)]
    

    plt.figure(figsize=(15, 8))
    c = 0
    for i in labels:
      dat = air_quality_data[air_quality_data["estado"] == i]
      es = list(dat["desmatado"])
      es.append(predic[c])
      es.append(predic[c+1])
      plt.plot(ano, es, label=i)
      c += 2
    plt.axvline(2021, color='k', linestyle='--')
    plt.legend()
    plt.xticks(ano, rotation=45)
    plt.show()
# Climate Pattern Prediction
elif choice == 4:
      # Load the dataset
    climate_pattern_data = load_dataset("climate pattern")
    print(climate_pattern_data.head())
    print(climate_pattern_data.tail())
    print(climate_pattern_data.info())
    print(climate_pattern_data.isnull().sum())
    climate_pattern_data['date'] = pd.to_datetime(climate_pattern_data['date'])
    print(climate_pattern_data.nunique())    

    # Plot 1: Countplot of weather
    plt.figure(figsize=(10, 5))
    sns.set_theme()
    sns.countplot(x='weather', data=climate_pattern_data, palette="ch:start=.2,rot=-.3")
    plt.xlabel("weather", fontweight='bold', size=13)
    plt.ylabel("Count", fontweight='bold', size=13)
    plt.title("Count of Weather", fontweight='bold', size=15)
    plt.show()    

    # Plot 2: Lineplot of temp_max
    plt.figure(figsize=(18, 8))
    sns.set_theme()
    sns.lineplot(x='date', y='temp_max', data=climate_pattern_data)
    plt.xlabel("Date", fontweight='bold', size=13)
    plt.ylabel("Temp_Max", fontweight='bold', size=13)
    plt.title("Lineplot of temp_max", fontweight='bold', size=15)
    plt.show()    

    # Plot 3: Lineplot of temp_min
    plt.figure(figsize=(18, 8))
    sns.set_theme()
    sns.lineplot(x='date', y='temp_min', data=climate_pattern_data)
    plt.xlabel("Date", fontweight='bold', size=13)
    plt.ylabel("Temp_Min", fontweight='bold', size=13)
    plt.title("Lineplot of temp_min", fontweight='bold', size=15)
    plt.show()    

    # Plot 4: Lineplot of wind
    plt.figure(figsize=(18, 8))
    sns.set_theme()
    sns.lineplot(x='date', y='wind', data=climate_pattern_data)
    plt.xlabel("Date", fontweight='bold', size=13)
    plt.ylabel("wind", fontweight='bold', size=13)
    plt.title("Lineplot of wind", fontweight='bold', size=15)
    plt.show()    

    # Plot 5: Pairplot of weather vs. other numerical variables
    sns.pairplot(climate_pattern_data.drop('date', axis=1), hue='weather', palette="hot")
    plt.title("Pairplot of weather vs. other numerical variables",
              fontweight='bold', size=10)

    plt.show()    
  

    # Plot 6: Catplot of weather vs. temp_max
    sns.catplot(x='weather', y='temp_max', data=climate_pattern_data, palette="hot")
    plt.title("Catplot of weather vs. temp_max",
              fontweight='bold', size=10)

    plt.show()   


    # Plot 8: Catplot of weather vs. temp_min
    sns.catplot(x='weather', y='temp_min', data=climate_pattern_data, palette="hot")
    plt.title("Catplot of weather vs. temp_min",
              fontweight='bold', size=10)
    plt.show()    


    # Plot 9: Catplot of weather vs. wind
    sns.catplot(x='weather', y='wind', data=climate_pattern_data, palette="hot")
    plt.title("Catplot of weather vs. wind",
              fontweight='bold', size=10)
    plt.show()    


    # Plot 10: Catplot of weather vs. precipitation
    sns.catplot(x='weather', y='precipitation', data=climate_pattern_data, palette="hot")
    plt.title("Catplot of weather vs. precipitation",
              fontweight='bold', size=10)
    plt.show()    

    # Plot 11: Scatterplots of weather vs. numerical variables
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle('Price Range vs all numerical factor')    

    sns.scatterplot(ax=axes[0, 0], data=climate_pattern_data, x='weather', y='precipitation')
    sns.scatterplot(ax=axes[0, 1], data=climate_pattern_data, x='weather', y='temp_max')
    sns.scatterplot(ax=axes[1, 0], data=climate_pattern_data, x='weather', y='temp_min')
    sns.scatterplot(ax=axes[1, 1], data=climate_pattern_data, x='weather', y='wind')
    plt.show()    

    # Data preprocessing: Label encoding for the "weather" column
    def LABEL_ENCODING(c1):
        label_encoder = preprocessing.LabelEncoder()
        climate_pattern_data[c1] = label_encoder.fit_transform(climate_pattern_data[c1])
        climate_pattern_data[c1].unique()    

    LABEL_ENCODING("weather")
    print(climate_pattern_data)    

    data = climate_pattern_data.drop('date', axis=1)
    x = data.drop('weather', axis=1)
    y = data['weather']    

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)    

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)    

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)    

    classifier = LogisticRegression(random_state=0)
    print(classifier.fit(X_train, y_train))
    y_pred = classifier.predict(X_test)
    print(y_pred)    

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    sns.heatmap(cm, annot=True)
    plt.show()    

    acc1 = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {acc1}")    

    classifier = SVC(kernel='linear', random_state=0)
    print(classifier.fit(X_train, y_train))
    y_pred = classifier.predict(X_test)    

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    acc2 = accuracy_score(y_test, y_pred)    

    print(f"Accuracy score: {acc2}")    

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    print(classifier.fit(X_train, y_train))
    y_pred = classifier.predict(X_test)    

    cm = confusion_matrix(y_test, y_pred)
    print(cm)    

    acc3 = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {acc3}")    

    classifier = GaussianNB()
    print(classifier.fit(X_train, y_train))
    y_pred = classifier.predict(X_test)    

    cm = confusion_matrix(y_test, y_pred)
    print(cm)    

    acc4 = accuracy_score(y_test, y_pred)
    print(f"Accuracy score : {acc4}")    

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    print(classifier.fit(X_train, y_train))
    y_pred = classifier.predict(X_test)    

    print(y_pred)    

    cm = confusion_matrix(y_test, y_pred)
    print(cm)    

    sns.heatmap(cm, annot=True)
    plt.show()    

    acc5 = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {acc5}")    

    forest = RandomForestClassifier(n_estimators=40, random_state=0)
    forest.fit(X_train, y_train)
    RandomForestClassifier(n_estimators=40, random_state=0)
    y_pred = forest.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)    

    sns.heatmap(cm, annot=True)    

    print(classification_report(y_test, y_pred))
    acc6 = forest.score(X_test, y_test)
    print(acc6)    

    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)    

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    acc7 = accuracy_score(y_test, y_pred)    

    print(acc7)    

    mylist = []
    mylist2 = []
    mylist.append(acc1)
    mylist2.append("Logistic Regression")
    mylist.append(acc2)
    mylist2.append("SVM")
    mylist.append(acc3)
    mylist2.append("KNN")
    mylist.append(acc4)
    mylist2.append("Naive Bayes")
    mylist.append(acc5)
    mylist2.append("DTR")
    mylist.append(acc6)
    mylist2.append("RF")
    mylist.append(acc7)
    mylist2.append("XGBoost")    

    plt.rcParams['figure.figsize'] = 8, 6
    sns.set_style("darkgrid")
    plt.figure(figsize=(22, 8))
    ax = sns.barplot(x=mylist2, y=mylist, palette="mako", saturation=1.5)
    plt.xlabel("Classification Models", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.title("Accuracy of different Classification Models", fontsize=20)
    plt.xticks(fontsize=11, horizontalalignment='center', rotation=8)
    plt.yticks(fontsize=13)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.2%}', (x + width/2, y + height * 1.02), ha='center',     fontsize='x-large')
    plt.tight_layout()
    plt.show()



else:
   print("Invalid choice. Please select a valid option (1, 2, 3, or 4).")
