import pandas as pd
import missingno as msno
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

if __name__=="__main__":
    pd.set_option('display.max_columns', None)

    data=pd.read_csv('data/Liver Patient Dataset (LPD)_train.csv',encoding='cp1252')
    #Some of the column names have redundant spaces
    for col in data.columns:
        data.rename(columns={col: col.strip()}, inplace=True)

    #renaming columns into more terse alternatives
    data.rename(columns={'Age of the patient': 'Age',
                         'Gender of the patient':'Gender',
                         'Alkphos Alkaline Phosphotase':'ALP',
                         'Sgpt Alamine Aminotransferase':'ALT',
                         'Sgot Aspartate Aminotransferase':'AST',
                         'Total Protiens': 'Total Proteins',
                         'ALB Albumin':'Albumin',
                         'A/G Ratio Albumin and Globulin Ratio':'A/G'},inplace=True)
    print(data)

    #removing NaN values
    msno.matrix(data)
    plt.show()

    print(f"\nTotal rows: {len(data)}\nAbsent values per column:")
    for col in data.columns:
        print(f"\t{col} - {data[col].isna().sum()}")


    print(f"\nProceeding to delete rows that contain NaN values\nRows before deletion: {len(data)}")
    data.dropna(inplace=True)
    print(f"Rows after deletion:{len(data)}")

    counts = data['Result'].value_counts()
    labels = counts.index
    values = counts.values
    plt.bar(labels, values)
    plt.xlabel('Result')
    plt.ylabel('Count')
    plt.show()

    print(data.describe())

    #check for outliers
    data[['Total Bilirubin','Direct Bilirubin']].boxplot()
    plt.ylabel('Values')
    plt.show()

    data[['ALP', 'ALT','AST']].boxplot()
    plt.ylabel('Values')
    plt.show()

    data[['Albumin']].boxplot()
    plt.ylabel('Values')
    plt.show()

    data[['Total Proteins']].boxplot()
    plt.ylabel('Values')
    plt.show()



    # Function to count outliers using Z-score
    def count_outliers_zscore(column):
        if pd.api.types.is_numeric_dtype(column):
            z_scores = (column - column.mean()) / column.std()
            outliers = np.abs(z_scores) > 3  # Those tha lie after 3 standart deviations
            return outliers.sum()
        else:
            return 0  # Skip non-numeric columns


    # Print outlier counts for each column
    print('Counting outliers:')
    for col in data.columns:
        outlier_count = count_outliers_zscore(data[col])
        print(f'\t{col}: {outlier_count}')


    #some encoding
    #gender
    genders={'Male':0,'Female':1}
    data['Gender']=data['Gender'].map(genders).astype(int)

    #those with 2 tend to have normal enzymes level, so they are not likely suffering from liver diseases
    #this mappping seems more logical
    results={2:0,1:1}
    data['Result']=data['Result'].map(results)

    #Determining predictors
    sns.heatmap(data.corr(),annot=True)
    plt.show()
    data.drop(columns=['Total Proteins','Age','Gender'],inplace=True)

    sns.heatmap(data.corr(),annot=True)
    plt.show()

    data['Globulin']=data['Albumin']/data['A/G']
    data.drop(columns=['Total Bilirubin','AST','A/G'],inplace=True)

    sns.heatmap(data.corr(),annot=True)
    plt.show()

    data.drop(columns=['Globulin'],inplace=True)
    print(data)

    sns.pairplot(data)
    plt.show()

    #splitting data
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data['Result'], test_size=0.2, random_state=42)

    print("Testing SVC:")


    param_grid = {
        'kernel': ['linear', 'poly','rbf'],
        'gamma': ['scale', 'auto']
    }
    svc_clf = SVC()
    tmp=data.sample(1000)
    X_svc_train=tmp.iloc[:, :-1]
    y_svc_train=tmp['Result']

    grid_search = GridSearchCV(svc_clf, param_grid, cv=5)
    grid_search.fit(X_svc_train, y_svc_train)

    best_params = grid_search.best_params_#linear, scale
    best_model = grid_search.best_estimator_
    print(f'Best parameters for SVC are: {grid_search.best_params_}')

    #best_model = SVC(kernel='linear',gamma='scale').fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    svc_acc = accuracy_score(y_test, y_pred)
    svc_recall = recall_score(y_test, y_pred)
    print(f'For SVC metrics are:\n\tAccuracy: {svc_acc}\n\tRecall:{svc_recall}\n\tF-score:{2 * svc_acc * svc_recall / (svc_acc + svc_recall)}')
    cm=confusion_matrix(y_test,y_pred)
    svc_type2_err=cm[1, 0]
    svc_type1_err=cm[0, 1]

    sns.heatmap(cm,annot=True)
    plt.grid(False)
    plt.show()

    #logistic regression
    param_grid = {
        'logisticregression__C': [0.1, 1, 10],
        'logisticregression__penalty': ['l1', 'l2'],
        'logisticregression__solver': ['liblinear', 'saga']
    }

    logreg = pipeline = make_pipeline(StandardScaler(), LogisticRegression())

    grid_search = GridSearchCV(logreg, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f'Best parameters for logistic regression are: {grid_search.best_params_}')
    best_model = grid_search.best_estimator_

    # Evaluate the model
    accuracy = best_model.score(X_test, y_test)
    logreg.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    logreg_acc = accuracy_score(y_test, y_pred)
    logreg_recall = recall_score(y_test, y_pred)
    print(f'For logistic regression metrics are:\n\tAccuracy: {logreg_acc}\n\tRecall:{logreg_recall}\n\tF-score:{2 * logreg_acc * logreg_recall / (logreg_acc + logreg_recall)}')
    cm = confusion_matrix(y_test, y_pred)
    logreg_type2_err=cm[1, 0]
    logreg_type1_err=cm[0, 1]

    sns.heatmap(cm,annot=True)
    plt.grid(False)
    plt.show()
    # Define the parameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [ 1,2,3,4,5,6,7,8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Create an instance of SMOTE
    oversampler = SMOTE()

    # Apply oversampling to the training data
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    tree_clf = DecisionTreeClassifier()
    grid_search = GridSearchCV(tree_clf, param_grid, cv=5)
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_params = grid_search.best_params_

    print(f'Best parameters for DTC are: {grid_search.best_params_}')
    best_model= grid_search.best_estimator_
    best_model.fit(X_train_resampled, y_train_resampled)
    y_pred = best_model.predict(X_test)
    DTC_acc = accuracy_score(y_test, y_pred)
    DTC_recall = recall_score(y_test, y_pred)
    print(f'For random forest metrics are:\n\tAccuracy: {DTC_acc}\n\tRecall:{DTC_recall}\n\tF-score:{2 * DTC_acc * DTC_recall / (DTC_acc + DTC_recall)}')
    cm = confusion_matrix(y_test, y_pred)
    dtc_type2_err = cm[1, 0]
    dtc_type1_err = cm[0, 1]
    sns.heatmap(cm,annot=True)
    plt.grid(False)
    plt.show()

    results=pd.DataFrame({
                    'Accuracy':[svc_acc,logreg_acc,DTC_acc],
                    'Recall':[svc_recall,logreg_recall,DTC_recall]
    }, index=['SVC','Logistic Regression','DTC'])
    results['F']=2*results['Accuracy']*results['Recall']/(results['Accuracy']+results['Recall'])
    results.plot(kind='bar')
    plt.xlabel('Algorithms')
    plt.ylabel('Metrics')
    plt.title('Performance Metrics by Algorithm')
    plt.legend(title='Metrics')
    plt.show()

    results = pd.DataFrame({
        'Type 1': [svc_type1_err, logreg_type1_err, dtc_type1_err]
    }, index=['SVC', 'LogReg', 'DTC'])
    results.plot(kind='bar')
    plt.xlabel('Algorithms')
    plt.ylabel('Metrics')
    plt.title('Performance Metrics by Algorithm')
    plt.legend(title='Metrics')
    plt.show()

    results = pd.DataFrame({
        'Type 2': [svc_type2_err, logreg_type2_err, dtc_type2_err]
    }, index=['SVC', 'LogReg', 'DTC'])
    results.plot(kind='bar')
    plt.xlabel('Algorithms')
    plt.ylabel('Metrics')
    plt.title('Performance Metrics by Algorithm')
    plt.legend(title='Metrics')
    plt.show()
