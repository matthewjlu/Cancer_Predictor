import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


#need to export the model and scaler so that the app doesn't train a ml model
#every time someone runs the app (we will write a binary file)
def main():
    data = get_clean_data()

    #no label
    features = data.drop(['diagnosis'], axis = 1)
    columns = features.columns
    max_column_values = features.max()
    mean_column_values = features.mean()
    slider_labels = [(col.upper(), col) for col in features.columns]
    model, scaler, X_train, X_test, y_train, y_test = create_model(data)
    #test_model(model, X_train, X_test, y_train, y_test)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('model/column_names.pkl', 'wb') as f:
        pickle.dump(columns, f)
    with open('model/max_values.pkl', 'wb') as f:
        pickle.dump(max_column_values, f)
    with open('model/slider_labels.pkl', 'wb') as f:
        pickle.dump(slider_labels, f)
    with open('model/mean_values.pkl', 'wb') as f:
        pickle.dump(mean_column_values, f)
    with open('model/features.pkl', 'wb') as f:
        pickle.dump(features, f)

#Removing the null values out of my dataset and returning the data
def get_clean_data():
    #no need to get out of the model folder
    data = pd.read_csv("data/data.csv")
    data.drop(["Unnamed: 32", "id"], axis = 1, inplace = True)
    data["diagnosis"] = data["diagnosis"].map({'M': 1, 'B': 0})
    return data

def create_model(data):
    X = data.drop(["diagnosis"], axis = 1)
    y = data["diagnosis"]

    #splitting the data and making the test variables 20% of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
        random_state = 7)
    
    #normalizing the features
    #we split before we scale to ensure that the training set doesn't have any
    #accidental information about the mean and sd of the test set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_test = scaler.transform(X_test.values)
    
    #using logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, scaler, X_train, X_test, y_train, y_test

#def test_model(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test.values)
    print(f"Accuracy of model: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report of model: \n"
        f"{classification_report(y_test, y_pred)}")
    
if __name__ == '__main__':
    main()