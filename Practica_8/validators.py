from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ValidationMethods:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.scaler = StandardScaler()
        
    def holdout(self, model):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0, shuffle=True)
        cols = X_train.columns 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = pd.DataFrame(X_train, columns=cols)
        X_test = pd.DataFrame(X_test, columns=cols)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Desempeño Hold-Out estratificado 70-30")
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (Hold-Out):", accuracy)

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix - Hold-Out 70/30")
        plt.show()

    def stratified_k_fold(self, model, n_splits=10):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        accuracies = []
        
        for fold, (train_index, test_index) in enumerate(skf.split(self.X, self.y), start=1):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            print(f"Fold {fold} Accuracy: {accuracy}")

        print(f"Average Accuracy (Stratified 10-Fold): {sum(accuracies) / len(accuracies)}")

    def leave_one_out(self, model):
        loo = LeaveOneOut()
        accuracies = []

        #  Revisión del bucle de LOOCV
        for train_index, test_index in loo.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            # Escalado de los datos
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Entrenamiento del modelo
            model.fit(X_train, y_train)

            # Predicción y cálculo de la precisión
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # Cálculo de la precisión promedio
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"Average Accuracy (Leave-One-Out): {avg_accuracy}")

