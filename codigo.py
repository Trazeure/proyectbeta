from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

# Cargar los datos de béisbol
baseball_data = pd.read_csv("batting.csv",encoding='ISO-8859-1')
baseball_data = baseball_data[baseball_data['Season'] == 2022]

# Cargar los datos de la NFL
nfl_data = pd.read_csv("nfldata.csv")
 
#Cargar los datos de Futbol soccer
futbol_data = pd.read_csv("estadisticasftmx.csv", encoding='ISO-8859-1')

# Reemplazar los valores no numéricos en la columna Lng de la NFL
nfl_data['Lng'] = nfl_data['Lng'].apply(lambda x: 99 if x[-1] == 'T' else x)

# Convertir la columna Lng de la NFL a valores numéricos
nfl_data['Lng'] = pd.to_numeric(nfl_data['Lng'])

# Definir función para hacer la predicción de béisbol
def predict_baseball_statistic(statistic):
    # Seleccionar las variables de interés
    if statistic == 'AVG':
        X = baseball_data[['H', 'AB']]
        y = baseball_data['AVG']
    elif statistic == 'HR':
        X = baseball_data[['AB', 'H']]
        y = baseball_data['HR']
    elif statistic == 'RBI':
        X = baseball_data[['AB', 'H', '2B', '3B']]
        y = baseball_data['RBI']
    else:
        print("Estadística no válida")
        return

    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

    # Entrenar el modelo de regresión lineal con los datos de entrenamiento
    reg = LinearRegression().fit(X_train, y_train)

    # Realizar predicciones con el modelo entrenado
    y_pred = reg.predict(X_test)

    # Evaluar el desempeño del modelo
    score = r2_score(y_test, y_pred)
    print("R2 score:", score)

    # Crear un dataframe con las variables que se utilizarán para hacer la predicción
    if statistic == 'AVG':
        new_data = baseball_data[['H', 'AB']]
    elif statistic == 'HR':
        new_data = baseball_data[['AB', 'H']]
    elif statistic == 'RBI':
        new_data = baseball_data[['AB', 'H', '2B', '3B']]
    else:
        new_data = None
        print("Estadística no válida")
        return

    # Hacer predicciones con el modelo entrenado
    if new_data is not None:
        pred = reg.predict(new_data)

        # Añadir las predicciones al dataframe
        baseball_data[statistic + '_pred'] = pred

        # Guardar los resultados en un archivo CSV
        baseball_data.to_csv(statistic + '_pred.csv')

        # Devolver el dataframe con las predicciones
        return baseball_data[['Name', statistic + '_pred']]

# Definir función para hacer la predicción de la NFL
def predict_nfl_statistic(statistic):
    # Seleccionar las variables de interés
    if statistic == 'Passing Yds':
        X = nfl_data[['Att', 'Cmp', 'Cmp %', 'Yds/Att', 'TD', 'INT', 'Rate', '1st', '1st%', '20+', '40+', 'Lng', 'Sck', 'SckY']]
        y = nfl_data['Pass Yds']
    elif statistic == 'TD':
        X = nfl_data[['Att', 'Cmp', 'Cmp %', 'Yds/Att', 'Pass Yds', 'INT', 'Rate', '1st', '1st%', '20+', '40+', 'Lng', 'Sck', 'SckY']]
        y = nfl_data['TD']
    else:
        print("Estadística no válida")
        return

    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

    # Entrenar el modelo de regresión lineal con los datos de entrenamiento
    reg = LinearRegression().fit(X_train, y_train)

    # Realizar predicciones con el modelo entrenado
    y_pred = reg.predict(X_test)

    # Evaluar el desempeño del modelo
    score = r2_score(y_test, y_pred)
    print("R2 score:", score)

    # Crear un dataframe con las variables que se utilizarán para hacer la predicción
    if statistic == 'Passing Yds':
        new_data = nfl_data[['Att', 'Cmp', 'Cmp %', 'Yds/Att', 'TD', 'INT', 'Rate', '1st', '1st%', '20+', '40+', 'Lng', 'Sck', 'SckY']]
    elif statistic == 'TD':
        new_data = nfl_data[['Att', 'Cmp', 'Cmp %', 'Yds/Att', 'Pass Yds', 'INT', 'Rate', '1st', '1st%', '20+', '40+', 'Lng', 'Sck', 'SckY']]
    else:
        new_data = None
        print("Estadística no válida")
        return

    # Hacer predicciones con el modelo entrenado
    if new_data is not None:
        pred = reg.predict(new_data)

        # Añadir las predicciones al dataframe
        nfl_data[ statistic + '_pred'] = pred

        # Guardar los resultados en un archivo CSV
        nfl_data.to_csv( statistic + '_pred.csv')

        # Devolver el dataframe con las predicciones
        return nfl_data[['Team', statistic + '_pred']]

# Definir función para hacer la predicción de Futbol soccer

def predict_ftmx_statistic(statistic):
    # Seleccionar las variables de interés
    if statistic == 'W Win':
        X = futbol_data[['D Draw', 'GF Goals For (GF). The number of goals this team have scored.', 'GD Goal Difference (GD). Goals Scored - Goals Conceded']]
        y = futbol_data[['W Win', 'L Loss', 'D Draw']]
    elif statistic == 'L Loss':
        X = futbol_data[['D Draw', 'GF Goals For (GF). The number of goals this team have scored.', 'GD Goal Difference (GD). Goals Scored - Goals Conceded']]
        y = futbol_data[['L Loss']]
    elif statistic == 'D Draw':
        X = futbol_data[['W Win', 'L Loss', 'GF Goals For (GF). The number of goals this team have scored.', 'GD Goal Difference (GD). Goals Scored - Goals Conceded']]
        y = futbol_data[['D Draw']]
    else:
        print("Estadística no válida")
        return

    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

    # Entrenar el modelo de regresión lineal con los datos de entrenamiento
    reg = LinearRegression().fit(X_train, y_train)

    # Hacer predicciones con el modelo entrenado
    pred = reg.predict(X)

    # Añadir las predicciones al dataframe
    if statistic == 'W Win':
        futbol_data['W Win_pred'] = pred[:, 0]
        futbol_data['L Loss_pred'] = pred[:, 1]
        futbol_data['D Draw_pred'] = pred[:, 2]
    elif statistic == 'L Loss':
        futbol_data['L Loss_pred'] = pred[:, 0]
    elif statistic == 'D Draw':
        futbol_data['D Draw_pred'] = pred[:, 0]

    # Guardar los resultados en un archivo CSV
    futbol_data.to_csv('predictions.csv')

    # Devolver el dataframe con las predicciones
    if statistic == 'W Win':
        return futbol_data[['Team', 'W Win_pred']]
    elif statistic == 'L Loss':
        return futbol_data[['Team', 'L Loss_pred']]
    elif statistic == 'D Draw':
        return futbol_data[['Team', 'D Draw_pred']]
    else:
        return None




# Definir la ruta para el formulario
@app.route("/", methods=["GET", "POST"])
def predict_form():
    if request.method == "POST":
        sport = request.form.get("sport")
        statistic = request.form.get("statistic")
        if sport == 'baseball':
            result = predict_baseball_statistic(statistic)
        elif sport == 'nfl':
            result = predict_nfl_statistic(statistic)
        elif sport == 'ftmx':
            result = predict_ftmx_statistic(statistic)
        else:
            return render_template("form.html", error="Debes seleccionar un deporte válido")

        if result is not None:
            return render_template("result.html", result=result.to_dict('records'), statistic=statistic)
        else:
            return render_template("form.html", error="Estadística no válida") 
    else:
        return render_template("form.html")





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
