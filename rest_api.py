from flask import Flask, jsonify, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import keras
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = keras.models.load_model('model.h5')


def getData():
    global subset_floats
    df = pd.read_csv('Tesla.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(df['Date'], inplace=True)
    # Drop unneeded columns
    df_copy = df.drop(['Date', 'High', 'Low', 'Close',
                      'Adj Close', 'Volume'], axis=1, inplace=False)

    def create_shape(df_copy, timesteps):
        X, y = [], []
        for i in range(timesteps, len(df_copy)):
            X.append(df_copy[i-timesteps:i, 0])
            y.append(df_copy[i, 0])
        return np.array(X), np.array(y)

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_df = scaler.fit_transform(df_copy)
    print("The scaled df is: ", scaled_df)
    print("The tail is: ", df_copy.tail(10))
    train = scaled_df[:-13]  # train_set (7 + time_steps)
    test = scaled_df[-13:]  # test_set

    X_train, y_train = create_shape(train, 5)
    # Creating shape(batch_size, time_steps, 1) for lstm model:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    X_test, y_test = create_shape(test, 5)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    predictions = model.predict(X_test)
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = y_test.reshape(y_test.shape[0], 1)
    y_test_inv = scaler.inverse_transform(y_test_inv)
    np.set_printoptions(precision=2, suppress=True)

# Extract the subset of values starting from index 1
    subset = predictions_inv[1:]

# Convert to string and remove square brackets
    subset_str = np.array_str(subset).replace('[', '').replace(']', '')

# Split the string into a list of strings using both commas and newlines as delimiters
    subset_list = subset_str.split()

# Convert each string to a float and append to a new list
    subset_floats = []
    for value in subset_list:
        subset_floats.append(float(value.strip()))
    # predictions_inv = np.delete(predictions_inv, 0)
      # Drop the last actual value
    print(subset_floats)
    y_test_inv = np.delete(y_test_inv, 6)


@app.route('/api', methods=['GET'])
def api():
    getData()
    global subset_floats
    data = {
        'day1': {'open': subset_floats[1]},
        'day2': {'open': subset_floats[2], },
        'day3': {'open': subset_floats[3]},
        'day4': {'open': subset_floats[4]},
        'day5': {'open': subset_floats[5]},
    }
    return json.dumps(data)


if __name__ == '__main__':
    app.run()
