'''
Goal of LSTM microservice:
1. LSTM microservice will accept the GitHub data from Flask microservice and will forecast the data for next 1 year based on past 30 days
2. It will also plot three different graph (i.e.  "Model Loss", "LSTM Generated Data", "All Issues Data") using matplot lib 
3. This graph will be stored as image in Google Cloud Storage.
4. The image URL are then returned back to Flask microservice.
'''
# Import all the required packages
from flask import Flask, jsonify, request, make_response
import os
from dateutil import *
from datetime import timedelta, date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from flask_cors import CORS

# Tensorflow (Keras & LSTM) related packages
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import LSTM
#from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
# Import required storage package from Google Cloud Storage
from google.cloud import storage
from google.oauth2 import service_account

# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)
# Initlize Google cloud storage client
credentials = service_account.Credentials.from_service_account_file("D:\Education\mastersIITC\spring 24\spm_jinit\hw5\code\LSTM-forecast\lstm-forecast-420421-db87ad05f096.json")
client = storage.Client(credentials=credentials)

LOCAL_IMAGE_PATH = "D:\Education\mastersIITC\spring 24\spm_jinit\hw5\code\LSTM-forecast\static\images\\"
BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm_jspscm70p/')
NO_DATA_URL = BASE_IMAGE_PATH + "no_data.png"

# Add response headers to accept all types of  requests

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

#  Modify response headers when returning to the origin

def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

'''
API route path is  "/api/forecast"
This API will accept only POST request
'''

@app.route('/api/forecast', methods=['POST'])
def forecast():
    body = request.get_json()
    issues = body["issues"]
    type = body["type"]
    repo_name = body["repo"]
    data_frame = pd.DataFrame(issues)
    df1 = data_frame.groupby([type], as_index=False).count()
    if(df1.empty):
        json_response = {
        "model_loss_image_url": NO_DATA_URL,
        "lstm_generated_image_url": NO_DATA_URL,
        "all_issues_data_image": NO_DATA_URL,
        "created_issues_max_day": NO_DATA_URL,
        "closed_issues_max_day": NO_DATA_URL
        }
        return jsonify(json_response)
        #df1.fillna(0)
        #d = date.today().strftime('%Y-%m-%d')
        #df1.loc[0] = [d,0,0,0,0,0]
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']

    df['ds'] = df['ds'].astype('datetime64[ns]')
    array = df.to_numpy()
    x = np.array([time.mktime(i[0].timetuple()) for i in array])
    y = np.array([i[1] for i in array])

    lzip = lambda *x: list(zip(*x))

    days = df.groupby('ds')['ds'].value_counts()
    Y = df['y'].values
    X = lzip(*days.index.values)[0]
    firstDay = min(X)

    '''
    To achieve data consistancy with both actual data and predicted values, 
    add zeros to dates that do not have orders
    [firstDay + timedelta(days=day) for day in range((max(X) - firstDay).days + 1)]
    '''
    Ys = [0, ]*((max(X) - firstDay).days + 1)
    days = pd.Series([firstDay + timedelta(days=i)
                      for i in range(len(Ys))])
    for x, y in zip(X, Y):
        Ys[(x - firstDay).days] = y

    # Modify the data that is suitable for LSTM
    Ys = np.array(Ys)
    Ys = Ys.astype('float32')
    Ys = np.reshape(Ys, (-1, 1))
    # Apply min max scaler to transform the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    Ys = scaler.fit_transform(Ys)
    # Divide training - test data with 80-20 split
    train_size = int(len(Ys) * 0.80)
    test_size = len(Ys) - train_size
    train, test = Ys[0:train_size, :], Ys[train_size:len(Ys), :]
    print('train size:', len(train), ", test size:", len(test))

    # Create the training and test dataset
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    '''
    Look back decides how many days of data the model looks at for prediction
    Here LSTM looks at approximately one month data
    '''
    look_back = 30
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Verifying the shapes
    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    # Model to forecast
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    #tf.
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model with training data and set appropriate hyper parameters
    history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

    '''
    Creating image URL
    BASE_IMAGE_PATH refers to Google Cloud Storage Bucket URL.Add your Base Image Path in line 145
    if you want to run the application local
    LOCAL_IMAGE_PATH refers local directory where the figures generated by matplotlib are stored
    These locally stored images will then be uploaded to Google Cloud Storage
    '''
    #BASE_IMAGE_PATH = os.environ.get(
        #'BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm_jspscm70p/')
    # DO NOT DELETE "static/images" FOLDER as it is used to store figures/images generated by matplotlib
    #LOCAL_IMAGE_PATH = "static/images/"
    #LOCAL_IMAGE_PATH = "D:\Education\mastersIITC\spring 24\spm_jinit\hw5\code\LSTM-forecast\static\images\\"

    # Creating the image path for model loss, LSTM generated image and all issues data image
    if 'issue_type' in body:
        type = body['issue_type']
    CREATED_ISSUES_MAX_DAY = "created_issues_max_day_" + type + "_" + repo_name + ".png"
    CREATED_ISSUES_MAX_DAY_URL = BASE_IMAGE_PATH + CREATED_ISSUES_MAX_DAY

    CLOSED_ISSUES_MAX_DAY = "closed_issues_max_day_" + type + "_" + repo_name + ".png"
    CLOSED_ISSUES_MAX_DAY_URL = BASE_IMAGE_PATH + CLOSED_ISSUES_MAX_DAY

    CLOSED_ISSUES_MAX_MONTH = "closed_issues_max_month_" + type + "_" + repo_name + ".png"
    CLOSED_ISSUES_MAX_MONTH_URL = BASE_IMAGE_PATH + CLOSED_ISSUES_MAX_MONTH

    
    MODEL_LOSS_IMAGE_NAME = "model_loss_" + type +"_"+ repo_name + ".png"
    MODEL_LOSS_URL = BASE_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME

    LSTM_GENERATED_IMAGE_NAME = "lstm_generated_data_" + type +"_" + repo_name + ".png"
    LSTM_GENERATED_URL = BASE_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME

    ALL_ISSUES_DATA_IMAGE_NAME = "all_issues_data_" + type + "_"+ repo_name + ".png"
    ALL_ISSUES_DATA_URL = BASE_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME

    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'lstm_jspscm70p')

    # Model summary()

    # Plot the model loss image
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss For ' + type)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)

    # Predict issues for test data
    y_pred = model.predict(X_test)

    # Plot the LSTM Generated image
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(np.arange(0, len(Y_train)), Y_train, 'g', label="history")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)),
             Y_test, marker='.', label="true")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)),
             y_pred, 'r', label="prediction")
    axs.legend()
    axs.set_title('LSTM Generated Data For ' + type)
    axs.set_xlabel('Time Steps')
    if 'issue_type' in body:
        axs.set_ylabel(body['issue_type'])
    else:
        axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)

    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('All Issues Data')
    axs.set_xlabel('Date')
    if 'issue_type' in body:
        axs.set_ylabel(body['issue_type'])
    else:
        axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME)

    month_names = ['January', 'February', 'March', 'April', 'May', 'June','July','August','Septeber', 'October', 'November','December']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_issues = pd.DataFrame(issues)
    df_issues['created_at'] = pd.to_datetime(df_issues['created_at'], errors='coerce')
    df_issues_count = df_issues.groupby(df_issues['created_at'].dt.day_name()).size()
    df_issues_count = pd.DataFrame({'Created_On': df_issues_count.index, 'Count': df_issues_count.values})
    df_issues_count = df_issues_count.groupby(['Created_On']).sum().reindex(day_names)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(df_issues_count['Count'], label='Issues')
    plt.title('day of the week maximum number of issues created')
    if 'issue_type' in body:
        axs.set_ylabel("Number of "+body['issue_type'])
    else:
        axs.set_ylabel('Number of Issues')
    plt.xlabel('Days')
    plt.savefig(LOCAL_IMAGE_PATH + CREATED_ISSUES_MAX_DAY)


    df_issues['closed_at'] = pd.to_datetime(df_issues['closed_at'], errors='coerce')
    df_issues_close_count = df_issues.groupby(df_issues['closed_at'].dt.day_name()).size()
    df_issues_close_count = pd.DataFrame({'Closed_On': df_issues_close_count.index, 'Count': df_issues_close_count.values})
    df_issues_close_count.set_index('Closed_On', inplace = True)
    df_issues_close_count = df_issues_close_count.reindex(day_names, fill_value=0)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(df_issues_close_count['Count'], label='Issues')
    plt.title('day of the week maximum number of issues closed')
    if 'issue_type' in body:
        axs.set_ylabel("Number of "+body['issue_type'])
    else:
        axs.set_ylabel('Number of Issues')
    plt.xlabel('Days')
    plt.savefig(LOCAL_IMAGE_PATH + CLOSED_ISSUES_MAX_DAY)


    
    df_issues_month = pd.DataFrame(issues)
    df_issues_month['closed_at'] = pd.to_datetime(df_issues_month['closed_at'], errors='coerce')
    df_issues_close_count_month = df_issues_month.groupby(df_issues_month['closed_at'].dt.month_name()).size()
    df_issues_close_count_month = pd.DataFrame({'Closed_On': df_issues_close_count_month.index, 'Count': df_issues_close_count_month.values})
    df_issues_close_count_month.set_index('Closed_On', inplace = True)
    df_issues_close_count_month = df_issues_close_count_month.reindex(month_names, fill_value=0)
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(df_issues_close_count_month['Count'], label='Issues')
    plt.title('month of the year that has maximum number of issues closed')
    if 'issue_type' in body:
        axs.set_ylabel("Number of "+body['issue_type'])
    else:
        axs.set_ylabel('Number of Issues')
    plt.xlabel('Months')
    plt.savefig(LOCAL_IMAGE_PATH + CLOSED_ISSUES_MAX_MONTH)
    

    # Uploads an images into the google cloud storage bucket
    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(CLOSED_ISSUES_MAX_MONTH)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + CLOSED_ISSUES_MAX_MONTH)
    new_blob = bucket.blob(MODEL_LOSS_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)
    new_blob = bucket.blob(ALL_ISSUES_DATA_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME)
    new_blob = bucket.blob(LSTM_GENERATED_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)
    new_blob = bucket.blob(CREATED_ISSUES_MAX_DAY)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + CREATED_ISSUES_MAX_DAY)
    new_blob = bucket.blob(CLOSED_ISSUES_MAX_DAY)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + CLOSED_ISSUES_MAX_DAY)

    # Construct the response
    json_response = {
        "model_loss_image_url": MODEL_LOSS_URL,
        "lstm_generated_image_url": LSTM_GENERATED_URL,
        "all_issues_data_image": ALL_ISSUES_DATA_URL,
        "created_issues_max_day": CREATED_ISSUES_MAX_DAY_URL,
        "closed_issues_max_day": CLOSED_ISSUES_MAX_DAY_URL,
        "closed_issues_max_month": CLOSED_ISSUES_MAX_MONTH_URL
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)


# Run LSTM app server on port 8080
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
