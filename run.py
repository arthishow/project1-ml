from proj1_helpers import *
from helper import preprocess_datasets, generate_prediction

DATA_PATH = 'data/'
PREDICTION_PATH = 'predictions/'
y_tr, x_tr, ids_tr = load_csv_data(DATA_PATH + "train.csv")
y_te, x_te, ids_te = load_csv_data(DATA_PATH + "test.csv")

x_tr_0, y_tr_0, x_tr_1, y_tr_1, x_tr_2, y_tr_2, x_tr_3, y_tr_3, x_te_0, x_te_1, x_te_2, x_te_3, jet_num_te = preprocess_datasets(x_tr, y_tr, x_te, y_te)

predicted_y_te = generate_prediction(x_tr_0, y_tr_0, x_tr_1, y_tr_1, x_tr_2, y_tr_2, x_tr_3, y_tr_3, x_te_0, x_te_1, x_te_2, x_te_3, jet_num_te)

create_csv_submission(ids_te, predicted_y_te, PREDICTION_PATH + "output.csv")
