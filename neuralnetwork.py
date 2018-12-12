__author__ = 'Jurgen Amedani'

import deepmatcher as dm
import torch


train, validation, test = dm.data.process(path='dataset',
    train='GS_Dogfood_300_test.csv', validation='GS_Dogfood_300_test.csv', test='GS_Dogfood_300_test.csv')

model = dm.MatchingModel(attr_summarizer='hybrid')
model.run_train(
    train,
    validation,
    epochs=10,
    batch_size=16,
    best_save_path='hybrid_model.pth',
    pos_neg_ratio=3)
train_table = train.get_raw_table()
train_table.head()
model.run_eval(test)
valid_predictions = model.run_prediction(validation, output_attributes=True)
valid_predictions.head()