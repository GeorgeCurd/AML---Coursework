from pandas import read_csv, Series
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from DataProcessing import test_Y, train_Y, test_normX, train_normX, X_names

#Identify key features using random forest/extra trees
model = ExtraTreesClassifier()
model.fit(train_normX, train_Y)
feat_importance = Series(model.feature_importances_, index=X_names)
print(feat_importance.index)
reduced = SelectFromModel(model, prefit=True, threshold=-np.inf, max_features=10)
train_normX_ETC_FS = reduced.transform(train_normX)





