import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc

def lgb_modelfit_nocv(params, dtrain, dtrain_target, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.01,
        'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain_target[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, 
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}



    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=xgvalid,
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=50, 
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    # print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1

path = '/media/alaa/Study/Houses_price_prediction/'

dtypes = {

        'OverallQual'  : 'float32',
        'YearBuilt'    : 'float16',
        'TotalBsmtSF'  : 'float32',
        'GrLivArea'    : 'float32',
        'GarageCars'   : 'float32',
        'SalePrice'    : 'float32',
        'Id'           : 'uint32'
        # 'TotRmsAbvGrd' : 'float32',
        # 'YearRemodAdd' : 'float32',
        # 'LotFrontage'  : 'float32',
        # 'GarageYrBlt'  : 'float16',
        }


# [u'Id', u'MSSubClass', u'MSZoning', u'LotFrontage', u'LotArea',
#        u'Street', u'Alley', u'LotShape', u'LandContour', u'Utilities',
#        u'LotConfig', u'LandSlope', u'Neighborhood', u'Condition1',
#        u'Condition2', u'BldgType', u'HouseStyle', u'OverallQual',
#        u'OverallCond', u'YearBuilt', u'YearRemodAdd', u'RoofStyle',
#        u'RoofMatl', u'Exterior1st', u'Exterior2nd', u'MasVnrType',
#        u'MasVnrArea', u'ExterQual', u'ExterCond', u'Foundation', u'BsmtQual',
#        u'BsmtCond', u'BsmtExposure', u'BsmtFinType1', u'BsmtFinSF1',
#        u'BsmtFinType2', u'BsmtFinSF2', u'BsmtUnfSF', u'TotalBsmtSF',
#        u'Heating', u'HeatingQC', u'CentralAir', u'Electrical', u'1stFlrSF',
#        u'2ndFlrSF', u'LowQualFinSF', u'GrLivArea', u'BsmtFullBath',
#        u'BsmtHalfBath', u'FullBath', u'HalfBath', u'BedroomAbvGr',
#        u'KitchenAbvGr', u'KitchenQual', u'TotRmsAbvGrd', u'Functional',
#        u'Fireplaces', u'FireplaceQu', u'GarageType', u'GarageYrBlt',
#        u'GarageFinish', u'GarageCars', u'GarageArea', u'GarageQual',
#        u'GarageCond', u'PavedDrive', u'WoodDeckSF', u'OpenPorchSF',
#        u'EnclosedPorch', u'3SsnPorch', u'ScreenPorch', u'PoolArea', u'PoolQC',
#        u'Fence', u'MiscFeature', u'MiscVal', u'MoSold', u'YrSold', u'SaleType',
#        u'SaleCondition', u'SalePrice'],
#       dtype='object')

print('load train...')
train_df = pd.read_csv(path+"train.csv", dtype=dtypes, usecols=['OverallQual','YearBuilt','TotalBsmtSF','GrLivArea', 'GarageCars','SalePrice']) #'TotRmsAbvGrd','YearRemodAdd','LotFrontage','GarageYrBlt',
train_df = train_df.reindex(
    np.random.permutation(train_df.index)) # randomize the order of the training data
print('load test...')
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['Id', 'OverallQual','YearBuilt','TotalBsmtSF','GrLivArea', 'GarageCars'])#['Id','OverallQual','YearBuilt','TotalBsmtSF','GrLivArea','GarageCars','TotRmsAbvGrd','YearRemodAdd','LotFrontage','GarageYrBlt'])


len_train = len(train_df)
# train_df=train_df.append(test_df)

# del test_df
gc.collect()


def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)

def normalize_linear_scale(examples_dataframe):
  """Returns a version of the input `DataFrame` that has 
     all its features normalized linearly."""
  return examples_dataframe.apply(linear_scale, axis=0)

def log_normalize(series):
  return series.apply(lambda x:math.log(abs(x+1.0)))

def clip(series, clip_to_min, clip_to_max):
  return series.apply(lambda x:(
    min(max(x, clip_to_min), clip_to_max)))

def z_score_normalize(series):
  mean = series.mean()
  std_dv = series.std()
  return series.apply(lambda x:(x - mean) / std_dv)

def binary_threshold(series, threshold):
  return series.apply(lambda x:(1 if x > threshold else 0))

def normalize(examples_dataframe, norm_type, clip_to_min=None, clip_to_max=None, threshold=None):
  """
  norm_type = 'log', 'clipping', 'z_score_norm', or 'binary_threshold'
    clip_to_min & clip_to_max only used in case of norm_type='clipping'
    threshold only used in case of norm_type='binary_threshold'
    
  Note: this function could be used to do different kind of normalization on different features
  by calling it multiple time for every group of features while specifying norm_type
  Returns a version of the input `DataFrame` that has all its features normalized."""
  if norm_type == 'log':
    return examples_dataframe.apply(log_normalize, axis=0)
  elif norm_type == 'clipping':
    return examples_dataframe.apply(clip, args=(clip_to_min, clip_to_max), axis=0)
  elif norm_type == 'z_score_norm':
    return examples_dataframe.apply(z_score_normalize, axis=0)
  elif norm_type == 'binary_threshold':
    return examples_dataframe.apply(binary_threshold, args=(threshold), axis=0)



train_df.info()

# test_df = train_df[len_train:]
# print(len(test_df))
# val_df = train_df[(len_train-1000):len_train]
# print(len(val_df))
# train_df = train_df[:(len_train-1000)]
# print(len(train_df))

target = 'SalePrice'
predictors = ['OverallQual','YearBuilt','TotalBsmtSF','GrLivArea', 'GarageCars'] # ['OverallQual','YearBuilt','TotalBsmtSF','GrLivArea', 'GarageCars','TotRmsAbvGrd','YearRemodAdd','LotFrontage','GarageYrBlt'] 
# categorical = ['app','device','os', 'channel', 'hour']


sub = pd.DataFrame()
sub['Id'] = test_df['Id'].astype('int')

gc.collect()

print("Training...")
# params = {
#     'learning_rate': 0.1,
#     # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
#     'num_leaves': 1400,  # we should let it be smaller than 2^(max_depth)
#     'max_depth': 4,  # -1 means no limit
#     'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
#     'max_bin': 100,  # Number of bucketed bin for feature values
#     'subsample': .7,  # Subsample ratio of the training instance.
#     'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
#     'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
#     'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
#     # 'scale_pos_weight':200, # because training data is extremely unbalanced 
#     'reg_alpha': 0,  # L1 regularization term on weights
#     'reg_lambda': 0, # L2 regularization term on weights
# }

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 1000,
    'learning_rate': 0.1,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'verbose': 0
}

scaled_trian = normalize_linear_scale(train_df[predictors])
scaled_test = normalize_linear_scale(test_df[predictors])
# print scaled_trian
# print scaled_test
bst = lgb_modelfit_nocv(params, 
                        scaled_trian,
                        train_df, 
                        scaled_test,
                        predictors, 
                        target, 
                        early_stopping_rounds=30, 
                        verbose_eval=True, 
                        num_boost_round=500)
                        # categorical_features=categorical) # the returned model will be the best iteration model ever

# del train_df
# # del val_df
# gc.collect()

print("Predicting...")
sub['SalePrice'] = bst.predict(scaled_test) #test_df[predictors]
filename = 'sub_lgb_trail_4.csv'
print("writing in...", filename)
sub.to_csv(filename,index=False)
print("done...")
print(sub.info())



# # # ### This peace of art graph is showing the Top 10 features mostly correlated with SalesPrice (according to their Correlation Coeffecient) and they are as follows in descending order:
# # #     - OverallQual
# # #     - GrLivArea
# # #     - GarageCars
# # #     - GarageArea
# # #     - TotalBsmtSF 
# # #     - 1stFlrSF
# # #     - FullBath
# # #     - TotRmsAbvGrd
# # #     - YearBuilt
# # # ### Some Notes about SalesPrice Correlation Matrix:
# # #     - GarageCars and GarageArea are one of the most correlated features, but they are so much correlated in themselves 
# # #     so its a ggod practice to just use only one of them, so we 'll take GarageCars as it is more corrleted with SalePrice. And it's the same case with TotalBsmtSF & 1stFlrSF and GrLivArea & TotRmsAbvGrd.
# # #     - YearBuilt --> is slightly correlated with SalePrice ... which is slightly unreasonable!!!