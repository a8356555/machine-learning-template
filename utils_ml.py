from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import joblib
from functools import partial

from .models import model_factory


NUMERICAL_COLS = ['feature_a']

ORDINAL_COLS_MIN_MAX_DICT = {
    'feature_b': (1, 10),
} 

CATEGORICAL_COLS_DICT = {
    'feature_c': 10,
}

YES_NO_COLS = ['feature_d']

ALL_COLS = NUMERICAL_COLS+list(ORDINAL_COLS_MIN_MAX_DICT)+YES_NO_COLS+list(CATEGORICAL_COLS_DICT)

def _get_ohe(x, categorical_num=1):
    data = [0]*categorical_num
    data[x] = 1
    return data


class StatisticalDataNormalizer:
    def __init__(self, debug=False):
        self.debug = debug
    
    def normalize(self, df, seed):
        df, used_numerical_cols = self._normalize_numerical_cols(df, seed)
        df, used_ordinal_cols = self._normalize_ordinal_cols(df, seed)
        df, used_categorical_cols, used_categorical_feature_names = self._normalize_categorical_cols(df, seed)
        used_cols = used_numerical_cols+used_ordinal_cols+used_categorical_cols + [col for col in YES_NO_COLS if col in df]
        used_feature_names = used_numerical_cols+used_ordinal_cols+used_categorical_feature_names + \
                             [col for col in YES_NO_COLS if col in df]
        
        df = df.rename(columns={k:'feat_'+k for k in used_cols})
        
        used_feature_cols = ['feat_'+k for k in used_cols]
        return df, used_feature_cols, used_feature_names
        
    def _normalize_numerical_cols(self, df, seed):
        used_cols = []
        for col in NUMERICAL_COLS:
            if col not in df.columns:
                continue
            if self.debug:
                print(f'numerical col {col} before\n', df[col].describe())
            mean = df[df[f'seed{seed}'].isin(['train', 'valid'])][col].mean()
            std = df[df[f'seed{seed}'].isin(['train', 'valid'])][col].std()
            df[col] = ((df[col] - mean) / std).fillna(0)
            if self.debug:
                print(f'numerical col {col} after\n', df[col].describe())
            used_cols.append(col)
        return df, used_cols
    
    def _normalize_ordinal_cols(self, df, seed):
        used_cols = []
        for col, (_min, _max) in ORDINAL_COLS_MIN_MAX_DICT.items():
            if col not in df.columns:
                continue
            if self.debug:
                print(f'ordinal col {col} before\n', df[col].describe())
            df[col] = df[col].clip(_min, _max)
            df[col] = (df[col]-_min)/(_max-_min)
            used_cols.append(col)
            if self.debug:
                print(f'ordinal col {col} after\n', df[col].describe())
        return df, used_cols
    
    def _normalize_categorical_cols(self, df, seed):
        used_cols = []
        used_feature_names = []
        for col, categorical_num in CATEGORICAL_COLS_DICT.items():
            if col not in df.columns:
                continue
            if self.debug:
                print(f'categorical col {col} before\n', df[col].describe())
            if categorical_num > 0:
                df[col] = df[col].apply(int)
                data = df[col].apply(lambda x:_get_ohe(x, categorical_num=categorical_num)).to_list()
                columns = [f'{col}_{i}' for i in range(categorical_num)]
                df_cat = pd.DataFrame(data, columns=columns)
            else:
                df[col] = df[col].apply(str)
                # get dummies through exist values
                df_cat = pd.get_dummies(df[col])
                columns = [col+'_'+str(cat) for cat in df_cat.columns]
                df_cat.columns = columns
            if self.debug:
                print(f'categorical col {col} after\n', df_cat)
            df = pd.concat([df.reset_index(drop=True), df_cat], axis=1)
            used_cols += columns
            used_feature_names.append(col)
        return df, used_cols, used_feature_names



class Trainer:
    def __init__(self, df, all_seed_list, label_col='gt', 
                 enable_shap_plot=False, model_name_list=['LGBM', 'LR'], debug=False, max_depth=-1, n_estimators=100):
        self.df = df
        self.all_seed_list = all_seed_list
        self.label_col = label_col
        self.enable_shap_plot = enable_shap_plot
        self.model_name_list = model_name_list
        self.is_multi_classes = df[label_col].nunique() > 2
        self.statistical_data_normalizer = StatisticalDataNormalizer(debug=debug)
        self.debug = debug
        
    def _get_shap_values(self, model, data, feature_cols):
        x_train, y_train, x_test, y_test = data
        df_shap = pd.DataFrame(x_train, columns=[col.replace('feat_', '') for col in feature_cols])
        # TODO: 只使用 x_train 但解決 additivity check 問題
        explainer = shap.Explainer(model, df_shap)
        shap_values = explainer(df_shap)
        return shap_values
        
    
    def _plot_beeswarm(self, shap_values, model_name, seed):
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.title(model_name)
        plt.show()
        
    def cross_validate(self):
        all_model_dict = {}
        feature_importances = []
        for seed in self.all_seed_list:
            models = self._prepare_models(seed)
            *data, feature_cols, feature_names = self._prepare_data(seed)
            for model in models:
                model = self.fit(model, data)
                self._write_outputs_to_df(model, data, seed)
                model_name = f'{str(model).split("(")[0]}({seed})'
                all_model_dict[model_name] = model                
                if not str(model).startswith('Logis'):
                    # TODO: SHAP additivity check error with LGBM 
                    continue
                
                shap_values = self._get_shap_values(model, data, feature_cols)
                feature_importances.append(list(np.abs(shap_values.values).mean(0))+[f'{str(model).split("(")[0]}', seed])
                if self.enable_shap_plot:
                    self._plot_beeswarm(shap_values, model_name, seed)
                    
        df_feat_importances = pd.DataFrame(feature_importances, 
                                           columns=[col.replace('feat_', '') for col in feature_cols]+['model', 'seed'])

        return self.df, all_model_dict, df_feat_importances
    
    def _prepare_models(self, seed):
            return [self._prepare_model(name, seed) for name in self.model_name_list]
    
    def _prepare_model(self, name, seed):
        return model_factory.create(name)(random_state=seed)
        
    def _prepare_data(self, seed):
        normalized_df, feature_cols, feature_names = self.statistical_data_normalizer.normalize(self.df.copy(), seed)
        df_train = normalized_df[normalized_df[f'seed{seed}'].isin(['train', 'valid'])]
        df_test = normalized_df[normalized_df[f'seed{seed}'] == 'test']

        # col_num = len(feature_cols)
        x_train = np.array(df_train[feature_cols], dtype=np.float64) # .reshape(-1, col_num)
        y_train = np.array(df_train[self.label_col])
        x_test = np.array(df_test[feature_cols], dtype=np.float64)
        y_test = np.array(df_test[self.label_col])
        return x_train, y_train, x_test, y_test, feature_cols, feature_names
    
    def fit(self, model, data):
        x_train, y_train, x_test, y_test = data
        model = model.fit(x_train, y_train)
        return model
    
    def _write_outputs_to_df(self, model, data, seed):
        x_train, y_train, x_test, y_test = data
        col = f'{str(model).split("(")[0]}({seed})'
        self.df[col] = None
        self.df.loc[self.df[f'seed{seed}'] == 'train', col] = self._get_outputs(model, x_train)
        self.df.loc[self.df[f'seed{seed}'] == 'test', col] = self._get_outputs(model, x_test)
        
    def _get_outputs(self, model, inputs):
        try:
            predictions = model.predict_proba(inputs)
        except:
            predictions = model.predict(inputs)
        if predictions.shape[1] == 2:
            predictions = predictions[:, 1]
        
        return predictions

    def save_model_trained_with_all_data(self, model_name='LR', save_folder=''):
        label_dict = {}
        seed = 99999
        self.df[f'seed{seed}'] = 'train'
        test_embryo_id = self.df.groupby(self.label_col).first()['embryo_id']
        self.df.loc[self.df['embryo_id'].isin(test_embryo_id), f'seed{seed}'] = 'test'
        
        model = self._prepare_model(model_name, seed)
        *data, feature_cols, feature_names = self._prepare_data(seed)
        model = self.fit(model, data)
        
        save_label_name = label_dict[self.label_col]
        save_model_name = f'{save_label_name}_model'
        save_model_path = os.path.join(save_folder, save_model_name)
        feature_col_txt = os.path.join(save_folder, f'{save_model_name}_used_feature_cols.txt')
        feature_name_txt = os.path.join(save_folder, f'{save_model_name}_used_feature_names.txt')
        print(f'saved at {save_model_path}, {feature_col_txt}, {feature_name_txt}')
        joblib.dump(model, save_model_path)
        with open(feature_col_txt, 'w') as f:
            for feature in feature_cols:
                f.write(f'{feature}\n')
                
        with open(feature_name_txt, 'w') as f:
            for feature_name in feature_names:
                f.write(f'{feature_name}\n')
        return model
        
        
        

class MetricCalculator:
    def __init__(self, all_model_dict, df_outputs, label_col='gt', acc_threshold=0.5):
        self.all_model_dict = all_model_dict
        self.df = df_outputs
        self.label_col = label_col
        self.acc_threshold = acc_threshold
    
    def calculate(self):
        metrics_data = []
        for model_pred_col in self.all_model_dict:
            seed = model_pred_col.split('(')[-1].replace(')', '')
            for phase in ['train', 'test']:
                df_temp = self.df[self.df[f'seed{seed}'] == phase]
                gt = df_temp[self.label_col]
                pred = df_temp[model_pred_col]
                metrics = self._get_metrics(gt, pred)
                metrics_data.append([model_pred_col.split('(')[0], 'all', phase, seed]+metrics)
        df_metrics = pd.DataFrame(metrics_data, 
                                  columns=['model', 'phase', 'seed', 'auc', 'acc', 'tpr(sensitivity/recall)', 'fpr', 'specificity', 'precision',
                                           'tn', 'fp', 'fn', 'tp', 'size'])
        return df_metrics
            
    def _get_metrics(self, gt, pred):
        auc = metrics.roc_auc_score(gt, pred)
        acc = metrics.accuracy_score(gt, pred>=self.acc_threshold)
        tn, fp, fn, tp = metrics.confusion_matrix(gt, pred>=self.acc_threshold).ravel()
        specificity = tn / (tn+fp)
        acc = (tn+tp)/(fn+fp+tn+tp)
        fpr = fp / (fp+tn)
        tpr = tp / (tp+fn)
        precision = tp/(tp+fp)
        size = len(gt)
        return [auc, acc, tpr, fpr, specificity, precision, tn, fp, fn, tp, size]