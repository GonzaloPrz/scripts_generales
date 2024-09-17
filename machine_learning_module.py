import numpy as np
import random
import pandas as pd
from pathlib import Path
from copy import copy

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,accuracy_score, roc_curve, f1_score, precision_score, recall_score, auc
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, StratifiedKFold,train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel as sfm
from sklearn.impute import KNNImputer
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

from matplotlib import pyplot as plt

from skopt import BayesSearchCV

from tqdm import tqdm

from copy import deepcopy 

import torch,itertools,pickle

import warnings

from expected_cost.ec import *
from expected_cost.utils import *
from psrcal.losses import LogLoss, CalLossLogLoss
from psrcal.calibration import calibrate, AffineCal, AffineCalLogLoss

class Model():
    def __init__(self,model,scaler,calibrator=None):
        self.scaler = scaler
        self.model = model
        self.calibrator = None

    def train(self,X,y):   
        features = X.columns
        
        X_t = pd.DataFrame(columns=features,data=self.scaler.fit_transform(X[features].values))

        self.model.fit(X_t,y)

        #Identify convergence warning messages and analyze them
        '''
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            self.model.fit(X_t,y)
            if len(w) > 0:
                for i in range(len(w)):
                    print(w[i].message)
        '''
    def eval(self,X):
        
        features = X.columns
        X_t = pd.DataFrame(columns=features,data=self.scaler.transform(X[features].values))

        if hasattr(self.model,'predict_log_proba'):
            logpost = self.model.predict_log_proba(X_t)
        else:
            prob = self.model.predict_proba(X_t)
            prob = np.clip(prob,1e-6,1-1e-6)
            logpost = np.log(prob)
           
        return logpost
    
def _get_priors_and_weights(labels, priors):
    data_priors = torch.bincount(labels)/float(labels.shape[0])
    if priors is None:
        priors = data_priors
        weights = torch.tensor(1.0)
    else:
        weights = priors[labels]/data_priors[labels] 
    return priors, weights

def LogLoss(log_probs, labels, norm=True, priors=None):
    labels = labels.clone().detach().long()

    priors, weights = _get_priors_and_weights(labels, priors)
    norm_factor = LogLoss(torch.log(priors.expand(log_probs.shape[0],-1)), labels, norm=False, priors=priors) if norm else 1.0

    # The loss on each sample is weighted by the inverse of the
    # frequency of the corresponding class in the test data
    # times the external prior
    ii = torch.arange(len(labels))
    losses = -log_probs[ii, labels]
    score  = torch.mean(weights*losses)

    return score / norm_factor

def bootstrap_scores(y_scores,y_true,metrics_names,IDs,costs=None,priors=None,n_bootstrap=100):
    """
    Bootstraps the scores and true labels.

    Args:
        y_scores: The predicted scores.
        y_true: The true labels.
        n_bootstrap: The number of bootstrap iterations. Defaults to 100.)
    
    Returns:

        y_scores_bt: The bootstrapped scores.
        y_true_bt: The bootstrapped true labels.

    Note:
        The function bootstraps the scores and true labels to estimate the variability of the evaluation metrics.
    """
    metrics = dict([(metric,np.empty(0)) for metric in metrics_names])
    y_scores_bt = np.empty((0,2))
    y_true_bt = np.empty(0)
    IDs_bt = np.empty(0)

    for _ in range(n_bootstrap):
        index_bt = np.random.choice(range(len(y_true)),size=len(y_true),replace=True)
        y_scores_bt = np.vstack((y_scores_bt,y_scores[index_bt,:]))
        y_true_bt = np.hstack([y_true_bt,y_true[index_bt]])
        IDs_bt = np.hstack([IDs_bt,IDs[index_bt]])
        #Change inf values to a very large number
        y_scores_bt = np.clip(y_scores_bt,-1e6,1e6)
        
        y_pred = bayes_decisions(scores=y_scores[index_bt,:],costs=costs,priors=priors,score_type='log_posteriors')[0]

        for m in metrics_names:
            if m == 'norm_cross_entropy':
                try: 
                    metrics[m] = np.hstack((metrics[m],LogLoss(log_probs=torch.tensor(y_scores[index_bt,:]),labels=torch.tensor(np.array(y_true[index_bt])),priors=torch.tensor(priors)).detach().numpy())) if priors is not None else np.hstack((metrics[m],LogLoss(log_probs=torch.tensor(y_scores_bt),labels=torch.tensor(np.array(y_true_bt))).detach().numpy()))
                except:
                    metrics[m] = np.hstack((metrics[m],np.nan))
            elif m == 'norm_expected_cost':
                try:
                    metrics[m] = np.hstack((metrics[m],np.array(average_cost(targets=y_true[index_bt],decisions=y_pred,costs=costs,priors=priors,adjusted=True))))
                except:
                    metrics[m] = np.hstack((metrics[m],np.nan))
            elif m == 'roc_auc':
                try:
                    metrics[m] = np.hstack((metrics[m],np.array(roc_auc_score(y_true=y_true[index_bt],y_score=y_scores[index_bt,1]))))
                except:
                    metrics[m] = np.hstack((metrics[m],np.nan))
            else:
                metrics[m] = np.hstack((metrics[m],np.array(eval(f'{m}_score')(y_true=y_true[index_bt],y_pred=y_pred))))

    return metrics, y_scores_bt, y_true_bt, IDs_bt

def train_models(model_type,scaler,X_dev,y_dev,random_seeds,hyperp,metrics_names,costs=None,priors=None,CV_type=None,IDs=None,boot_train=False,boot_val=0,n_features_to_select=None):
    """
    Trains a model, tunes hyperparameters, calibrates probabilities, and evaluates performance.

    Args:
        model: The machine learning model to be trained and evaluated.
        scaler: The scaler object used to normalize the input features.
        X_dev: The development set input features for training and validation.
        y_dev: The development set target variable for training and validation.
        random_seed: The random seed for reproducibility.
        hyperp_space: The hyperparameter space for randomized search.
        costs: The cost matrix used to calculate expected costs. Defaults to None, which sets it to a 2x2 cost matrix with zeros on the diagonal and ones elsewhere.
        priors: The prior probabilities of the target classes. Defaults to None, which uses the empirical class distribution.
        CV_out: The outer cross-validation object or iterator. Defaults to None, which sets it to StratifiedShuffleSplit with shuffling and the provided random seed.
        CV_in: The inner cross-validation object or iterator. Defaults to None, which sets it to StratifiedShuffleSplit with shuffling and the provided random seed.
        cal: Whether to calibrate the probabilities. Defaults to True.
        cheat: Whether to use true labels for calibration. Defaults to False.
        feature_selection: Whether to perform feature selection. Defaults to False.
        IDs: The sample IDs. Defaults to None.
        boot_train: Whether to bootstrap training samples. Defaults to False.
        boot_val: The number of bootstrap iterations for validation. Defaults to 0.

    Returns:
        results: A dictionary containing the evaluation results and other information.
    """

    results = dict()    

    results['model'] = pd.DataFrame(columns=[param for param in hyperp.keys()] + ['random_seed'])

    logpost_val = dict((f'random_seed_{random_seed}',[]) for random_seed in random_seeds)
    logpost_val_bt = dict((f'random_seed_{random_seed}',[]) for random_seed in random_seeds)

    all_metrics = dict((f'random_seed_{random_seed}',dict([(metric,[]) for metric in metrics_names])) for random_seed in random_seeds)
    
    y_true = dict([(f'random_seed_{random_seed}',np.empty(0,dtype=int)) for random_seed in random_seeds])
    y_true_bt = dict([(f'random_seed_{random_seed}',np.empty(0,dtype=int)) for random_seed in random_seeds])

    y_pred = dict([(f'random_seed_{random_seed}',np.empty(0,dtype=int)) for random_seed in random_seeds])
    y_pred_bt = dict([(f'random_seed_{random_seed}',np.empty(0,dtype=int)) for random_seed in random_seeds])

    IDs_train = dict([(f'random_seed_{random_seed}',pd.DataFrame({'fold':[],'ID':[]})) for random_seed in random_seeds])
    IDs_val = dict([(f'random_seed_{random_seed}',pd.DataFrame({'fold':[],'ID':[]})) for random_seed in random_seeds])
    
    IDs_val_bt = dict([(f'random_seed_{random_seed}',pd.DataFrame({'ID':[]})) for random_seed in random_seeds])

    all_features = X_dev.columns
    selected = dict([(feature,0) for feature in all_features])

    for random_seed in random_seeds:
        if CV_type is None:
            CV = StratifiedShuffleSplit(random_state=random_seed)
        else:
            CV = type(CV_type)(n_splits=CV_type.get_n_splits(),random_state=random_seed,shuffle=True) if type(CV_type) is StratifiedKFold else CV_type

        logpost_val[f'random_seed_{random_seed}'] = np.empty((0,2))

        if costs == None:
            costs = CostMatrix([[0,1],[1,0]])

        model = Model(model=model_type(**hyperp),scaler=scaler)


        for k,(train_out,val_idx) in enumerate(CV.split(X_dev,y_dev)):            
            boot_idx = np.random.choice(train_out,replace=True,size=len(train_out)) if boot_train else train_out
            IDs_train[f'random_seed_{random_seed}'] = pd.concat((IDs_train[f'random_seed_{random_seed}'],pd.DataFrame({'fold':[k]*len(IDs[boot_idx]),'ID':IDs[boot_idx]})),axis=0)             
            IDs_val[f'random_seed_{random_seed}'] = pd.concat((IDs_val[f'random_seed_{random_seed}'],pd.DataFrame({'fold':[k]*len(IDs[val_idx]),'ID':IDs[val_idx]})),axis=0)

            X_train = X_dev.iloc[boot_idx]
            y_train = y_dev.iloc[boot_idx]
            if n_features_to_select is not None:
                features = feature_selection(model=model,X=X_train,y=y_train,metric='roc_auc',n_features_to_select=n_features_to_select)
            else:
                features = X_train.columns
            
            model.train(X=X_train[features],y=y_train)

            X_val = X_dev.iloc[val_idx]

            logpost_val[f'random_seed_{random_seed}'] = np.vstack((logpost_val[f'random_seed_{random_seed}'],model.eval(X_val[features])))

            y_true[f'random_seed_{random_seed}'] = np.hstack((y_true[f'random_seed_{random_seed}'],y_dev[val_idx]))

        for feature in all_features:
            selected[feature] += feature in features
        
        IDs_train[f'random_seed_{random_seed}'].reset_index(drop=True,inplace=True)
        IDs_val[f'random_seed_{random_seed}'].reset_index(drop=True,inplace=True)
        
        if boot_val > 0:
            all_metrics[f'random_seed_{random_seed}'], y_scores_bt_, y_true_bt_, IDs_bt = bootstrap_scores(y_scores=logpost_val[f'random_seed_{random_seed}'],y_true=y_true[f'random_seed_{random_seed}'], metrics_names=metrics_names,IDs=IDs_val[f'random_seed_{random_seed}']['ID'],costs=costs,priors=priors,n_bootstrap=boot_val)
            y_true_bt[f'random_seed_{random_seed}'] = y_true_bt_
            IDs_val_bt[f'random_seed_{random_seed}']= IDs_bt
            logpost_val_bt[f'random_seed_{random_seed}'] = y_scores_bt_
        else:
            all_metrics[f'random_seed_{random_seed}'] = get_metrics(y_scores=logpost_val[f'random_seed_{random_seed}'],y_true=y_true[f'random_seed_{random_seed}'],y_pred=y_pred[f'random_seed_{random_seed}'],metrics_names=metrics_names,cmatrix=costs,priors=priors)
            logpost_val_bt[f'random_seed_{random_seed}'] = logpost_val[f'random_seed_{random_seed}']
            y_true_bt[f'random_seed_{random_seed}'] = y_true[f'random_seed_{random_seed}']
            IDs_val_bt[f'random_seed_{random_seed}'] = IDs_val[f'random_seed_{random_seed}'].ID

        y_pred_bt[f'random_seed_{random_seed}'] = bayes_decisions(scores=logpost_val_bt[f'random_seed_{random_seed}'],costs=costs,priors=priors,score_type='log_posteriors')[0]

    results['raw_logpost_val'] = logpost_val_bt
    results['y_true_val'] = y_true_bt
    results['y_pred_val'] = y_pred_bt
    results['IDs_val'] = IDs_val_bt
    results['metrics_val'] = all_metrics
    results['selected_features'] = selected

    return results

def test_model(model,X_test,y_test,metrics_names,IDs,costs=None,priors=None,bootstrap=0):
    y_scores = model.eval(X_test)
    if costs == None:
        costs = CostMatrix([[0,1],[1,0]])
        
    y_pred = bayes_decisions(scores=y_scores,costs=costs,priors=priors,score_type='log_posteriors')[0]
    
    if costs == None:
        costs = CostMatrix([[0,1],[1,0]])
        
    if bootstrap > 0:        
        all_metrics, y_scores_bt, y_true_bt, IDs_bt = bootstrap_scores(y_scores=y_scores,y_true=y_test, metrics_names=metrics_names,IDs=IDs,costs=costs,priors=priors,n_bootstrap=bootstrap)
    else:
        all_metrics = get_metrics(y_scores=y_scores,y_true=y_test,y_pred=y_pred,metrics_names=metrics_names,cmatrix=costs,priors=priors)
        y_scores_bt = y_scores
        y_true_bt = y_test
        IDs_bt = IDs
    y_pred_bt = bayes_decisions(scores=y_scores_bt,costs=costs,priors=priors,score_type='log_posteriors')[0]
    
    return all_metrics,y_scores_bt,y_true_bt,y_pred_bt, IDs_bt

def feature_selection(model,X,y,metric='roc_auc',n_features_to_select=None):
    if n_features_to_select is None:
        n_features_to_select = int(X.shape[1]/2)

    all_features = X.columns
    features = all_features.copy()

    metrics = dict([(feature,np.nan) for feature in all_features])

    for i in range(n_features_to_select):
        for feature in features:
            features_ = features[features != feature]
            model.train(X[features_],y)
            y_scores = model.eval(X[features_])
            y_pred = bayes_decisions(scores=y_scores,costs=None,priors=None,score_type='log_posteriors')[0]
            all_metrics = get_metrics(y_scores=y_scores,y_true=y,y_pred=y_pred,metrics_names=[metric])
            metrics[feature] = all_metrics[metric]
        if 'norm' in metric:
            feature_to_eliminate = min(metrics,key=metrics.get)
        else:
            feature_to_eliminate = max(metrics,key=metrics.get)
        
        features = features[features != feature_to_eliminate]

    return features

def process_iteration_train(i,mod,hyperp,features,random_seeds_test,random_seeds_train, 
                    data,y,ID,test_size,feature_selection, 
                    scaler,metrics_names,cmatrix,CV_type,boot_train,boot_val,
                    held_out,path_to_save):    

    all_results = pd.DataFrame()
    all_scores = pd.DataFrame()
    selected_features = pd.DataFrame()

    n_seeds_train = len(random_seeds_train)
        
    selected_features_to_append = pd.DataFrame(columns=list(hyperp.columns) + ['random_seed_test'] + features)

    params = hyperp.loc[i,:].to_dict()

    for r,random_seed_test in enumerate(random_seeds_test):
        print(f'Random seed test {r+1}/{len(random_seeds_test)}')
        if held_out:
            ID_train, ID_test, _, _ = train_test_split(ID, y, test_size=test_size, random_state=random_seed_test, stratify=y)
            ID_train = ID_train.reset_index(drop=True)
            ID_test = ID_test.reset_index(drop=True)
            data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=test_size, random_state=random_seed_test, stratify=y)

            data_train = data_train.reset_index(drop=True)
            data_test = data_test.reset_index(drop=True)

            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

        else:
            ID_train = ID
            ID_test = pd.Series()
            data_train = data
            y_train = y
            data_test = pd.DataFrame()
            y_test = np.empty(0)

        n_features_to_select = int(np.floor(data_train.shape[1] / 2)) if feature_selection else None
        
        if 'random_test' in str(path_to_save):
            path_to_save = Path(path_to_save.parent, f'random_test_{random_seed_test}')
        else:
            path_to_save = Path(path_to_save, f'random_test_{random_seed_test}')

        path_to_save.mkdir(parents=True, exist_ok=True)

        results = train_models(model_type=mod,scaler=scaler, X_dev=data_train[features], y_dev=y_train, random_seeds=random_seeds_train,
                               hyperp=params, metrics_names=metrics_names, costs=cmatrix, priors=None,
                               CV_type=CV_type, IDs=ID_train, boot_train=boot_train, boot_val=boot_val, n_features_to_select=n_features_to_select)
        
        for random_seed in random_seeds_train:
            
            df_append = pd.DataFrame(columns=list(hyperp.columns) + ['random_seed_train','random_seed_test','bootstrap'] + metrics_names)

            df_append['random_seed_train'] = [random_seed]*np.max((1, boot_val))
            df_append['random_seed_test'] = [random_seed_test]*np.max((1, boot_val))
            df_append['bootstrap'] = np.arange(boot_val) if boot_val > 0 else np.nan

            for metric in metrics_names:
                df_append[metric] = results['metrics_val'][f'random_seed_{random_seed}'][metric]

            scores_append = pd.DataFrame.from_dict({'random_seed_train': [random_seed]*results['raw_logpost_val'][f'random_seed_{random_seed}'].shape[0],
                                                        'random_seed_test': [random_seed_test]*results['raw_logpost_val'][f'random_seed_{random_seed}'].shape[0],
                                                        'ID': results['IDs_val'][f'random_seed_{random_seed}'],
                                                        'raw_logpost': results['raw_logpost_val'][f'random_seed_{random_seed}'][:, 1],
                                                        'y_true': results['y_true_val'][f'random_seed_{random_seed}'],
                                                        'y_pred': results['y_pred_val'][f'random_seed_{random_seed}']})

            for param in params.keys():
                scores_append[param] = params[param]
                df_append[param] = params[param]

            if all_results.empty:
                all_results = df_append.copy()
            else:
                all_results = pd.concat((all_results, df_append), ignore_index=True, axis=0)

            if all_scores.empty:
                all_scores = scores_append.copy()
            else:
                all_scores = pd.concat((all_scores, scores_append), ignore_index=True, axis=0)

        selected_features_to_append = pd.DataFrame(columns=list(hyperp.columns) + ['random_seed_test'] + features,index=[0])
        selected_features_to_append['random_seed_test'] = random_seed_test

        for param in params.keys():
            selected_features_to_append[param] = params[param]
        for feature in features:
            selected_features_to_append[feature] = 1 if results['selected_features'][feature] > 7/10*n_seeds_train else 0

        if selected_features.empty:
            selected_features = selected_features_to_append.copy()
        else:
            selected_features = pd.concat((selected_features,selected_features_to_append),ignore_index=True,axis=0)
        
        ID_test.to_csv(Path(path_to_save, f'ID_test.csv'), index=False)
        pickle.dump(data_test, open(Path(path_to_save, f'data_test.pkl'), 'wb'))
        pickle.dump(y_test, open(Path(path_to_save, f'y_test.pkl'), 'wb'))
        pickle.dump(data_train, open(Path(path_to_save, f'data_train.pkl'), 'wb'))
        pickle.dump(y_train, open(Path(path_to_save, f'y_train.pkl'), 'wb'))

        conf_int = pd.DataFrame(columns=list(hyperp.columns) + [f'inf_{metric}' for metric in metrics_names] + [f'mean_{metric}' for metric in metrics_names] + [f'sup_{metric}' for metric in metrics_names],index=[0])

        for param in params.keys():
            conf_int[param] = params[param]
        
        for metric in metrics_names:
            conf_int[f'inf_{metric}'] = np.nanpercentile(all_results[metric],2.5)
            conf_int[f'mean_{metric}'] = np.nanmean(all_results[metric])
            conf_int[f'sup_{metric}'] = np.nanpercentile(all_results[metric],97.5)

    return all_results, all_scores, selected_features, conf_int

def get_feature_importance(model,X,y,scaler=MinMaxScaler(),CV=None,random_seed=0):
    
        X.reset_index(drop=True,inplace=True)
    
        X = X[[col for col in X.columns if not isinstance(X[col][0],str)]]
    
        if CV is None:
            train_val_iter = StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=random_seed).split(X,y)
        else:
            train_val_iter = CV.split(X,y)
        
        feature_importances = []
    
        for k,(train_index, val_index) in enumerate(train_val_iter):
        
            X_train = pd.DataFrame(index=train_index,columns=X.columns)
    
            X_val = pd.DataFrame(index=val_index,columns=X.columns)
    
            y_train = y[train_index].reset_index(drop=True)
    
            for feature in X.columns:
                sclr = scaler.fit(X.loc[train_index,feature].values.reshape(-1,1))
    
                X_train[feature] = sclr.transform(X.loc[train_index,feature].values.reshape(-1,1))
                X_val[feature] = sclr.transform(X.loc[val_index,feature].values.reshape(-1,1))        
    
            X_train.reset_index(drop=True,inplace=True)
            X_val.reset_index(drop=True,inplace=True)
            
            model.fit(X_train,y_train)   

            if hasattr(model,'feature_importances_'):
                feature_importances.append(model.feature_importances_)
            elif hasattr(model,'coef_'):
                feature_importances.append(np.abs(model.coef_[0]))
            else:
                raise ValueError('Model does not have feature_importances_ or coef_ attribute')
            
        feature_importances_df = pd.DataFrame(index=model.feature_names_in_,columns=['importance'])
        feature_importances_df['importance'] = np.mean(feature_importances,axis=0)
        feature_importances_df.sort_values(by='importance',ascending=False,inplace=True)

        return feature_importances_df

def get_youden_threshold(y_train,y_score_train):
    
    fpr, tpr, thresholds = roc_curve(y_train, y_score_train)
        
    idx = np.argmax(tpr - fpr)
            
    return thresholds[idx]

def hyperparam_tuning(model,hyperp_space,scoring,X,y,search_method,n_iter,CV=None):

    if search_method == 'bayes':
        search = BayesSearchCV(model,hyperp_space,n_iter=n_iter,scoring=scoring,n_jobs=-1,random_state=42,cv=CV).fit(X,y)
    elif search_method == 'random':
        search = RandomizedSearchCV(model,hyperp_space,n_iter=n_iter,scoring=scoring,n_jobs=-1,random_state=42,cv=CV).fit(X,y)

    best_model = search.best_estimator_
    best_params = search.best_params_

    return {'model': best_model, 'params': best_params}

def plot_roc_curve(tprs,fprs,aucs,legend,plot_std,color_chance='k',color_curve='b'):

    interp_tprs = []
    mean_fpr = np.linspace(0,1,100)

    fig, ax = plt.subplots()

    for i in np.arange(len(tprs)):
        interp_tpr = np.interp(mean_fpr,fprs[i],tprs[i])

        interp_tpr[0] = 0.0

        interp_tprs.append(interp_tpr)

    mean_tpr = np.mean(interp_tprs,axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr,mean_tpr)
    std_auc = np.nanstd(aucs)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=3, color=color_chance, label="Chance", alpha=0.8)

    ax.plot(
    mean_fpr,
    mean_tpr,
    color=color_curve,
    label=legend + '\n' + r"AUC = %0.2f $\pm$ %0.2f" % (mean_auc, std_auc),
    lw=3,
    alpha=0.8,
    )
    
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    std_tpr = np.std(interp_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    if plot_std:
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax.legend(loc="lower right")

    return ax, fig

def plot_roc_curve_from_scores(score_dict,legend,y_true_key='y_true',score_key='raw_logpost',iterate_over='random_seed',plot_std=True,color_chance='k',color_curve='b'):
    tprs = []
    fprs = []
    aucs = []
    
    for it in np.unique(score_dict[iterate_over]):
        index = np.where(score_dict[iterate_over] == it)[0]
        score = score_dict[score_key][index,:]
        y_true = score_dict[y_true_key][index]
        
        fpr, tpr, _ = roc_curve(y_true=y_true,y_score=score[:,1])
        auc_ = roc_auc_score(y_true=y_true,y_score=score[:,1])
                
        tprs.append(tpr)
        fprs.append(fpr)
        aucs.append(auc_)
        
    return plot_roc_curve(tprs,fprs,aucs,legend,plot_std,color_chance=color_chance,color_curve=color_curve)

def train_tune_calibrate_eval(model,scaler,X_dev,y_dev,X_test,y_test,random_seed,hyperp_space,n_iter,metrics_names,scoring='norm_cross_entropy',costs=None,priors=None,CV_out=None,CV_in=None,cal=True,cheat=False,feature_selection=False,IDs=None,ID_test=None,boot_train=False,boot_val=0,boot_test=0):
    """
    Trains a model, tunes hyperparameters, calibrates probabilities, and evaluates performance.

    Args:
        model: The machine learning model to be trained and evaluated.
        scaler: The scaler object used to normalize the input features.
        X_dev: The development set input features for training and validation.
        y_dev: The development set target variable for training and validation.
        X_test: The test set input features for evaluation.
        y_test: The test set target variable for evaluation.
        random_seed: The random seed for reproducibility.
        hyperp_space: The hyperparameter space for randomized search.
        n_iter: The number of iterations for randomized search.
        scoring: The metric used to select the best model. Defaults to 'norm_cross_entropy'.
        costs: The cost matrix used to calculate expected costs. Defaults to None, which sets it to a 2x2 cost matrix with zeros on the diagonal and ones elsewhere.
        priors: The prior probabilities of the target classes. Defaults to None, which uses the empirical class distribution.
        CV_out: The outer cross-validation object or iterator. Defaults to None, which sets it to StratifiedShuffleSplit with shuffling and the provided random seed.
        CV_in: The inner cross-validation object or iterator. Defaults to None, which sets it to StratifiedShuffleSplit with shuffling and the provided random seed.
        cal: Whether to calibrate the probabilities. Defaults to True.
        cheat: Whether to use true labels for calibration. Defaults to False.
        feature_selection: Whether to perform feature selection. Defaults to False.
        IDs: The sample IDs. Defaults to None.
        boot_train: Whether to bootstrap training samples. Defaults to False.
        boot_val: The number of bootstrap iterations for validation. Defaults to 0.

    Returns:
        results: A dictionary containing the evaluation results and other information.
    """

    if CV_out is None:
        CV_out = StratifiedShuffleSplit(shuffle=True,random_state=random_seed)
    
    if CV_in is None:
        CV_in = StratifiedShuffleSplit(shuffle=True,random_state=random_seed)
    
    if costs == None:
        costs = CostMatrix([[0,1],[1,0]])

    IDs_train = pd.DataFrame({'fold':[],'ID':[]})
    IDs_val = pd.DataFrame({'fold':[],'ID':[]})
    y_true = np.empty(0,dtype=int)

    metrics = dict([(metric,[]) for metric in metrics_names])
    
    #if all(isinstance(hyperp_space[hyperp],list) for hyperp in hyperp_space.keys()) and np.prod([len(hyperp_space[hyperp]) for hyperp in hyperp_space.keys()]) < n_iter:
    #    combinations = list(itertools.product(*hyperp_space.values()))
    #    n_iter = len(combinations)

    mean_metrics = dict([(metric,[]) for metric in metrics_names])

    for k,(train_out,val_idx) in enumerate(CV_out.split(X_dev,y_dev)):
        np.random.seed(0)
        boot_idx = np.random.choice(train_out,replace=True,size=len(train_out)) if boot_train else train_out
        IDs_train= pd.concat((IDs_train,pd.DataFrame({'fold':[k]*len(IDs[boot_idx]),'ID':IDs[boot_idx]})),axis=0)             
        IDs_val = pd.concat((IDs_val,pd.DataFrame({'fold':[k]*len(IDs[val_idx]),'ID':IDs[val_idx]})),axis=0)

        y_true = np.hstack([y_true,y_dev[val_idx]])

    models,raw_logpost,selected_features = generate_models(model_type=model,hyperp_space=hyperp_space,X_dev=X_dev,y_dev=y_dev,n_iter=n_iter,scaler=scaler,CV=CV_out,boot_train=boot_train,feature_selection=feature_selection)

    n_iter = len(models)

    cal_logpost = dict([(f'model_{i}',[]) for i in range(n_iter)])

    raw_logpost_bt = dict([(f'model_{i}',[]) for i in range(n_iter)])
    cal_logpost_bt = dict([(f'model_{i}',[]) for i in range(n_iter)])
    y_true_bt = dict([(f'model_{i}',[]) for i in range(n_iter)])
    y_pred_bt = dict([(f'model_{i}',[]) for i in range(n_iter)])
    IDs_val_bt = dict([(f'model_{i}',[]) for i in range(n_iter)])

    all_metrics = dict([(f'model_{i}',dict([(metric,[]) for metric in metrics_names])) for i in range(n_iter)])

    for i in range(n_iter):   
        features = X_dev.columns[selected_features[f'model_{i}'] > CV_out.get_n_splits(X_dev,y_dev)/2]
        if cal:
            print(f'Calibrating model {i}/{n_iter}')
            if cheat:
                cal_logpost[f'model_{i}'] = cheat_calibration(scores=raw_logpost[f'model_{i}'],y=y_true,loss=AffineCalLogLoss,priors=priors)
            else:
                cal_logpost[f'model_{i}'] = cv_calibration(mod=models[f'model_{i}'],X_cal=X_dev,y_cal=y_dev,features=features,CV_out=CV_out,CV_in=CV_in,loss=AffineCalLogLoss,priors=priors)
        else:
            cal_logpost[f'model_{i}'] = raw_logpost[f'model_{i}']

        y_pred = bayes_decisions(scores=cal_logpost[f'model_{i}'],costs=costs,priors=priors,score_type='log_posteriors')[0]

        if boot_val > 0:
            all_metrics[f'model_{i}'], index_bt = bootstrap_scores(y_scores=cal_logpost[f'model_{i}'],y_true=y_true,y_pred=y_pred, metrics_names=metrics_names,costs=costs,priors=priors,n_bootstrap=boot_val)
            raw_logpost_bt[f'model_{i}'] = raw_logpost[f'model_{i}'][index_bt,:]
            cal_logpost_bt[f'model_{i}'] = cal_logpost[f'model_{i}'][index_bt,:]
            y_true_bt[f'model_{i}'] = y_true[index_bt]
            y_pred_bt[f'model_{i}'] = bayes_decisions(scores=cal_logpost[f'model_{i}'][index_bt,:],costs=costs,priors=priors,score_type='log_posteriors')[0]
            IDs_val_bt[f'model_{i}']= IDs_val.loc[index_bt,'ID']
        else:
            all_metrics[f'model_{i}'] = get_metrics(y_scores=cal_logpost[f'model_{i}'],y_true=y_true,y_pred=y_pred,metrics_names=metrics_names,cmatrix=costs,priors=priors)
            raw_logpost_bt[f'model_{i}'] = raw_logpost[f'model_{i}']
            cal_logpost_bt[f'model_{i}'] = cal_logpost[f'model_{i}']
            y_true_bt[f'model_{i}'] = y_true
            y_pred_bt[f'model_{i}'] = y_pred
            IDs_val_bt[f'model_{i}'] = IDs_val['ID'].values
            
        for metric in metrics_names:
            mean_metrics[metric].append(np.nanmean(all_metrics[f'model_{i}'][metric]))
    
    if 'norm' in scoring:
        best_index = np.argmin(mean_metrics[scoring])
    else:

        best_index = np.argmax(mean_metrics[scoring])

    best_model = models[f'model_{best_index}']
    features = X_dev.columns[selected_features[f'model_{best_index}'] > CV_out.get_n_splits(X_dev,y_dev)/2]

    best_model.train(X_dev[features],y_dev)
    if cal:
        try:
           _ , cal_params = calibrate(torch.as_tensor(best_model.eval(X_dev[features]),dtype=torch.float32),torch.as_tensor(y_dev.values,dtype=torch.int64),torch.as_tensor(raw_logpost[f'model_{best_index}'],dtype=torch.float32),AffineCalLogLoss,priors=priors)
        except:
            cal_params = None
    else:
        cal_params = None

    results = dict()    
   
    results['model'] = best_model.model
    results['raw_logpost_val'] = raw_logpost_bt[f'model_{best_index}']
    results['cal_logpost_val'] = cal_logpost_bt[f'model_{best_index}']
    results['y_true_val'] = y_true_bt[f'model_{best_index}']
    results['y_pred_val'] = bayes_decisions(scores=cal_logpost_bt[f'model_{best_index}'],costs=costs,priors=priors,score_type='log_posteriors')[0]
    results['IDs_val'] = pd.DataFrame({'ID':IDs_val_bt[f'model_{best_index}']})
    results['metrics_val'] = all_metrics[f'model_{best_index}']

    results['selected_features'] = {'feature':X_dev.columns,'score':selected_features[f'model_{best_index}']}

    results['cal_params'] = cal_params

    if not X_test.empty:

        raw_logpost_test = best_model.eval(X_test[features])
        if cal:
            try:
                cal_logpost_test_, _ = calibrate(torch.as_tensor(best_model.eval(X_dev[features]),dtype=torch.float32),torch.as_tensor(y_dev.values,dtype=torch.int64),torch.as_tensor(raw_logpost_test,dtype=torch.float32),AffineCalLogLoss,priors=priors)
                cal_logpost_test = cal_logpost_test_.detach().numpy()
            except:
                cal_logpost_test = raw_logpost_test
        else:
            cal_logpost_test = raw_logpost_test

        y_pred_test = bayes_decisions(scores=cal_logpost_test,costs=costs,priors=priors,score_type='log_posteriors')[0]

        if boot_test > 0:
            results['metrics_test'],index_bt = bootstrap_scores(y_scores=cal_logpost_test,y_true=y_test.values,y_pred=y_pred_test,metrics_names=metrics_names,costs=costs,priors=priors,n_bootstrap=boot_test)
        
        else:
            results['metrics_test'] = get_metrics(y_scores=cal_logpost_test,y_true=y_test.values,y_pred=y_pred_test,metrics_names=metrics_names,cmatrix=costs,priors=priors)
            index_bt = np.arange(len(y_test))
        
        results['raw_logpost_test'] = raw_logpost_test[index_bt,:]
        results['cal_logpost_test'] = cal_logpost_test[index_bt,:]

        results['y_pred_test'] = y_pred_test[index_bt]
        results['y_true_test'] = y_test[index_bt]

        results['IDs_test'] = pd.DataFrame({'ID':ID_test[index_bt]})
        
    return results

def train_tune_calibrate(model,scaler,X_dev,y_dev,random_seed,hyperp_space,n_iter,metrics_names,scoring='norm_cross_entropy',costs=None,priors=None,CV_out=None,CV_in=None,cal=True,cheat=False,feature_selection=False,IDs=None,boot_train=False,boot_val=0):
    """
    Trains a model, tunes hyperparameters, calibrates probabilities, and evaluates performance.

    Args:
        model: The machine learning model to be trained and evaluated.
        scaler: The scaler object used to normalize the input features.
        X_dev: The development set input features for training and validation.
        y_dev: The development set target variable for training and validation.
        X_test: The test set input features for evaluation.
        y_test: The test set target variable for evaluation.
        random_seed: The random seed for reproducibility.
        hyperp_space: The hyperparameter space for randomized search.
        n_iter: The number of iterations for randomized search.
        scoring: The metric used to select the best model. Defaults to 'norm_cross_entropy'.
        costs: The cost matrix used to calculate expected costs. Defaults to None, which sets it to a 2x2 cost matrix with zeros on the diagonal and ones elsewhere.
        priors: The prior probabilities of the target classes. Defaults to None, which uses the empirical class distribution.
        CV_out: The outer cross-validation object or iterator. Defaults to None, which sets it to StratifiedShuffleSplit with shuffling and the provided random seed.
        cheat: Whether to use true labels for calibration. Defaults to False.
        feature_selection: Whether to perform feature selection. Defaults to False.
        IDs: The sample IDs. Defaults to None.
        boot_train: Whether to bootstrap training samples. Defaults to False.
        boot_val: The number of bootstrap iterations for validation. Defaults to 0.

    Returns:
        results: A dictionary containing the evaluation results and other information.
    """

    if CV_out is None:
        CV_out = StratifiedShuffleSplit(shuffle=True,random_state=random_seed)
    
    if CV_in is None:
        CV_in = StratifiedShuffleSplit(shuffle=True,random_state=random_seed)
    
    if costs == None:
        costs = CostMatrix([[0,1],[1,0]])

    IDs_train = pd.DataFrame({'fold':[],'ID':[]})
    IDs_val = pd.DataFrame({'fold':[],'ID':[]})
    y_true = np.empty(0,dtype=int)

    metrics = dict([(metric,[]) for metric in metrics_names])
    
    #if all(isinstance(hyperp_space[hyperp],list) for hyperp in hyperp_space.keys()) and np.prod([len(hyperp_space[hyperp]) for hyperp in hyperp_space.keys()]) < n_iter:
    #    combinations = list(itertools.product(*hyperp_space.values()))
    #    n_iter = len(combinations)

    mean_metrics = dict([(metric,[]) for metric in metrics_names])

    for k,(train_out,val_idx) in enumerate(CV_out.split(X_dev,y_dev)):
        
        boot_idx = np.random.choice(train_out,replace=True,size=len(train_out)) if boot_train else train_out
        IDs_train= pd.concat((IDs_train,pd.DataFrame({'fold':[k]*len(IDs[boot_idx]),'ID':IDs[boot_idx]})),axis=0)             
        IDs_val = pd.concat((IDs_val,pd.DataFrame({'fold':[k]*len(IDs[val_idx]),'ID':IDs[val_idx]})),axis=0)

        y_true = np.hstack([y_true,y_dev[val_idx]])

    models,raw_logpost,selected_features = generate_models(model_type=model,hyperp_space=hyperp_space,X_dev=X_dev,y_dev=y_dev,n_iter=n_iter,scaler=scaler,CV=CV_out,boot_train=boot_train,feature_selection=feature_selection)

    n_iter = len(models)

    cal_logpost = dict([(f'model_{i}',[]) for i in range(n_iter)])

    raw_logpost_bt = dict([(f'model_{i}',[]) for i in range(n_iter)])
    cal_logpost_bt = dict([(f'model_{i}',[]) for i in range(n_iter)])
    y_true_bt = dict([(f'model_{i}',[]) for i in range(n_iter)])
    y_pred_bt = dict([(f'model_{i}',[]) for i in range(n_iter)])
    IDs_val_bt = dict([(f'model_{i}',[]) for i in range(n_iter)])

    all_metrics = dict([(f'model_{i}',dict([(metric,[]) for metric in metrics_names])) for i in range(n_iter)])

    for i in range(n_iter):   
        features = X_dev.columns[selected_features[f'model_{i}'] > CV_out.get_n_splits(X_dev,y_dev)/2]
        if cal:
            print(f'Calibrating model {i}/{n_iter}')
            if cheat:
                cal_logpost[f'model_{i}'] = cheat_calibration(scores=raw_logpost[f'model_{i}'],y=y_true,loss=AffineCalLogLoss,priors=priors)
            else:
                cal_logpost[f'model_{i}'] = cv_calibration(mod=models[f'model_{i}'],X_cal=X_dev,y_cal=y_dev,features=features,CV_out=CV_out,CV_in=CV_in,loss=AffineCalLogLoss,priors=priors)
        else:
            cal_logpost[f'model_{i}'] = raw_logpost[f'model_{i}']

        y_pred = bayes_decisions(scores=cal_logpost[f'model_{i}'],costs=costs,priors=priors,score_type='log_posteriors')[0]

        if boot_val > 0:
            all_metrics[f'model_{i}'], y_scores_bt_, y_true_bt_, y_pred_bt_, IDs_bt = bootstrap_scores(y_scores=cal_logpost[f'model_{i}'],y_true=y_true,y_pred=y_pred, metrics_names=metrics_names,IDs=IDs_val['ID'],costs=costs,priors=priors,n_bootstrap=boot_val)
            cal_logpost_bt[f'model_{i}'] = y_scores_bt_
            y_true_bt[f'model_{i}'] = y_true_bt_
            y_pred_bt[f'model_{i}'] = y_pred_bt_
            IDs_val_bt[f'model_{i}']= IDs_bt
        else:
            all_metrics[f'model_{i}'] = get_metrics(y_scores=cal_logpost[f'model_{i}'],y_true=y_true,y_pred=y_pred,metrics_names=metrics_names,cmatrix=costs,priors=priors)
            raw_logpost_bt[f'model_{i}'] = raw_logpost[f'model_{i}']
            cal_logpost_bt[f'model_{i}'] = cal_logpost[f'model_{i}']
            y_true_bt[f'model_{i}'] = y_true
            y_pred_bt[f'model_{i}'] = y_pred
            IDs_val_bt[f'model_{i}'] = IDs_val['ID'].values

    results = dict()    
   
    results['model'] = models
    results['raw_logpost_val'] = raw_logpost_bt
    results['cal_logpost_val'] = cal_logpost_bt
    results['y_true_val'] = y_true_bt
    results['y_pred_val'] ={f'model_{i}': bayes_decisions(scores=cal_logpost_bt[f'model_{i}'],costs=costs,priors=priors,score_type='log_posteriors')[0] for i in range(n_iter)}
    results['IDs_val'] = IDs_val_bt
    results['metrics_val'] = all_metrics

    results['selected_features'] = {'feature':X_dev.columns,'score':selected_features}

    return results

def generate_models(model_type,hyperp_space,X_dev,y_dev,n_iter,scaler,CV,boot_train=False,feature_selection=False):
    #Perform randomized search for hyperparameter tuning
    #Get a random combination of hyperparameters from hyperparameter space
    
    combinations = []

    if all(isinstance(hyperp_space[hyperp],list) for hyperp in hyperp_space.keys()) and np.prod([len(hyperp_space[hyperp]) for hyperp in hyperp_space.keys()]) < n_iter:
        combinations = list(itertools.product(*hyperp_space.values()))
        hyperp_space = dict([(key,values) for key,values in zip(hyperp_space.keys(),combinations)])
        n_iter = len(combinations)
    
    models = dict([(f'model_{i}',[]) for i in range(n_iter)])
    raw_logpost = dict([(f'model_{i}',[]) for i in range(n_iter)])
    cal_logpost = dict([(f'model_{i}',[]) for i in range(n_iter)])
    selected_features = dict([(f'model_{i}',np.zeros(X_dev.shape[1])) for i in range(n_iter)])
    
    for i in tqdm(range(n_iter)):
        if len(combinations) == 0:
            np.random.seed(0)
            random_combination = {param: values.rvs() if not isinstance(values,list) else random.choice(values) for param, values in hyperp_space.items() }
        else:
            random_combination = {param: values for param, values in zip(hyperp_space.keys(),combinations[i])}

        if hasattr(model_type(),'random_state'):
            random_combination['random_state'] = 42

        if hasattr(model_type(),'max_iter'):
            random_combination['max_iter'] = 10000
            
        if hasattr(model_type(),'probability'):
            random_combination['probability'] = True

        models[f'model_{i}'] = Model(model=model_type(**random_combination),scaler=scaler)
        raw_logpost[f'model_{i}'] = np.empty((0,2)) 
        cal_logpost[f'model_{i}'] = np.empty((0,2))

        mod = deepcopy(models[f'model_{i}'])
        for train_out,val_idx in CV.split(X_dev,y_dev):
            
            boot_idx = np.random.choice(train_out,replace=True,size=len(train_out)) if boot_train else train_out
            
            X_train = X_dev.iloc[boot_idx]
            y_train = y_dev.iloc[boot_idx]

            X_val = X_dev.iloc[val_idx]

            if feature_selection:
                if type(mod.model) == SVC:
                    rfe_model = mod.model
                    rfe_model.kernel = 'linear'
                else:
                    rfe_model = mod.model
                rfe = RFE(estimator=rfe_model,n_features_to_select=None,step=1)
                rfe.fit(X_train,y_train)
                features = X_train.columns[rfe.support_]
                selected_features[f'model_{i}'] += rfe.support_

            else:
                features = X_train.columns
                selected_features[f'model_{i}'] += np.ones(X_dev.shape[1])

            mod.train(X=X_train[features],y=y_train)  

            logpost_val = mod.eval(X=X_val[features])

            raw_logpost[f'model_{i}'] = np.vstack([raw_logpost[f'model_{i}'],logpost_val])

    return (models,raw_logpost,selected_features)

def cheat_calibration(scores,y,loss=AffineCalLogLoss,priors=None):
    cal_logpost_val, cal_params = calibrate(torch.as_tensor(scores,dtype=torch.float32),torch.as_tensor(y.values,dtype=torch.int64),torch.as_tensor(scores,dtype=torch.float32),loss,priors=priors)
    return cal_logpost_val, cal_params 

def cv_calibration(mod,X_cal,y_cal,features,CV_out,CV_in,loss=AffineCalLogLoss,priors=None):
    logpost_train = np.empty((0,2))
    cal_logpost = np.empty((0,2))
    y_true = np.empty(0)
    
    for train,val in CV_out.split(X_cal,y_cal):
        X_train = X_cal.iloc[train]
        y_train = y_cal.iloc[train]
        
        X_val = X_cal.iloc[val]
        for train_in,val_in in CV_in.split(X_train,y_train):
            X_train_cal = X_train.iloc[train_in]
            y_train_cal = y_train.iloc[train_in]
            X_test_cal = X_train.iloc[val_in]
            y_true = np.hstack([y_true,y_train.iloc[val_in]])

            mod.train(X=X_train_cal[features],y=y_train_cal)  
            
            logpost = mod.eval(X=X_test_cal[features])

            logpost_train = np.vstack([logpost_train,logpost])

        raw_logpost = mod.eval(X=X_val)
        try:
            cal_logpost_val = calibrate(torch.as_tensor(logpost_train,dtype=torch.float32),torch.as_tensor(y_true,dtype=torch.int64),torch.as_tensor(raw_logpost,dtype=torch.float32),loss,priors=priors)[0] #Ver si le puedo pasar las priors
            cal_logpost_val = cal_logpost_val.detach().numpy()
        except:
            cal_logpost_val = raw_logpost
            
        cal_logpost = np.vstack([cal_logpost,cal_logpost_val])

    return cal_logpost
