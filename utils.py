import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_absolute_error, mean_squared_error,median_absolute_error
from sklearn.feature_selection import RFECV
import torch,itertools,json
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNNC 
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.svm import SVR 

from expected_cost.ec import *
from expected_cost.utils import *
from psrcal.losses import LogLoss

from joblib import Parallel, delayed

import tqdm,pdb

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample 

from skopt import BayesSearchCV

class Model():
    def __init__(self,model,scaler=None,imputer=None,calibrator=None):
        self.model = model
        self.scaler = scaler() if scaler is not None else None
        self.imputer = imputer() if imputer is not None else None
        self.calibrator = calibrator() if calibrator is not None else None

    def train(self,X,y):   
        features = X.columns
        
        X_t = pd.DataFrame(columns=features,data=self.scaler.fit_transform(X[features].values)) if self.scaler is not None else X
        X_t = pd.DataFrame(columns=features,data=self.imputer.fit_transform(X_t[features].values)) if self.imputer is not None else X_t

        params = self.model.get_params()
        if 'n_estimators' in params.keys():
            params['n_estimators'] = int(params['n_estimators'])
        if 'n_neighbors' in params.keys():
            params['n_neighbors'] = int(params['n_neighbors'])
        if 'max_depth' in params.keys():
            params['max_depth'] = int(params['max_depth'])

        self.model.set_params(**params)
        if hasattr(self.model,'precompute'):
            self.model.precompute = True
        self.model.fit(X_t,y)
        
    def eval(self,X,problem_type='clf'):
        
        features = X.columns
        X_t = pd.DataFrame(columns=features,data=self.scaler.transform(X[features].values)) if self.scaler is not None else X
        X_t = pd.DataFrame(columns=features,data=self.imputer.transform(X_t[features].values)) if self.imputer is not None else X_t

        if problem_type == 'clf':
            if hasattr(self.model,'predict_log_proba'):
                score = self.model.predict_log_proba(X_t)
            else:
                prob = self.model.predict_proba(X_t)
                prob = np.clip(prob,1e-6,1-1e-6)
                score = np.log(prob)
        else:
            score = self.model.predict(X_t)    

        return score

def get_metrics_clf(y_scores,y_true,metrics_names,cmatrix=None,priors=None,threshold=None):
    """
    Calculates evaluation metrics for the predicted scores and true labels.

    Args:
        y_scores: The predicted scores.
        y_true: The true labels.
        metrics_names: The names of the evaluation metrics.
        cmatrix: The cost matrix used to calculate expected costs. Defaults to None.
        priors: The prior probabilities of the target classes. Defaults to None.

    Returns:
        metrics: A dictionary containing the evaluation metrics.

    Note:
        The function calculates the evaluation metrics for the predicted scores and true labels.
    """
    if cmatrix is None:
        cmatrix = CostMatrix([[0,1],[1,0]])
    metrics = dict([(metric,[]) for metric in metrics_names])

    y_pred = bayes_decisions(scores=y_scores,costs=cmatrix,priors=priors,score_type='log_posteriors')[0] if threshold is None else np.array(np.exp(y_scores[:,1]) > threshold,dtype=int)

    y_scores = np.clip(y_scores,-1e6,1e6)

    for m in metrics_names:
        if m == 'norm_cross_entropy':
            try:
                metrics[m] = LogLoss(log_probs=torch.tensor(y_scores),labels=torch.tensor(np.array(y_true),dtype=torch.int),priors=torch.tensor(priors)).detach().numpy() if priors is not None else LogLoss(log_probs=torch.tensor(y_scores),labels=torch.tensor(np.array(y_true),dtype=torch.int)).detach().numpy()
            except: 
                metrics[m] = np.nan
        elif m == 'norm_expected_cost':
            try:
                metrics[m] = average_cost(targets=np.array(y_true,dtype=int),decisions=np.array(y_pred,dtype=int),costs=cmatrix,priors=priors,adjusted=True)
            except:
                metrics[m] = np.nan
        elif m == 'roc_auc':
            try:
                metrics[m] = roc_auc_score(y_true=y_true,y_score=y_scores[:,1])
            except:
                metrics[m] = np.nan
        else:
            metrics[m] = eval(f'{m}_score')(y_true=np.array(y_true,dtype=int),y_pred=y_pred)

    return metrics,y_pred

def get_metrics_reg(y_scores,y_true,metrics_names):
    """
    Calculates evaluation metrics for the predicted scores and true labels.

    Args:
        y_scores: The predicted scores.
        y_true: The true labels.
        metrics_names: The names of the evaluation metrics.

    Returns:
        metrics: A dictionary containing the evaluation metrics.

    Note:
        The function calculates the evaluation metrics for the predicted scores and true labels.
    """
    metrics = dict((metric,np.nan) for metric in metrics_names)	
    for metric in metrics_names:
        try: 
            metrics[metric] = eval(metric)(y_true=y_true,y_pred=y_scores)
        except:
            metrics[metric] = np.nan

    return metrics

def conf_int_95(data):
    mean = np.nanmean(data)
    inf = np.nanpercentile(data,2.5)
    sup = np.nanpercentile(data,97.5) 
    return mean, inf, sup
            
def CV(i,model,X,y,all_features,threshold,iterator,random_seeds_train,metrics,IDs,n_boot=0,cmatrix=None,priors=None,problem_type='clf'):
    
    print(f'Modelo: {i}')

    if hasattr(model.model,'random_state'):
        model.model.random_state = 42

    if cmatrix is None:
        cmatrix = CostMatrix([[0,1],[1,0]])

    model_params = model.model.get_params()
    features = dict((feature,0) for feature in all_features)
    
    for feature in X.columns:
        features[feature] = 1
    
    model_params.update(features)
    model_params.update({'threshold':threshold})
    
    model_params = pd.DataFrame(model_params,index=[0])

    metrics_bootstrap = dict([(metric,np.empty(np.max((1,n_boot)))) for metric in metrics])

    outputs_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0],2,len(random_seeds_train))) if problem_type == 'clf' else np.empty((np.max((1,n_boot)),X.shape[0],len(random_seeds_train)))

    y_true_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0],len(random_seeds_train)))
    y_pred_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0],len(random_seeds_train)))
    IDs_dev_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0],len(random_seeds_train)),dtype=object)
    X_dev_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0],X.shape[1],len(random_seeds_train)))

    metrics_oob = dict([(metric,np.empty(np.max((1,n_boot)))) for metric in metrics])

    for r_train,random_seed in enumerate(random_seeds_train):
        iterator.random_state = random_seed
        X_dev = np.empty((X.shape[0],X.shape[1]))
        y_dev = np.empty(X.shape[0])
        IDs_dev = np.empty(X.shape[0],dtype=object)
        
        outputs_dev = np.empty((X.shape[0],2)) if problem_type == 'clf' else np.empty((X.shape[0]))

        for train_index, test_index in iterator.split(X,y):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.train(X_train, y_train)
            
            X_dev[test_index,:] = X_test
            y_dev[test_index] = y_test
            IDs_dev[test_index] = IDs[test_index]
            
            if problem_type == 'clf':
                outputs_dev[test_index,:] = model.eval(X_test,problem_type) 
            else:
                outputs_dev[test_index] = model.eval(X_test,problem_type)

        for b in range(np.max((1,n_boot))):
            boot_index = resample(np.arange(X.shape[0]),n_samples=X.shape[0],replace=True,random_state=b) if n_boot > 0 else np.arange(X.shape[0])
            y_true_bootstrap[b,:,r_train] = y_dev[boot_index]
            IDs_dev_bootstrap[b,:,r_train] = IDs_dev[boot_index]
            X_dev_bootstrap[b,:,:,r_train] = X.iloc[boot_index,:].reset_index(drop=True)
            if problem_type == 'clf':
                outputs_bootstrap[b,:,:,r_train] = outputs_dev[boot_index,:] 
                metrics_,y_pred = get_metrics_clf(outputs_dev[boot_index,:],y_dev[boot_index],metrics,cmatrix,priors,threshold) 
            else:
                outputs_bootstrap[b,:,r_train] = outputs_dev[boot_index]
                metrics_ = get_metrics_reg(outputs_dev[boot_index],y_dev[boot_index],metrics)
                y_pred = outputs_dev[boot_index]

            y_pred_bootstrap[b,:,r_train] = y_pred
            
            for metric in metrics:
                metrics_bootstrap[metric][b] = metrics_[metric]
            
            if n_boot == 0:
                break
            
            oob_index = np.setdiff1d(np.arange(outputs_dev.shape[0]),boot_index)
            if problem_type == 'clf':
                metrics_,y_pred = get_metrics_clf(outputs_dev[oob_index,:],y_dev[oob_index],metrics,cmatrix,priors)
            else:
                metrics_ = get_metrics_reg(outputs_dev[oob_index],y_dev[oob_index],metrics)
                y_pred = outputs_dev[oob_index]

            for metric in metrics:
                metrics_oob[metric][b] = metrics_[metric]

    return model_params,metrics_bootstrap,outputs_bootstrap,y_true_bootstrap,y_pred_bootstrap,IDs_dev_bootstrap,metrics_oob

def CVT(model,scaler,imputer,X,y,iterator,random_seeds_train,hyperp,feature_sets,metrics,IDs,n_boot=0,cmatrix=None,priors=None,parallel=True,problem_type='clf',thresholds=[None]):
    
    features = X.columns
    
    if hasattr(model(),'random_state') and model != SVR:
        hyperp['random_state'] = 42

    all_models = pd.DataFrame(columns=list(hyperp.columns) + list(features))

    all_metrics_bootstrap = dict([(metric,np.empty((hyperp.shape[0]*len(feature_sets),np.max((1,n_boot))))) for metric in metrics])
    all_metrics_oob = dict([(metric,np.empty((hyperp.shape[0]*len(feature_sets),np.max((1,n_boot))))) for metric in metrics])

    all_outputs_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0],2,hyperp.shape[0]*len(feature_sets),len(random_seeds_train))) if problem_type == 'clf' else np.empty((np.max((1,n_boot)),X.shape[0],hyperp.shape[0]*len(feature_sets),len(random_seeds_train)))
    
    y_true_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0],len(random_seeds_train)))

    all_y_pred_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0],hyperp.shape[0]*len(feature_sets),len(random_seeds_train)))

    IDs_dev_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0],len(random_seeds_train)))

    if parallel == True:
        results = Parallel(n_jobs=-1)(delayed(CV)(i,Model(model(**hyperp.iloc[c,:]),scaler,imputer),X[feature_set],y,X.columns,threshold,iterator,random_seeds_train,metrics,IDs,json_log_file,n_boot,cmatrix,priors,problem_type) for i,(c,feature_set,threshold) in enumerate(itertools.product(range(hyperp.shape[0]),feature_sets,thresholds)))
        
        all_models = pd.concat([result[0] for result in results],ignore_index=True,axis=0)
        
        for metric in metrics:
            all_metrics_bootstrap[metric] = np.concatenate(([result[1][metric].reshape(1,all_metrics_bootstrap[metric].shape[-1]) for result in results]),axis=0)
            all_metrics_oob[metric] = np.concatenate(([result[6][metric].reshape(1,all_metrics_oob[metric].shape[-1]) for result in results]),axis=0)

        all_outputs_bootstrap = np.concatenate(([result[2].reshape(all_outputs_bootstrap.shape[0],all_outputs_bootstrap.shape[1],2,1,all_outputs_bootstrap.shape[-1]) for result in results]),axis=2) if problem_type == 'clf' else np.concatenate(([result[2].reshape(all_outputs_bootstrap.shape[0],all_outputs_bootstrap.shape[1],1,all_outputs_bootstrap.shape[-1]) for result in results]),axis=2)
        y_true_bootstrap = results[0][3]
        all_y_pred_bootstrap = np.concatenate(([result[4].reshape(all_y_pred_bootstrap.shape[0],all_y_pred_bootstrap.shape[1],1,all_y_pred_bootstrap.shape[-1]) for result in results]),axis=1)
        IDs_dev_bootstrap = results[0][5]
    else:
        for c,threshold in itertools.product(range(hyperp.shape[0]),thresholds):
            params = hyperp.iloc[c,:].to_dict()
            for f,feature_set in enumerate(feature_sets): 
                all_models.loc[c*len(feature_sets)+f,params.keys()] = hyperp.iloc[c,:].values
                all_models.loc[c*len(feature_sets)+f,features] = [1 if feature in feature_set else 0 for feature in features]
                _,metrics_bootstrap_c,outputs_bootstrap_c, y_true_bootstrap, y_pred_bootstrap,IDs_dev_bootstrap,metrics_oob_c = CV(c*len(feature_sets)*len(thresholds)+f,Model(model(**params),scaler,imputer),X[feature_set],y,X.columns,threshold,iterator,random_seeds_train,metrics,IDs,json_log_file,n_boot,cmatrix,priors,problem_type)
                
                for metric in metrics:
                    all_metrics_bootstrap[metric][c*len(feature_sets) + f,:] = metrics_bootstrap_c[metric]
                    all_metrics_oob[metric][c*len(feature_sets) + f,:] = metrics_oob_c[metric]
                
                if problem_type == 'clf':
                    all_outputs_bootstrap[:,:,:,c*len(feature_sets) + f,:] = outputs_bootstrap_c
                else:
                    all_outputs_bootstrap[:,:,c*len(feature_sets) + f,:] = outputs_bootstrap_c
                
                all_y_pred_bootstrap[:,:,c*len(feature_sets) + f,:] = y_pred_bootstrap
        
    return all_models,all_outputs_bootstrap,all_y_pred_bootstrap,all_metrics_bootstrap,y_true_bootstrap,IDs_dev_bootstrap,all_metrics_oob

def css(metrics,scoring='roc_auc',problem_type='clf'):
    inf_conf_int = np.empty(metrics[scoring].shape[0])
    sup_conf_int = np.empty(metrics[scoring].shape[0])

    for model in range(metrics[scoring].shape[0]):
        _, inf_conf_int[model], sup_conf_int[model] = conf_int_95(metrics[scoring][model,:])
        
    if problem_type == 'clf':
        if 'norm' not in scoring:
            best = np.argmax(inf_conf_int)
        else:
            best = np.argmin(sup_conf_int)
    else:
        if 'error' in scoring:
            best = np.argmin(sup_conf_int)
        else:
            best = np.argmax(inf_conf_int)
            
    return best

def select_best_models(metrics,scoring='roc_auc',problem_type='clf'):

    best = css(metrics,scoring,problem_type)
    return best

def BBCCV(model,scaler,imputer,X,y,iterator,random_seeds_train,hyperp,feature_sets,thresholds,metrics,IDs,json_log_file,n_boot=1000,cmatrix=None,priors=None,parallel=True,scoring='roc_auc',problem_type='clf'):
    
    all_models,all_outputs_bootstrap,all_y_pred_bootstrap,all_metrics_bootstrap,y_true_dev_bootstrap,IDs_dev_bootstrap,all_metrics_oob = CVT(model,scaler,imputer,X,y,iterator,random_seeds_train,hyperp,feature_sets,thresholds,metrics,IDs,json_log_file,n_boot,cmatrix,priors,parallel,problem_type)
    best_model = select_best_models(all_metrics_bootstrap,scoring,problem_type)
    
    return all_models,all_outputs_bootstrap,all_y_pred_bootstrap,all_metrics_bootstrap,y_true_dev_bootstrap,IDs_dev_bootstrap,all_metrics_oob,best_model

def test_model(model, X_dev, y_dev, X_test, y_test, metrics, IDs_test,
               n_boot_train=0, n_boot_test=0, cmatrix=None, priors=None, problem_type='clf'):
    if not isinstance(X_dev, pd.DataFrame):
        X_dev = pd.DataFrame(X_dev)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    n_boot_train = max(1, n_boot_train)
    n_boot_test = max(1, n_boot_test)
    n_boot = n_boot_train * n_boot_test

    outputs_bootstrap = np.empty((n_boot, X_test.shape[0], 2)) if problem_type == 'clf' else np.empty((n_boot, X_test.shape[0]))
    y_true_bootstrap = np.empty((n_boot, X_test.shape[0]))
    y_pred_bootstrap = np.empty((n_boot, X_test.shape[0]))
    IDs_test_bootstrap = np.empty((n_boot, X_test.shape[0]), dtype=object)
    metrics_test_bootstrap = {metric: np.empty(n_boot) for metric in metrics}

    idx = 0
    for b_train in range(n_boot_train):
        boot_index_train = resample(np.arange(X_dev.shape[0]), n_samples=X_dev.shape[0], replace=True, random_state=b_train) if n_boot_train > 1 else np.arange(X_dev.shape[0])

        model.train(X_dev.iloc[boot_index_train], y_dev[boot_index_train])

        for b_test in range(n_boot_test):
            boot_index = resample(np.arange(X_test.shape[0]), n_samples=X_test.shape[0], replace=True, random_state=b_train * n_boot_train + b_test) if n_boot_test > 1 else np.arange(X_test.shape[0])

            y_true_bootstrap[idx, :] = y_test[boot_index]
            IDs_test_bootstrap[idx, :] = IDs_test[boot_index]
            outputs = model.eval(X_test.iloc[boot_index, :], problem_type)

            if problem_type == 'clf':
                metrics_test, y_pred = get_metrics_clf(outputs, y_test[boot_index], metrics, cmatrix, priors)
                outputs_bootstrap[idx, :, :] = outputs
                y_pred_bootstrap[idx, :] = y_pred
            else:
                metrics_test = get_metrics_reg(outputs, y_test[boot_index], metrics)
                outputs_bootstrap[idx, :] = outputs

            for metric in metrics:
                metrics_test_bootstrap[metric][idx] = metrics_test[metric]

            idx += 1

    return metrics_test_bootstrap, outputs_bootstrap, y_true_bootstrap, y_pred_bootstrap, IDs_test_bootstrap

def nestedCVT_bayes(model,scaler,imputer,X,y,n_iter,iterator_outer,random_seeds_outer,hyperp,metrics,IDs,n_boot=0,cmatrix=None,priors=None,scoring='roc_auc',problem_type='clf'):
    
    features = X.columns
    iterator_inner = type(iterator_outer)(n_splits=iterator_outer.get_n_splits(),shuffle=True,random_state=42)

    all_models = pd.DataFrame(columns=['random_seed','fold'] + list(hyperp.keys()) + list(features))
    best_models = pd.DataFrame(columns=['random_seed','fold'] + list(hyperp.keys()) + list(features))

    all_metrics_bootstrap = dict([(metric,np.empty((n_iter,np.max((1,n_boot))))) for metric in metrics])
    metrics_bootstrap_best = dict([(metric,np.empty(np.max((1,n_boot)))) for metric in metrics])
    
    all_outputs_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0]*len(random_seeds_outer),2,n_iter)) if problem_type == 'clf' else np.empty((np.max((1,n_boot)),X.shape[0]*len(random_seeds_outer),n_iter))
    outputs_bootstrap_best = np.empty((np.max((1,n_boot)),X.shape[0]*len(random_seeds_outer),2)) if problem_type == 'clf' else np.empty((np.max((1,n_boot)),X.shape[0]*len(random_seeds_outer)))

    y_true_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0]*len(random_seeds_outer)))

    all_y_pred_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0]*len(random_seeds_outer),n_iter))
    y_pred_bootstrap_best = np.empty((np.max((1,n_boot)),X.shape[0]*len(random_seeds_outer)))

    IDs_val_bootstrap = np.empty((np.max((1,n_boot)),X.shape[0]*len(random_seeds_outer)),dtype=object)

    model_rfecv = model()
    model_bayes = model()

    if hasattr(model_rfecv,'kernel'):
        model_rfecv.kernel = 'linear'
    if hasattr(model_rfecv,'random_state'):
        model_rfecv.random_state = 42
        model_bayes.random_state = 42

    search = BayesSearchCV(model_bayes,hyperp,scoring=scoring,n_iter=n_iter,cv=iterator_inner,random_state=42,n_jobs=-1)
    
    y_val = np.empty((X.shape[0],len(random_seeds_outer)))
    IDs_val = np.empty((X.shape[0],len(random_seeds_outer)),dtype=object)
    outputs_val = np.empty((X.shape[0],2,n_iter*iterator_outer.get_n_splits(),len(random_seeds_outer))) if problem_type == 'clf' else np.empty((X.shape[0],n_iter*iterator_outer.get_n_splits(),len(random_seeds_outer)))
    outputs_val_best = np.empty((X.shape[0],2,len(random_seeds_outer))) if problem_type == 'clf' else np.empty((X.shape[0],len(random_seeds_outer)))
    
    for r,random_seed in enumerate(random_seeds_outer):
        iterator_outer.random_state = random_seed

        for fold,(train_index,test_index) in enumerate(iterator_outer.split(X,y)): 
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ID_train, ID_test = IDs[train_index], IDs[test_index]

            y_val[test_index,r] = y_test
            IDs_val[test_index,r] = ID_test
            if model == KNNC or model == KNNR:
                feature_set = features
            else:    
                rfecv = RFECV(estimator=model_rfecv,step=1,scoring=scoring,cv=iterator_inner,n_jobs=-1)
                feature_set = features[rfecv.fit(X_train,y_train).support_]

            search.fit(X_train[feature_set],y_train)
                
            best_models.loc[best_models.shape[0],'random_seed'] = random_seed
            best_models.loc[best_models.shape[0]-1,'fold'] = fold
            best_models.loc[best_models.shape[0]-1,hyperp.keys()] = search.best_params_
            best_models.loc[best_models.shape[0]-1,features] = [1 if feature in feature_set else 0 for feature in features]

            for p,params in enumerate(search.cv_results_['params']):
                
                all_models.loc[all_models.shape[0],'random_seed'] = random_seed
                all_models.loc[all_models.shape[0]-1,'fold'] = fold
                for param in params:
                    all_models.loc[all_models.shape[0]-1,param] = params[param]
                all_models.loc[all_models.shape[0]-1,features] = [1 if feature in feature_set else 0 for feature in features]

                mod = Model(model(**params),scaler)
                mod.train(X_train[feature_set],y_train) 
                if problem_type == 'clf':
                    outputs_val[test_index,:,p,r] = mod.eval(X_test[feature_set],problem_type) 
                else:
                    outputs_val[test_index,p,r] = mod.eval(X_test[feature_set],problem_type)

            if problem_type == 'clf':
                outputs_val_best[test_index,:,r] = outputs_val[test_index,:,search.best_index_,r]   
            else:
                outputs_val_best[test_index,r] = outputs_val[test_index,search.best_index_,r]
    for b in range(np.max((1,n_boot))):
        boot_index = resample(np.arange(outputs_val.shape[0]),n_samples=outputs_val.shape[0],replace=True,random_seed=b) if n_boot > 0 else np.arange(outputs_val.shape[0])

        y_true_bootstrap[b,:] = np.concatenate([y_val[boot_index,r] for r in range(len(random_seeds_outer))])
        IDs_val_bootstrap[b,:] = np.concatenate([IDs_val[boot_index,r] for r in range(len(random_seeds_outer))])
        for i in range(n_iter):
            if problem_type == 'clf':
                outputs = np.concatenate([outputs_val[boot_index,:,i,r] for r in range(len(random_seeds_outer))],axis=0)
                all_outputs_bootstrap[b,:,:,i] = outputs

                metrics_bootstrap,y_pred = get_metrics_clf(outputs,np.concatenate([y_val[boot_index,r] for r in range(len(random_seeds_outer))]),metrics,cmatrix,priors)
                all_y_pred_bootstrap[b,:,i] = y_pred
            else:
                outputs = np.concatenate([outputs_val[boot_index,i,r] for r in range(len(random_seeds_outer))],axis=0)
                all_outputs_bootstrap[b,:,i] = outputs
                metrics_bootstrap = get_metrics_reg(outputs,np.concatenate([y_val[boot_index,r] for r in range(len(random_seeds_outer))]),metrics)
            for metric in metrics:
                all_metrics_bootstrap[metric][i,b] = metrics_bootstrap[metric]
    
        if problem_type == 'clf':
            outputs_best = np.concatenate([outputs_val_best[boot_index,:,r] for r in range(len(random_seeds_outer))],axis=0)
            outputs_bootstrap_best[b,:,:] = outputs_best
        
            metrics_bootstrap,y_pred = get_metrics_clf(outputs_best,np.concatenate([y_val[boot_index,r] for r in range(len(random_seeds_outer))]),metrics,cmatrix,priors)
            y_pred_bootstrap_best[b,:] = y_pred
        else:
            outputs_best = np.concatenate([outputs_val_best[boot_index,r] for r in range(len(random_seeds_outer))],axis=0)
            outputs_bootstrap_best[b,:] = outputs_best
            metrics_bootstrap = get_metrics_reg(outputs_best,np.concatenate([y_val[boot_index,r] for r in range(len(random_seeds_outer))]),metrics)
        for metric in metrics:
            metrics_bootstrap_best[metric][b] = metrics_bootstrap[metric]

    return all_models,best_models,all_outputs_bootstrap,outputs_bootstrap_best,all_y_pred_bootstrap,y_pred_bootstrap_best,all_metrics_bootstrap,metrics_bootstrap_best,y_true_bootstrap,IDs_val_bootstrap

def compare(models,X_dev,y_dev,iterator,random_seeds_train,metric_name,IDs_dev,n_boot=100,cmatrix=None,priors=None,problem_type='clf'):
    metrics = np.empty((np.max((1,n_boot)),len(models)))
    
    for m,model in enumerate(models.keys()):
        _,metrics_bootstrap,_,_,_,_,_ = CV(0,models[model],X_dev[model],y_dev,X_dev[model].columns,iterator,random_seeds_train,metric_name,IDs_dev,n_boot=n_boot,cmatrix=cmatrix,priors=priors,problem_type=problem_type)
        metrics[:,m] = metrics_bootstrap[metric_name[0]]
    return metrics
