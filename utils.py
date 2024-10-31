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

from bayes_opt import BayesianOptimization

class Model():
    def __init__(self,model,scaler=None,imputer=None,calibrator=None):
        self.model = model
        self.scaler = scaler() if scaler is not None else None
        self.imputer = imputer() if imputer is not None else None
        self.calibrator = calibrator() if calibrator is not None else None

    def train(self,X,y):   
        features = X.columns
        
        X_t = pd.DataFrame(columns=features,data=self.scaler.fit_transform(X.values)) if self.scaler is not None else X
        X_t = pd.DataFrame(columns=features,data=self.imputer.fit_transform(X_t.values)) if self.imputer is not None else X_t

        params = self.model.get_params()
        if 'n_estimators' in params.keys():
            params['n_estimators'] = int(params['n_estimators']) if params['n_estimators'] is not None else None
        if 'n_neighbors' in params.keys():
            params['n_neighbors'] = int(params['n_neighbors'])
        if 'max_depth' in params.keys():
            params['max_depth'] = int(params['max_depth']) if params['max_depth'] is not None else None

        self.model.set_params(**params)
        if hasattr(self.model,'precompute'):
            self.model.precompute = True
        self.model.fit(X_t,y)
        
    def eval(self,X,problem_type='clf'):
        
        features = X.columns
        X_t = pd.DataFrame(columns=features,data=self.scaler.transform(X.values)) if self.scaler is not None else X
        X_t = pd.DataFrame(columns=features,data=self.imputer.transform(X_t.values)) if self.imputer is not None else X_t

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
            
def CV(model_class,params,scaler,imputer,X,y,all_features,threshold,iterator,random_seeds_train,IDs,cmatrix=None,priors=None,problem_type='clf'):
    
    """
    Cross-validation function to train and evaluate a model with specified parameters, 
    feature engineering, and evaluation metrics. Supports both classification and regression.

    Parameters
    ----------
    model_class : class
        The model class (e.g., sklearn model) to instantiate for training and evaluation.
    params : dict
        Parameters to initialize the model.
    scaler : object
        Scaler instance to preprocess the feature data.
    imputer : object
        Imputer instance to handle missing values.
    X : pd.DataFrame
        Input features data.
    y : pd.Series or np.array
        Target values.
    all_features : list
        List of all possible feature names, marking presence or absence for feature engineering.
    threshold : float
        Decision threshold for classification tasks.
    iterator : cross-validation generator
        Cross-validation iterator to split the dataset.
    random_seeds_train : list
        List of random seeds for reproducibility in training and evaluation.
    IDs : np.array
        Array of sample identifiers for tracking predictions.
    cmatrix : CostMatrix, optional
        Cost matrix for classification; defaults to [[0,1],[1,0]] if not provided.
    priors : dict, optional
        Class priors for probability calibration in classification.
    problem_type : str, optional
        Specifies 'clf' for classification or 'reg' for regression tasks (default is 'clf').

    Returns
    -------
    model_params : pd.DataFrame
        DataFrame of model parameters used for training.
    outputs_dev : np.array
        Array of model outputs per cross-validation fold for each sample.
    y_dev : np.array
        Array of true target values across folds.
    y_pred : np.array
        Array of predicted values across folds.
    IDs_dev : np.array
        Array of IDs for samples used in predictions across folds.

    """
     
    if cmatrix is None:
        cmatrix = CostMatrix([[0,1],[1,0]])

    model_params = params.copy()

    features = dict((feature,0) for feature in all_features)
    
    for feature in X.columns:
        features[feature] = 1
    
    model_params.update(features)

    if problem_type == 'clf':
        model_params.update({'threshold':threshold})
            
    model_params = pd.DataFrame(model_params,index=[0])

    X_dev = np.empty((len(random_seeds_train),X.shape[0],X.shape[1]))
    y_dev = np.empty((len(random_seeds_train),X.shape[0]))
    IDs_dev = np.empty((len(random_seeds_train),X.shape[0]),dtype=object)
    y_pred = np.empty((len(random_seeds_train),X.shape[0]))

    outputs_dev = np.empty((len(random_seeds_train),X.shape[0],2)) if problem_type == 'clf' else np.empty((len(random_seeds_train),X.shape[0]))

    for r,random_seed in enumerate(random_seeds_train):

        iterator.random_state = random_seed

        for train_index, test_index in iterator.split(X,y):
            model = Model(model_class(**params),scaler,imputer)
            if hasattr(model.model,'random_state'):
                model.model.random_state = 42

            model.train(X.loc[train_index], y[train_index])

            X_dev[r,test_index] = X.loc[test_index]
            y_dev[r,test_index] = y[test_index]

            IDs_dev[r,test_index] = IDs[test_index]

            outputs_dev[r,test_index] = model.eval(X.loc[test_index],problem_type)

        if problem_type == 'clf':
            _,y_pred[r] = get_metrics_clf(outputs_dev[r],y_dev[r],[],cmatrix,priors,threshold) 
        else:
            y_pred[r] = outputs_dev[r]

    return model_params,outputs_dev,y_dev,y_pred,IDs_dev

def CVT(model,scaler,imputer,X,y,iterator,random_seeds_train,hyperp,feature_sets,IDs,thresholds=[None],cmatrix=None,priors=None,parallel=True,problem_type='clf'):
    
    """
    Cross-validation testing function for model training and evaluation with hyperparameter 
    tuning, feature set selection, and parallel processing options. Supports classification 
    and regression tasks.

    Parameters
    ----------
    model : class
        The model class (e.g., sklearn model) to instantiate for training and evaluation.
    scaler : object
        Scaler instance to preprocess feature data.
    imputer : object
        Imputer instance to handle missing values.
    X : pd.DataFrame
        Input features data.
    y : pd.Series or np.array
        Target values.
    iterator : cross-validation generator
        Cross-validation iterator to split the dataset.
    random_seeds_train : list
        List of random seeds for reproducibility in training and evaluation.
    hyperp : pd.DataFrame
        DataFrame of hyperparameter values for each model configuration.
    feature_sets : list of lists
        List of feature subsets to evaluate, each list containing feature names.
    IDs : np.array
        Array of sample identifiers for tracking predictions.
    thresholds : list, optional
        List of decision thresholds for classification; defaults to [None].
    cmatrix : CostMatrix, optional
        Cost matrix for classification; defaults to [[0,1],[1,0]] if not provided.
    priors : dict, optional
        Class priors for probability calibration in classification.
    parallel : bool, optional
        If True, enables parallel processing for cross-validation tasks (default is True).
    problem_type : str, optional
        Specifies 'clf' for classification or 'reg' for regression tasks (default is 'clf').

    Returns
    -------
    all_models : pd.DataFrame
        DataFrame of all model configurations, including hyperparameters and feature sets.
    all_outputs : np.array
        Array of model outputs for all pooled samples across each configuration and random 
        seed in cross-validation.
    all_y_pred : np.array
        Array of predicted values across configurations and folds.
    y_true : np.array
        Array of true target values.
    IDs_dev : np.array
        Array of IDs for samples used in predictions across folds.

    """
    
    features = X.columns
    
    if hasattr(model(),'random_state') and model != SVR:
        hyperp['random_state'] = 42

    all_models = pd.DataFrame(columns=list(hyperp.columns) + list(features))

    if parallel == True:
        results = Parallel(n_jobs=-1)(delayed(CV)(model,hyperp.loc[c,:].to_dict(),scaler,imputer,X[feature_set],y,X.columns,threshold,iterator,random_seeds_train,IDs,cmatrix,priors,problem_type) for c,feature_set,threshold in itertools.product(hyperp.index,feature_sets,thresholds))
        
        all_models = pd.concat([result[0] for result in results],ignore_index=True,axis=0)
        
        all_outputs = np.concatenate(([np.expand_dims(result[1],axis=0) for result in results]),axis=0)
        y_true = results[0][2]
        all_y_pred = np.concatenate(([np.expand_dims(result[3],axis=0) for result in results]),axis=0)
        IDs_dev = results[0][4]
    else:
        all_outputs = np.empty((hyperp.shape[0]*len(feature_sets),len(random_seeds_train),X.shape[0],2)) if problem_type == 'clf' else np.empty((hyperp.shape[0]*len(feature_sets),len(random_seeds_train),X.shape[0]))
    
        y_true = np.empty((len(random_seeds_train),X.shape[0]))

        all_y_pred = np.empty((hyperp.shape[0]*len(feature_sets),len(random_seeds_train),X.shape[0]))

        IDs_dev = np.empty((len(random_seeds_train),X.shape[0]))

        for c, (comb,threshold,feature_set) in enumerate(itertools.product(range(hyperp.shape[0]),thresholds,feature_sets)):
            params = hyperp.iloc[comb,:].to_dict()
            all_models.loc[c,params.keys()] = hyperp.iloc[comb,:].values
            all_models.loc[c,features] = [1 if feature in feature_set else 0 for feature in features]
            _,outputs_c, y_true, y_pred,IDs_dev = CV(c,model,params,scaler,imputer,X[feature_set],y,X.columns,threshold,iterator,random_seeds_train,IDs,cmatrix,priors,problem_type)
            
            if problem_type == 'clf':
                all_outputs[c] = outputs_c
            else:
                all_outputs[c] = outputs_c
            
            all_y_pred[c] = y_pred
            
    return all_models,all_outputs,all_y_pred,y_true,IDs_dev

def test_model(model_class,params,scaler,imputer, X_dev, y_dev, X_test, y_test, metrics, IDs_test,
               n_boot_train=0, n_boot_test=0, cmatrix=None, priors=None, problem_type='clf',threshold=None):
    
    """
    Tests a model on specified development and test datasets, using optional bootstrapping for 
    training and evaluation, and calculates metrics based on provided criteria. Supports both 
    classification and regression.

    Parameters
    ----------
    model_class : class
        The model class (e.g., sklearn model) to instantiate for training and evaluation.
    params : dict
        Parameters to initialize the model.
    scaler : object
        Scaler instance to preprocess the feature data.
    imputer : object
        Imputer instance to handle missing values.
    X_dev : pd.DataFrame or np.array
        Development data features for training the model.
    y_dev : pd.Series or np.array
        Target values for the development dataset.
    X_test : pd.DataFrame or np.array
        Test data features for evaluating the model.
    y_test : pd.Series or np.array
        True target values for the test dataset.
    metrics : list of str
        List of metric names to evaluate (e.g., 'accuracy', 'precision').
    IDs_test : np.array
        Array of sample identifiers for tracking predictions on the test dataset.
    n_boot_train : int, optional
        Number of bootstrap samples for training; if 0, no bootstrapping is used (default is 0).
    n_boot_test : int, optional
        Number of bootstrap samples for testing; if 0, no bootstrapping is used (default is 0).
    cmatrix : CostMatrix, optional
        Cost matrix for classification; defaults to None.
    priors : dict, optional
        Class priors for probability calibration in classification.
    problem_type : str, optional
        Specifies 'clf' for classification or 'reg' for regression tasks (default is 'clf').
    threshold : float, optional
        Decision threshold for classification tasks.

    Returns
    -------
    metrics_test_bootstrap : dict
        Dictionary of test metrics, with metric names as keys and arrays of bootstrap metric values.
    outputs_bootstrap : np.array
        Array of model outputs for each test sample across bootstrap iterations.
    y_true_bootstrap : np.array
        Array of true target values for each bootstrap iteration of test samples.
    y_pred_bootstrap : np.array
        Array of predicted values across bootstrap iterations (for classification tasks).
    IDs_test_bootstrap : np.array
        Array of sample IDs used in bootstrap test predictions.

    """

    if not isinstance(X_dev, pd.DataFrame):
        X_dev = pd.DataFrame(X_dev)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    outputs_bootstrap = np.empty((0, 2)) if problem_type == 'clf' else np.empty(0)
    y_true_bootstrap = np.empty(0)
    y_pred_bootstrap = np.empty(0)
    IDs_test_bootstrap = np.empty(0, dtype=object)
    metrics_test_bootstrap = {metric: np.empty(0) for metric in metrics}

    for b_train in range(np.max((1,n_boot_train))):
        boot_index_train = resample(X_dev.index, n_samples=X_dev.shape[0], replace=True, random_state=b_train) if n_boot_train > 0 else X_dev.index
        model = Model(model_class(**params),scaler,imputer)
        model.train(X_dev.loc[boot_index_train], y_dev[boot_index_train])

        for b_test in range(np.max((1,n_boot_test))):
            boot_index = resample(X_test.index, n_samples=X_test.shape[0], replace=True, random_state=b_train * np.max((1,n_boot_train)) + b_test) if n_boot_test > 0 else X_test.index

            y_true_bootstrap = np.concatenate((y_true_bootstrap,y_test[boot_index]))
            IDs_test_bootstrap = np.concatenate((IDs_test_bootstrap,IDs_test[boot_index]))
            outputs = model.eval(X_test.loc[boot_index, :], problem_type)

            if problem_type == 'clf':
                metrics_test, y_pred = get_metrics_clf(outputs, y_test[boot_index], metrics, cmatrix, priors,threshold)
                y_pred_bootstrap = np.concatenate((y_pred_bootstrap,y_pred))
            else:
                metrics_test = get_metrics_reg(outputs, y_test[boot_index], metrics)
            
            outputs_bootstrap = np.concatenate((outputs_bootstrap,outputs))

            for metric in metrics:
                metrics_test_bootstrap[metric] = np.concatenate((metrics_test_bootstrap[metric],[metrics_test[metric]]))

    return metrics_test_bootstrap, outputs_bootstrap, y_true_bootstrap, y_pred_bootstrap, IDs_test_bootstrap

def nestedCVT(model_class,scaler,imputer,X,y,n_iter,iterator_outer,iterator_inner,random_seeds_outer,hyperp_space,IDs,scoring='roc_auc_score',problem_type='clf',cmatrix=None,priors=None,threshold=None):
    
    """
    Conducts nested cross-validation with recursive feature elimination (RFE) and hyperparameter tuning 
    to select and evaluate the best model configuration. Supports classification and regression tasks.

    Parameters
    ----------
    model_class : class
        The model class (e.g., sklearn model) to instantiate for training and evaluation.
    scaler : callable
        Scaler function to initialize and fit a scaler for feature preprocessing.
    imputer : callable
        Imputer function to initialize and fit an imputer for handling missing values.
    X : pd.DataFrame
        Input features data.
    y : pd.Series or np.array
        Target values.
    n_iter : int
        Number of iterations for hyperparameter tuning.
    iterator_outer : cross-validation generator
        Outer cross-validation iterator to split the dataset.
    iterator_inner : cross-validation generator
        Inner cross-validation iterator for tuning and feature selection.
    random_seeds_outer : list
        List of random seeds for reproducibility across outer folds.
    hyperp_space : dict
        Dictionary defining hyperparameter search space for model tuning.
    IDs : np.array
        Array of sample identifiers for tracking predictions across folds.
    scoring : str, optional
        Metric name for model selection (e.g., 'roc_auc_score') (default is 'roc_auc_score').
    problem_type : str, optional
        Specifies 'clf' for classification or 'reg' for regression tasks (default is 'clf').
    cmatrix : CostMatrix, optional
        Cost matrix for classification; defaults to None.
    priors : dict, optional
        Class priors for probability calibration in classification.
    threshold : float, optional
        Decision threshold for classification tasks.

    Returns
    -------
    all_models : pd.DataFrame
        DataFrame of all model configurations, including hyperparameters, feature selections, and scores.
    outputs_best : np.array
        Array of model outputs for each sample across configurations and random seeds.
    y_true : np.array
        Array of true target values for each sample across configurations.
    y_pred_best : np.array
        Array of predicted values across configurations and random seeds.
    IDs_val : np.array
        Array of IDs for samples used in predictions across outer folds.
        
    """
    
    features = X.columns
    
    iterator_inner.random_state = 42
    all_models = pd.DataFrame(columns=['random_seed','fold','threshold',scoring] + list(hyperp_space.keys()) + list(features))

    outputs_best = np.empty((len(random_seeds_outer),X.shape[0],2)) if problem_type == 'clf' else np.empty((len(random_seeds_outer),X.shape[0]))

    y_true = np.empty((len(random_seeds_outer),X.shape[0]))

    y_pred_best = np.empty((len(random_seeds_outer),X.shape[0]))

    model_rfecv = model_class()

    if hasattr(model_rfecv,'kernel'):
        model_rfecv.kernel = 'linear'
    if hasattr(model_rfecv,'probability'):
        model_rfecv.probability = True
    if hasattr(model_rfecv,'random_state'):
        model_rfecv.random_state = 42
    
    IDs_val = np.empty((len(random_seeds_outer),X.shape[0]),dtype=object)
    
    for r,random_seed in enumerate(random_seeds_outer):
        iterator_outer.random_state = random_seed
        
        for k,(train_index_out,test_index_out) in enumerate(iterator_outer.split(X,y)): 
            X_dev, X_test = X.loc[train_index_out], X.loc[test_index_out]
            y_dev, y_test = y[train_index_out], y[test_index_out]
            
            X_dev = X_dev.reset_index(drop=True)
            y_dev = y_dev.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            y_true[r,test_index_out] = y_test
            #ID_dev, ID_test = IDs[train_index_out], IDs[test_index_out]
            IDs_val[r,test_index_out] = IDs[test_index_out]

            scaler_ = scaler().fit(X_dev)
            imputer_ = imputer().fit(X_dev)

            X_dev = pd.DataFrame(columns=X.columns,data=imputer_.transform(scaler_.transform(X_dev)))
            X_test = pd.DataFrame(columns=X.columns,data=imputer_.transform(scaler_.transform(X_test)))
            best_features = rfe(Model(model_rfecv,scaler,imputer),X_dev,y_dev,iterator_inner,scoring,problem_type,cmatrix,priors,threshold)
            best_params, best_score = tuning(model_class,scaler,imputer,X_dev[best_features],y_dev,hyperp_space,iterator_inner,n_iter,scoring,problem_type,cmatrix,priors,threshold)

            if 'n_estimators' in best_params.keys():
                best_params['n_estimators'] = int(best_params['n_estimators'])
            elif 'n_neighbors' in best_params.keys():
                best_params['n_neighbors'] = int(best_params['n_neighbors'])
            elif 'max_depth' in best_params.keys():
                best_params['max_depth'] = int(best_params['max_depth'])
            
            if hasattr(model_class(),'random_state'):
                best_params['random_state'] = int(42)
            if hasattr(model_class(),'probability'):
                best_params['probability'] = True

            append_dict = {'random_seed':random_seed,'fold':k,'threshold':threshold,scoring:best_score}
            append_dict.update(best_params)
            append_dict.update({feature:1 if feature in best_features else 0 for feature in X_dev.columns}) 

            all_models.loc[len(all_models.index),:] = append_dict

            model = Model(model_class(**best_params),scaler,imputer)
            model.train(X_dev[best_features],y_dev)
            
            if problem_type == 'clf':
                outputs_best_ = model.eval(X_test[best_features],problem_type)
                if threshold is not None:
                    y_pred_best_ = [1 if np.exp(x) > threshold else 0 for x in outputs_best[r,test_index_out][:,1]]
                else:
                    y_pred_best_= bayes_decisions(scores=outputs_best[r,test_index_out],costs=cmatrix,priors=priors,score_type='log_posteriors')[0]
                
                y_pred_best[r,test_index_out] = y_pred_best_

            else:
                outputs_best_ = model.eval(X_test[best_features],problem_type)
                y_pred_best[r,test_index_out] = outputs_best_
            outputs_best[r,test_index_out] = outputs_best_
            
    return all_models,outputs_best,y_true,y_pred_best,IDs_val

def rfe(model, X, y, iterator, scoring='roc_auc_score', problem_type='clf',cmatrix=None,priors=None,threshold=None):
    
    """
    Performs recursive feature elimination (RFE) to select the best subset of features based on a 
    scoring metric. Iteratively removes features that lead to the smallest decrease in the scoring metric.

    Parameters
    ----------
    model : object
        Model instance to train and evaluate on the feature subsets.
    X : pd.DataFrame
        Feature dataset for model training and evaluation.
    y : pd.Series or np.array
        Target variable for training and validation.
    iterator : cross-validation generator
        Cross-validation iterator to split the data into training and validation sets.
    scoring : str, optional
        Scoring metric used to evaluate feature subsets (e.g., 'roc_auc_score') (default is 'roc_auc_score').
    problem_type : str, optional
        Specifies 'clf' for classification or 'reg' for regression tasks (default is 'clf').
    cmatrix : CostMatrix, optional
        Cost matrix for classification, defaults to None.
    priors : dict, optional
        Class priors for probability calibration in classification.
    threshold : float, optional
        Decision threshold for classification tasks.

    Returns
    -------
    best_features : list
        List of selected features after recursive feature elimination.
        
    """

    features = list(X.columns)
    
    # Ascending if error, loss, or other metrics where lower is better
    ascending = any(x in scoring for x in ['error', 'loss', 'norm'])
    best_score = np.inf if ascending else -np.inf
    best_features = features.copy()

    while len(features) > 1:
        scorings = {}  # Dictionary to hold scores for each feature removal
        
        for feature in features:
            print('Evaluating without feature:', feature)
            
            outputs = np.empty((X.shape[0], 2)) if problem_type == 'clf' else np.empty(X.shape[0])
            y_pred = np.empty(X.shape[0])
            y_true = np.empty(X.shape[0])
            
            for train_index, val_index in iterator.split(X, y):
                X_train = X.iloc[train_index][[f for f in features if f != feature]]
                X_val = X.iloc[val_index][[f for f in features if f != feature]]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                
                model.train(X_train, y_train)
                
                if problem_type == 'clf':
                    outputs_ = model.eval(X_val,problem_type)
                    if threshold is not None:
                        y_pred[val_index] = [1 if np.exp(x) > threshold else 0 for x in outputs_[:,1]]
                    else:
                        y_pred[val_index] = bayes_decisions(scores=outputs_,costs=cmatrix,priors=priors,score_type='log_posteriors')[0]
                else:
                    outputs[val_index] = model.eval(X_val,problem_type)
                    y_pred[val_index] = outputs[val_index]
                y_true[val_index] = y_val
            # Choose the appropriate scoring function
            if scoring == 'roc_auc_score':
                scorings[feature] = eval(scoring)(y_true, outputs[:, 1] if problem_type == 'clf' else outputs)
            else:
                scorings[feature] = eval(scoring)(y_true, y_pred)
            # Add other scoring metrics as needed

        # Sort features by score to find the best to remove
        scorings = pd.DataFrame(list(scorings.items()), columns=['feature', 'score']).sort_values(
            by='score', ascending=ascending).reset_index(drop=True)
        
        best_feature_score = scorings['score'][0]
        feature_to_remove = scorings['feature'][0]
        
        # If improvement is found, update best score and feature set
        if new_best(best_score, best_feature_score, not ascending):
            best_score = best_feature_score
            features.remove(feature_to_remove)
            best_features = features.copy()
            print(f"Removing feature: {feature_to_remove}, New Best Score: {best_score}")
        else:
            # Stop if no improvement
            print("No further improvement. Stopping feature elimination.")
            break

    return best_features

def new_best(old,new,greater=True):
    if greater:
        return new > old
    else:
        return new < old

def tuning(model,scaler,imputer,X,y,hyperp_space,iterator,n_iter=50,scoring='roc_auc_score',problem_type='clf',cmatrix=None,priors=None,threshold=None):
    
    search = BayesianOptimization(lambda **params: scoring_bo(params,model,scaler,imputer,X,y,iterator,scoring,problem_type,cmatrix,priors,threshold),hyperp_space)
    search.maximize(n_iter=n_iter)
    return search.max['params'], search.max['target']

def scoring_bo(params,model_class,scaler,imputer,X,y,iterator,scoring,problem_type,cmatrix=None,priors=None,threshold=None):

    """
    Evaluates a model's performance using cross-validation and a specified scoring metric, 
    facilitating hyperparameter optimization with Bayesian optimization or similar approaches.

    Parameters
    ----------
    params : dict
        Dictionary of model hyperparameters.
    model_class : class
        The model class to instantiate for training and evaluation.
    scaler : callable
        Function to initialize and fit a scaler for feature preprocessing.
    imputer : callable
        Function to initialize and fit an imputer for missing value handling.
    X : pd.DataFrame
        Feature data for training and evaluation.
    y : pd.Series or np.array
        Target variable.
    iterator : cross-validation generator
        Cross-validation iterator to split the data into training and testing sets.
    scoring : str
        Scoring metric for evaluating model performance (e.g., 'roc_auc_score').
    problem_type : str
        Specifies 'clf' for classification or 'reg' for regression tasks.
    cmatrix : CostMatrix, optional
        Cost matrix for classification, defaults to None.
    priors : dict, optional
        Class priors for probability calibration in classification tasks.
    threshold : float, optional
        Decision threshold for classification predictions.

    Returns
    -------
    float
        The computed score based on the chosen scoring metric and the model's cross-validated performance.
    """

    if 'n_estimators' in params.keys():
        params['n_estimators'] = int(params['n_estimators'])
    elif 'n_neighbors' in params.keys():
        params['n_neighbors'] = int(params['n_neighbors'])
    elif 'max_depth' in params.keys():
        params['max_depth'] = int(params['max_depth'])
    if 'random_state' in params.keys():
        params['random_state'] = int(42)
    
    if hasattr(model_class(),'probability'):
        params['probability'] = True
        
    y_true = np.empty(X.shape[0])
    y_pred = np.empty(X.shape[0])
    outputs = np.empty((X.shape[0],2)) if problem_type == 'clf' else np.empty(X.shape[0])
    
    for train_index, test_index in iterator.split(X,y):
        model = Model(model_class(**params),scaler,imputer)
        model.train(X.loc[train_index],y[train_index])
        if problem_type == 'clf':
            outputs[test_index] = model.eval(X.loc[test_index],problem_type)
            if threshold is not None:
                y_pred[test_index] = [1 if np.exp(x) > threshold else 0 for x in outputs[test_index,1]]
            else:
                y_pred[test_index] = bayes_decisions(scores=outputs[test_index],costs=cmatrix,priors=priors,score_type='log_posteriors')[0]
        else:
            outputs[test_index] = model.eval(X.loc[test_index],problem_type)
        y_true[test_index] = y[test_index]
    
    if 'error' in scoring:
        return -eval(scoring)(y_true=y,y_pred=outputs)
    elif scoring == 'roc_auc_score':
        return eval(scoring)(y_true=y,y_score=outputs[:,1])
    else:
        return eval(scoring)(y_true=y,y_pred=y_pred)

def compare(models,X_dev,y_dev,iterator,random_seeds_train,metric_name,IDs_dev,n_boot=100,cmatrix=None,priors=None,problem_type='clf'):
    metrics = np.empty((np.max((1,n_boot)),len(models)))
    
    for m,model in enumerate(models.keys()):
        _,metrics_bootstrap,_,_,_,_,_ = CV(0,models[model],X_dev[model],y_dev,X_dev[model].columns,iterator,random_seeds_train,metric_name,IDs_dev,n_boot=n_boot,cmatrix=cmatrix,priors=priors,problem_type=problem_type)
        metrics[:,m] = metrics_bootstrap[metric_name[0]]
    return metrics

def css(metrics,scoring='roc_auc',problem_type='clf'):
    inf_conf_int = np.empty(metrics[scoring].shape[0])
    sup_conf_int = np.empty(metrics[scoring].shape[0])

    for model in range(metrics[scoring].shape[0]):
        _, inf_conf_int[model], sup_conf_int[model] = conf_int_95(metrics[scoring][model])
        
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