import numpy as np
from sklearn.metrics import *
import torch,itertools
import pandas as pd
from sklearn.svm import SVR 

from sklearn import metrics

from expected_cost.ec import *
from expected_cost.utils import *
from expected_cost.calibration import *

from psrcal.losses import LogLoss
from psrcal.calibration import *

from joblib import Parallel, delayed

import math

from bayes_opt import BayesianOptimization

class Model():
    def __init__(self,model,scaler=None,imputer=None,calmethod=None,calparams=None):
        self.model = model
        self.scaler = scaler() if scaler is not None else None
        self.imputer = imputer() if imputer is not None else None
        self.calmethod = calmethod
        self.calparams = calparams

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
        if 'max_iter' in params.keys():
            params['max_iter'] = int(params['max_iter']) if params['max_iter'] is not None else None
        if 'gpu_id' in params.keys():
                params['gpu_id'] = None
                
        self.model.set_params(**params)
        if hasattr(self.model,'precompute'):
            self.model.precompute = True
        self.model.fit(X_t,y)
        
    def eval(self,X,problem_type='clf'):
        
        features = X.columns
        X_t = pd.DataFrame(columns=features,data=self.scaler.transform(X.values)) if self.scaler is not None else X
        X_t = pd.DataFrame(columns=features,data=self.imputer.transform(X_t.values)) if self.imputer is not None else X_t
        if problem_type == 'clf':
            prob = self.model.predict_proba(X_t)
            prob = np.clip(prob,1e-2,1-1e-2)
            score = np.log(prob)
        else:
            score = self.model.predict(X_t)

        score_filled = score.copy()

        if problem_type == 'clf':
            nan_indices_col0 = np.isnan(score[:, 0]) # True where first column is NaN
            nan_indices_col1 = np.isnan(score[:, 1])  # True where second column is NaN

            # Replace them accordingly:
            score_filled[nan_indices_col0, 0] = np.log(1e-2)
            score_filled[nan_indices_col1, 1] =  np.log(1-1e-2)
            
        return score_filled
    
    def calibrate(self,logpost_tst,targets_tst,logpost_trn=None,targets_trn=None):
        
        if logpost_trn is None:
            cal_outputs_test = calibration_with_crossval(logpost=logpost_tst,targets=targets_tst,calmethod=self.calmethod,calparams=self.calparams)        
            calmodel = None
        else:
            cal_outputs_test, calmodel = calibration_train_on_heldout(logpost_trn=logpost_trn,targets_trn=targets_trn,logpost_tst=logpost_tst,calmethod=self.calmethod,calparams=self.calparams,return_model=True)        

        return cal_outputs_test, calmodel
    
def get_metrics_clf(y_scores,y_true,metrics_names,cmatrix=None,priors=None,threshold=None,weights=None):
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
    try:
        if np.isnan(threshold):
            threshold = None
    except:
        threshold = None

    if cmatrix is None:
        cmatrix = CostMatrix.zero_one_costs(K=y_scores.shape[1])

    y_pred = bayes_decisions(scores=y_scores,costs=cmatrix,priors=priors,score_type='log_posteriors')[0] if threshold is None else np.array(y_scores[:,1] > threshold,dtype=int)

    if not (np.array_equal(np.unique(y_pred), np.unique(y_true)) & (len(np.unique(y_pred)) == 2)):
        metrics_names = list(set(metrics_names) - set(['accuracy']))

    metrics = dict([(metric,[]) for metric in metrics_names])

    for m in metrics_names:
        if m == 'norm_cross_entropy':
            metrics[m] = float(LogLoss(log_probs=torch.tensor(y_scores),labels=torch.tensor(np.array(y_true),dtype=torch.int),priors=torch.tensor(priors)).detach().numpy()) if priors is not None else float(LogLoss(log_probs=torch.tensor(y_scores),labels=torch.tensor(np.array(y_true),dtype=torch.int)).detach().numpy())
        elif m == 'norm_expected_cost':
            metrics[m] = average_cost(targets=np.array(y_true,dtype=int),decisions=np.array(y_pred,dtype=int),costs=cmatrix,priors=priors,adjusted=True)
        elif m == 'roc_auc':
            metrics[m] = roc_auc_score(y_true=y_true,y_score=y_scores[:,1],sample_weight=weights)
        else:
            metrics[m] = eval(f'{m}_score')(y_true=np.array(y_true,dtype=int),y_pred=y_pred,sample_weight=weights)
        
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
        metrics[metric] = eval(metric)(y_true=y_true,y_pred=y_scores)

    return metrics

def conf_int_95(data):
    mean = np.nanmean(data)
    inf = np.nanpercentile(data,2.5)
    sup = np.nanpercentile(data,97.5) 
    return mean, inf, sup

def initialize_hyperparameters(model_key,config,default_hp,hp_range):
    """
    Initialize hyperparameter DataFrame for a given model.
    In a production system this might be replaced by loading a pre-tuned configuration.
    """
    # Default hyperparameters (as a pandas DataFrame)

    n = 0
    hp = pd.DataFrame(default_hp.get(model_key),index=[0])

    while hp.shape[0] < config['n_iter']+1 and n < 1000:        
        # If no default hyperparameters are available, generate random hyperparameters
        np.random.seed(n)
        new_hp = {key: np.random.choice(hp_range[model_key][key]) for key in hp_range[model_key].keys()}
        
        hp = pd.concat([hp, pd.DataFrame(new_hp,index=[0])], ignore_index=True)

        #Drop duplicates:
        hp = hp.drop_duplicates()
        n += 1

    return hp

def generate_feature_sets(features, config, data_shape):
    """
    Generate a list of feature subsets for evaluation.
    Either compute all combinations up to a maximum length or generate a random sample.
    """
    n_possible = int(config["feature_sample_ratio"] * data_shape[0] * (1 - config["test_size"]) * ((config["n_folds"] - 1) / config["n_folds"])) - 1
    # Determine total number of combinations.
    num_comb = sum(math.comb(len(features), k+1) for k in range(len(features)-1))
    feature_sets = []
    if config["n_iter_features"] > num_comb:
        for k in range(len(features)-1):
            for comb in itertools.combinations(features, k+1):
                feature_sets.append(list(comb))
    else:
        for _ in range(int(config["n_iter_features"])):
            # Use np.random.choice without replacement
            n_iter = 0
            np.random.seed(n_iter)
            new_set = list(np.random.choice(features, np.min((len(features),n_possible)), replace=True))
            #Eliminate duplicates
            new_set = list(set(new_set))
            while sorted(new_set) in feature_sets and n_iter < 100:
                n_iter += 1
                np.random.seed(n_iter)
                new_set = list(set(np.random.choice(features, np.min((len(features),n_possible)), replace=True)))
            feature_sets.append(sorted(new_set))    
            
    # Always include the full feature set.
    feature_sets.append(list(features))
    
    # Ensure that the feature sets are unique.
    feature_sets = list(set([tuple(set(feature_set)) for feature_set in feature_sets]))
    feature_sets = [list(feature_set) for feature_set in feature_sets]
    return feature_sets

def CV(model_class, params, scaler, imputer, X, y, feature_set,all_features, threshold, iterator, random_seeds_train, IDs, problem_type='clf',calmethod=None,calparams=None):
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

    model_params = params.copy()
    features = {feature: 0 for feature in all_features}
    features.update({feature: 1 for feature in feature_set})
    model_params.update(features)

    model_params = pd.DataFrame(model_params, index=[0])

    n_seeds = len(random_seeds_train)
    n_samples, n_features = X.shape
    
    n_classes = len(np.unique(y))

    X_dev = np.empty((n_seeds, n_samples, n_features))
    y_dev = np.empty((n_seeds, n_samples))
    IDs_dev = np.empty((n_seeds, n_samples), dtype=object)
    outputs_dev = np.empty((n_seeds, n_samples, n_classes)) if problem_type == 'clf' else np.empty((n_seeds, n_samples))
    cal_outputs_dev = np.empty_like(outputs_dev)

    iterator.random_state = 42

    for r, random_seed in enumerate(random_seeds_train):
        iterator.random_state = random_seed

        for train_index, test_index in iterator.split(X, y):
            model = Model(model_class(**params), scaler, imputer,calmethod,calparams)
            if hasattr(model.model, 'random_state'):
                model.model.random_state = 42
            
            X_dev[r, test_index] = X.iloc[test_index]
            y_dev[r, test_index] = y.iloc[test_index].values.squeeze()
            IDs_dev[r, test_index] = IDs[test_index]

            model.train(X.iloc[train_index][feature_set], y.iloc[train_index])

            outputs_dev[r, test_index] = model.eval(X.iloc[test_index][feature_set], problem_type)
            if calmethod is not None:
                cal_outputs_dev[r, test_index],_ = model.calibrate(outputs_dev[r,test_index],y_dev[r,test_index])
            else:
                cal_outputs_dev[r, test_index] = outputs_dev[r, test_index]

    if problem_type == 'clf':
        model_params['threshold'] = threshold

    return model_params, outputs_dev, cal_outputs_dev, X_dev, y_dev, IDs_dev

def CVT(model, scaler, imputer, X, y, iterator, random_seeds_train, hyperp, feature_sets, IDs, thresholds=[None], parallel=True, problem_type='clf',calmethod=None,calparams=None):
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
    if not isinstance(thresholds,list):
        thresholds = [thresholds]

    if hasattr(model(), 'random_state') and model != SVR:
        hyperp['random_state'] = 42
    
    all_models = pd.DataFrame(columns=list(hyperp.columns) + list(features),index=range(hyperp.shape[0]*len(feature_sets)*len(thresholds)))
    all_outputs = np.empty((hyperp.shape[0]*len(feature_sets)*len(thresholds), len(random_seeds_train), X.shape[0], len(np.unique(y))))
    all_cal_outputs = np.empty_like(all_outputs)
    X_dev = np.empty((hyperp.shape[0]*len(feature_sets)*len(thresholds), len(random_seeds_train), X.shape[0], X.shape[1]))
    y_true = np.empty((hyperp.shape[0]*len(feature_sets)*len(thresholds), len(random_seeds_train), X.shape[0]))
    IDs_dev = np.empty((hyperp.shape[0]*len(feature_sets)*len(thresholds), len(random_seeds_train), X.shape[0]), dtype=object)

    def process_combination(c,f,t,problem_type,calmethod,calparams):
        params = hyperp.loc[c, :].to_dict()
        return c, f, t, CV(model, params, scaler, imputer, X, y, feature_sets[f],features, thresholds[t], iterator, [int(seed) for seed in random_seeds_train], IDs, problem_type,calmethod,calparams)
        
    if parallel:
        results = Parallel(n_jobs=-1,timeout=300)(delayed(process_combination)(c, f,threshold,problem_type,calmethod,calparams) for c, f, threshold in itertools.product(range(hyperp.shape[0]), range(len(feature_sets)), range(len(thresholds))))
        for c,f,t, result in results:
            all_models.loc[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t, :] = result[0].iloc[0]
            all_outputs[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t, :, :] = result[1]
            all_cal_outputs[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t, :, :] = result[2]
            X_dev[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t, :, :] = result[3]
            y_true[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t, :] = result[4]
            IDs_dev[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t, :] = result[5]
    else:
        for c,f,t in itertools.product(range(hyperp.shape[0]), range(len(feature_sets)), range(len(thresholds))):
            _,_,_, result = process_combination(c, f, t,problem_type,calmethod,calparams)
            all_models.loc[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t, :] = result[0].iloc[0]
            all_outputs[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t, :, :] = result[1]
            all_cal_outputs[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t, :, :] = result[2]
            X_dev[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t, :, :] = result[3]
            y_true[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t, :] = result[4]
            IDs_dev[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t, :] = result[5]
    
    return all_models, all_outputs, all_cal_outputs, X_dev[0], y_true[0], IDs_dev[0]

def test_model(model_class,params,scaler,imputer, X_dev, y_dev, X_test,problem_type='clf'):
    
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
    problem_type : str, optional
        Specifies 'clf' for classification or 'reg' for regression tasks (default is 'clf').

    Returns
    -------
    outputs : np.array
        Array of model outputs for each test sample.
    """

    if not isinstance(X_dev, pd.DataFrame):
        X_dev = pd.DataFrame(X_dev)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    model = Model(model_class(**params),scaler,imputer)
    model.train(X_dev, y_dev)

    outputs = model.eval(X_test, problem_type)

    return outputs

def compute_metrics(j, model_index, r, outputs, y_dev, IDs, metrics_names, n_boot, problem_type, cmatrix=None, priors=None, threshold=None, bayesian=False):
    # Calculate the metrics using the bootstrap method
    if outputs.ndim == 4 and problem_type == 'clf':
        outputs = outputs[:,np.newaxis,:,:,:]
    elif outputs.ndim == 3 and problem_type == 'reg':
        outputs = outputs[:,np.newaxis,:,:]
    if outputs.shape[-1] > 2:
        metrics_names = list(set(metrics_names) - set(['roc_auc','f1','recall','precision']))
      
    results, sorted_IDs = get_metrics_bootstrap(outputs[j,model_index,r], y_dev[j,r], IDs[j,r],metrics_names, n_boot=n_boot, cmatrix=cmatrix,priors=priors,threshold=threshold,problem_type=problem_type,bayesian=bayesian)

    metrics_result = {}
    for metric in metrics_names:
        metrics_result[metric] = results[metric]
    return j,model_index,r,metrics_result,sorted_IDs

def get_metrics_bootstrap(samples, targets, IDs, metrics_names, n_boot=2000,cmatrix=None,priors=None,threshold=None,problem_type='clf',bayesian=False):
    
    all_metrics = dict((metric,np.zeros(n_boot)) for metric in metrics_names)
   
    for metric in metrics_names:
        if bayesian:
            weights = np.random.dirichlet(np.ones(samples.shape[0]))
        else:
            weights = None
        #Sort IDs and keep indices to adjust samples and targets' order
        indices_ = np.argsort(IDs)
        samples = samples[indices_]
        targets = targets[indices_]
        sorted_IDs = [IDs[indices_]]
        
        for b in range(n_boot):
            np.random.seed(b)
            indices = np.random.choice(indices_, len(indices_), replace=True)
            sorted_IDs.append(sorted_IDs[0][indices])

            while len(np.unique(targets[indices])) == 1:
                np.random.seed(b)
                indices = np.random.choice(indices_, len(indices_), replace=True)
            if problem_type == 'clf':
                metric_value, y_pred = get_metrics_clf(samples[indices], targets[indices], [metric], cmatrix,priors,threshold,weights)
            else:
                metric_value = get_metrics_reg(samples[indices], targets[indices], [metric])
            if (len(metric_value) == 0) or (not isinstance(metric_value[metric],float)):
                
                continue

            all_metrics[metric][b] = metric_value[metric]
        
    return all_metrics, sorted_IDs

def nestedCVT(model_class,scaler,imputer,X,y,n_iter,iterator_outer,iterator_inner,random_seeds_outer,hyperp_space,IDs,init_points=5,scoring='roc_auc_score',problem_type='clf',priors=None,threshold=None,feature_selection=True,parallel=True):
    
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

    model_rfecv = model_class()

    if hasattr(model_rfecv,'kernel'):
        model_rfecv.kernel = 'linear'
    if hasattr(model_rfecv,'probability'):
        model_rfecv.probability = True
    if hasattr(model_rfecv,'random_state') and problem_type == 'clf':
        model_rfecv.random_state = int(42)

    def parallel_train(r,random_seed):
        models_r = pd.DataFrame(columns=['random_seed','fold','threshold',scoring] + list(hyperp_space.keys()) + list(features))

        outputs_best_r = np.empty((X.shape[0],2)) if problem_type == 'clf' else np.empty((X.shape[0]))

        y_true_r = np.empty(X.shape[0])

        y_pred_best_r = np.empty((X.shape[0]))

        IDs_val_r = np.empty(X.shape[0],dtype=object)

        iterator_outer.random_state = random_seed
        
        for k,(train_index_out,test_index_out) in enumerate(iterator_outer.split(X,y)): 
            X_dev, X_test = X.loc[train_index_out], X.loc[test_index_out]
            y_dev, y_test = y[train_index_out], y[test_index_out]
            
            X_dev = X_dev.reset_index(drop=True)
            y_dev = y_dev.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            y_true_r[test_index_out] = y_test
            #ID_dev, ID_test = IDs[train_index_out], IDs[test_index_out]
            IDs_val_r[test_index_out] = IDs[test_index_out]

            scaler_ = scaler().fit(X_dev)
            imputer_ = imputer().fit(X_dev)

            X_dev = pd.DataFrame(columns=X.columns,data=imputer_.transform(pd.DataFrame(columns=X_dev.columns,data=scaler_.transform(X_dev))))
            X_test = pd.DataFrame(columns=X.columns,data=imputer_.transform(pd.DataFrame(columns=X_test.columns,data=scaler_.transform(X_test))))
            print(f'Random seed {r+1}, Fold {k+1}')

            best_features = rfe(Model(model_class(),scaler,imputer),X_dev,y_dev,iterator_inner,scoring,problem_type,priors,threshold) if feature_selection else X.columns
            
            best_params, best_score = tuning(model_class,scaler,imputer,X_dev[best_features],y_dev,hyperp_space,iterator_inner,init_points=init_points,n_iter=n_iter,scoring=scoring,problem_type=problem_type,priors=priors,threshold=threshold)
            
            if 'n_estimators' in best_params.keys():
                best_params['n_estimators'] = int(best_params['n_estimators'])
            elif 'n_neighbors' in best_params.keys():
                best_params['n_neighbors'] = int(best_params['n_neighbors'])
            elif 'max_depth' in best_params.keys():
                best_params['max_depth'] = int(best_params['max_depth'])
            if 'gpu_id' in best_params.keys():
                best_params['gpu_id'] = None
            
            if hasattr(model_class(),'random_state') and model_class != SVR:
                best_params['random_state'] = int(42)
            if hasattr(model_class(),'probability') and model_class != SVR:
                best_params['probability'] = True

            append_dict = {'random_seed':random_seed,'fold':k,'threshold':threshold,scoring:best_score}
            append_dict.update(best_params)
            append_dict.update({feature:1 if feature in best_features else 0 for feature in X_dev.columns}) 

            models_r.loc[len(models_r.index),:] = append_dict

            model = Model(model_class(**best_params),scaler,imputer)
            model.train(X_dev[best_features],y_dev)
            
            if problem_type == 'clf':
                outputs_best_ = model.eval(X_test[best_features],problem_type)
                if threshold is not None:
                    y_pred_best_ = [1 if x > threshold else 0 for x in outputs_best_r[test_index_out][:,1]]
                else:
                    y_pred_best_= bayes_decisions(scores=outputs_best_r[test_index_out],priors=priors,score_type='log_posteriors')[0]
                
                y_pred_best_r[test_index_out] = y_pred_best_

            else:
                outputs_best_ = model.eval(X_test[best_features],problem_type)
                y_pred_best_r[test_index_out] = outputs_best_
            outputs_best_r[test_index_out] = outputs_best_

        return models_r,outputs_best_r,y_true_r,y_pred_best_r,IDs_val_r
    
    results = Parallel(n_jobs=-1 if parallel else 1)(delayed(parallel_train)(r,random_seed_train) for r,random_seed_train in enumerate(random_seeds_outer))
    all_models = pd.concat([result[0] for result in results],ignore_index=True,axis=0)
    outputs_best = np.concatenate(([np.expand_dims(result[1],axis=0) for result in results]),axis=0)
    y_true = np.concatenate(([np.expand_dims(result[2],axis=0) for result in results]),axis=0)
    y_pred_best = np.concatenate(([np.expand_dims(result[3],axis=0) for result in results]),axis=0)
    IDs_val = np.concatenate(([np.expand_dims(result[4],axis=0) for result in results]),axis=0)

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
    ascending = any(x in scoring for x in ['error', 'loss'])
    best_score = np.inf if ascending else -np.inf
    best_features = features.copy()

    while len(features) > 1:
        scorings = {}  # Dictionary to hold scores for each feature removal
        
        for feature in features:
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
                    outputs[val_index] = outputs_
                    if threshold is not None:
                        y_pred[val_index] = [1 if x > threshold else 0 for x in outputs_[:,1]]
                    else:
                        y_pred[val_index] = bayes_decisions(scores=outputs_,costs=cmatrix,priors=priors,score_type='log_posteriors')[0]
                else:
                    outputs[val_index] = model.eval(X_val,problem_type)
                    y_pred[val_index] = outputs[val_index]
                y_true[val_index] = y_val
            # Choose the appropriate scoring function
            #Replace nans in outpts with -1e6
            
            
            if scoring == 'roc_auc_score':
                scorings[feature] = eval(scoring)(y_true, outputs[:, 1])
            elif scoring == 'norm_expected_cost':
                scorings[feature] = -average_cost(targets=np.array(y_true,dtype=int),decisions=np.array(y_pred,dtype=int),costs=cmatrix,priors=priors,adjusted=True)
            elif scoring == 'norm_cross_entropy':
                scorings[feature] = -LogLoss(log_probs=torch.tensor(outputs),labels=torch.tensor(np.array(y_true),dtype=torch.int),priors=torch.tensor(priors)).detach().numpy() if priors is not None else -LogLoss(log_probs=torch.tensor(outputs),labels=torch.tensor(np.array(y_true),dtype=torch.int)).detach().numpy()
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
            print(f'Removing feature: {feature}. New best score: {best_score}')
        else:
            print('No improvement found. Stopping RFE.')
            # Stop if no improvement
            break

    return best_features

def new_best(old,new,greater=True):
    if greater:
        return new > old
    else:
        return new < old

def tuning(model,scaler,imputer,X,y,hyperp_space,iterator,init_points=5,n_iter=50,scoring='roc_auc_score',problem_type='clf',cmatrix=None,priors=None,threshold=None,random_state=42):
    
    def objective(**params):
        return scoring_bo(params, model, scaler, imputer, X, y, iterator, scoring, problem_type, 
                          cmatrix, priors, threshold)
    
    search = BayesianOptimization(f=objective,pbounds=hyperp_space,verbose=2,random_state=random_state)
    #search = BayesSearchCV(model(),hyperp_space,scoring=lambda params,X,y: scoring_bo(params,model,scaler,imputer,X,y,iterator,scoring,problem_type,cmatrix,priors,threshold),n_iter=50,cv=None,random_state=42,verbose=2)
    search.maximize(init_points=init_points,n_iter=n_iter)
    #search.fit(X,y)
    best_params = search.max['params']
    int_params = ['n_estimators', 'n_neighbors', 'max_depth']
    for param in int_params:
        if param in best_params:
            best_params[param] = int(best_params[param])
    return best_params, search.max['target']

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
    
    if hasattr(model_class,'probability') and problem_type == 'clf':
        params['probability'] = True
        
    y_true = np.empty(X.shape[0])
    y_pred = np.empty(X.shape[0])
    outputs = np.empty((X.shape[0],2)) if problem_type == 'clf' else np.empty(X.shape[0])
    
    for train_index, test_index in iterator.split(X,y):
        model = Model(model_class(**params),scaler,imputer)
        model.train(X.loc[train_index],y[train_index])
        outputs[test_index] = model.eval(X.loc[test_index],problem_type)

        outputs[test_index] = np.clip(outputs[test_index],-1e6,1e6)
        if problem_type == 'clf':
            if threshold is not None:
                y_pred[test_index] = [1 if x > threshold else 0 for x in outputs[test_index,1]]
            else:
                y_pred[test_index] = bayes_decisions(scores=outputs[test_index],costs=cmatrix,priors=priors,score_type='log_posteriors')[0]
        else:
            y_pred[test_index] = outputs[test_index]
        y_true[test_index] = y[test_index]
    
    scoring_func = getattr(metrics, scoring, None)

    if 'error' in scoring:
        return -scoring_func(y_true, outputs)
    elif scoring == 'norm_expected_cost':
        return -average_cost(targets=np.array(y_true,dtype=int),decisions=np.array(y_pred,dtype=int),costs=cmatrix,priors=priors,adjusted=True)
    elif scoring == 'norm_cross_entropy':
        return -LogLoss(log_probs=torch.tensor(outputs),labels=torch.tensor(np.array(y_true),dtype=torch.int),priors=torch.tensor(priors)).detach().numpy() if priors is not None else -LogLoss(log_probs=torch.tensor(outputs),labels=torch.tensor(np.array(y_true),dtype=torch.int)).detach().numpy()
    elif scoring == 'roc_auc_score':
        return scoring_func(y_true, outputs[:,1])
    else:
        return scoring_func(y_true, y_pred)

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