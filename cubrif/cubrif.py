# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import cubrifc

class cubrif:
    tmp_preddata = "tmp_brif_preddata.txt"
    max_factor_levels = 30
    max_integer_classes = 20
    n_numeric_cuts = 31
    n_integer_cuts = 31
    ps = 0
    max_depth = 20
    min_node_size = 1
    ntrees = 200
    nthreads = 4
    blocksize = 128
    GPU = 0
    vote_method = 1
    seed = 2022
    na_numeric = 0
    na_integer = -9999
    na_factor = 'NA'
    _model = None
    _problem_type = 'classification'
    def __init__(self, param = dict()):
        if "tmp_preddata" in param:
            self.tmp_preddata = param['tmp_preddata']
        if "max_factor_levels" in param:
            self.max_factor_levels = param['max_factor_levels']
        if "max_integer_classes" in param:
            self.max_integer_classes = param['max_integer_classes']
        if "n_numeric_cuts" in param:
            self.n_numeric_cuts = param['n_numeric_cuts']
        if "n_integer_cuts" in param:
            self.n_integer_cuts = param['n_integer_cuts']        
        if "ps" in param:
            self.ps = param['ps']
        if "max_depth" in param:
            self.max_depth = param['max_depth']   
        if "min_node_size" in param:
            self.min_node_size = param['min_node_size']
        if "ntrees" in param:
            self.ntrees = param['ntrees']   
        if "nthreads" in param:
            self.nthreads = param['nthreads']
        if "blocksize" in param:
            self.blocksize = param['blocksize']
        if "GPU" in param:
            self.GPU = param['GPU']
        if "vote_method" in param:
            self.vote_method = param['vote_method']    
        if "seed" in param:
            self.seed = param['seed']
    def set_param(self, param = dict()):
        if "tmp_preddata" in param:
            self.tmp_preddata = param['tmp_preddata']
        if "max_factor_levels" in param:
            self.max_factor_levels = param['max_factor_levels']
        if "max_integer_classes" in param:
            self.max_integer_classes = param['max_integer_classes']
        if "n_numeric_cuts" in param:
            self.n_numeric_cuts = param['n_numeric_cuts']
        if "n_integer_cuts" in param:
            self.n_integer_cuts = param['n_integer_cuts']        
        if "ps" in param:
            self.ps = param['ps']
        if "max_depth" in param:
            self.max_depth = param['max_depth']   
        if "min_node_size" in param:
            self.min_node_size = param['min_node_size']
        if "ntrees" in param:
            self.ntrees = param['ntrees']   
        if "nthreads" in param:
            self.nthreads = param['nthreads']
        if "blocksize" in param:
            self.blocksize = param['blocksize']
        if "GPU" in param:
            self.GPU = param['GPU']
        if "vote_method" in param:
            self.vote_method = param['vote_method']         
        if "seed" in param:
            self.seed = param['seed']  
        if "na_numeric" in param:
            self.na_numeric = param['na_numeric']   
        if "na_integer" in param:
            self.na_integer = param['na_integer']
        if "na_factor" in param:
            self.na_factor = param['na_factor'] 
            
    def get_param(self):
        return(dict({
                    "tmp_preddata":self.tmp_preddata,
                    "max_factor_levels":self.max_factor_levels,
                    "max_integer_classes":self.max_integer_classes,
                    "n_numeric_cuts":self.n_numeric_cuts,
                    "n_integer_cuts":self.n_integer_cuts,
                    "ps":self.ps,
                    "max_depth":self.max_depth,
                    "min_node_size":self.min_node_size,
                    "ntrees":self.ntrees,
                    "nthreads":self.nthreads,
                    "blocksize":self.blocksize,
                    "GPU":self.GPU,
                    "vote_method":self.vote_method,
                    "seed":self.seed,
                    "na_numeric":self.na_numeric,
                    "na_integer":self.na_integer,
                    "na_factor":self.na_factor
                    }))
    
    def fit(self, df, target_col):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas data frame.")
        n = df.shape[0]
        if n < 16:
            raise ValueError("Too few training data points. Model cannot run.")
        p = df.shape[1] - 1
        if p < 1:
            raise ValueError("Too few predictors.")
        target_col_num = 0
        colnames = list(df.columns.values)
        if isinstance(target_col, str):
            target_col_num = colnames.index(target_col)
        elif isinstance(target_col, int):
            target_col_num = target_col
        else:
            raise ValueError('target_col must be str or int.')
        if target_col_num < 0 or target_col_num > p:
            raise ValueError('target_col_num out of bound.')
        
        target_col_name = colnames[target_col_num]
        del colnames[target_col_num]
        colnames = [target_col_name] + colnames
        df = df[colnames]
        coltypes = df.dtypes
        vartypes = ["" for j in range(p+1)]
        num_ind = []
        int_ind = []
        fac_ind = []
        
        for j in range(p+1):
            if coltypes[j] == np.float64:
                vartypes[j] = "numeric"
                num_ind = num_ind + [j]
            elif coltypes[j] == np.int64:
                vartypes[j] = "integer"
                int_ind = int_ind + [j]
            elif coltypes[j] == object:
                vartypes[j] = "factor"
                fac_ind = fac_ind + [j]
            else:
                raise ValueError("Column types must be np.float64, np.int64 or object")
        
        # check unique levels
        for j in fac_ind:
            nu = len(df.iloc[:,j].unique())
            if(nu > self.max_factor_levels):
                raise ValueError("Variable " + str(colnames[j]) + ' has ' + str(nu) + " unique levels. If this is intended, adjust max_factor_levels and re-run.")

        if(vartypes[0] == 'numeric'):
            self._problem_type = "regression"
        else:
            self._problem_type = "classification"
            
        n_discard_bits = 0
        if n % 32 != 0:
            n_discard_bits = 32 - n % 32
        if n_discard_bits > 0:
            if n < n_discard_bits:
                raise ValueError("Too few training data points.")
            pad = df.sample(n_discard_bits, axis = 0)
            df = pd.concat([df,pad])
            n = df.shape[0]
            
        df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows
        df.sort_values(by=target_col_name, inplace=True, ignore_index = True)
        indxmat = np.reshape(np.arange(n), (32, int(n/32)))
        indxmat = indxmat.transpose()
        indxmat = np.reshape(indxmat, (n,1))
        df = df.reindex(indxmat.flatten())    

        self._model = cubrifc.fit(colnames, vartypes, 
                  num_ind, df.iloc[:,num_ind].to_numpy(na_value = self.na_numeric),
                  int_ind, df.iloc[:,int_ind].to_numpy(na_value = self.na_integer),
                  fac_ind, df.iloc[:,fac_ind].to_numpy(na_value = self.na_factor),
                  n, p, self.get_param())
    
    
    def predict(self, test_df, type = 'score'):
        if self._model == None:
            raise ValueError('Model is not available.')
        n = test_df.shape[0]
        p = test_df.shape[1]
        test_colnames = list(test_df.columns.values)
        test_coltypes = test_df.dtypes
        test_vartypes = ["" for j in range(p)]
        test_num_ind = []
        test_int_ind = []
        test_fac_ind = []
        for j in range(p):
            if test_coltypes[j] == np.float64:
                test_vartypes[j] = "numeric"
                test_num_ind = test_num_ind + [j+1]
            elif test_coltypes[j] == np.int64:
                test_vartypes[j] = "integer"
                test_int_ind = test_int_ind + [j+1]
            elif test_coltypes[j] == object:
                test_vartypes[j] = "factor"
                test_fac_ind = test_fac_ind + [j+1]
            else:
                raise ValueError("Column types must be np.float64, np.int64 or object")
        cubrifc.predict(self._model, self.tmp_preddata, test_colnames, test_vartypes,
                             test_num_ind, test_df.iloc[:,[test_num_ind[i] - 1 for i in range(len(test_num_ind))]].to_numpy(na_value = self.na_numeric),
                             test_int_ind, test_df.iloc[:,[test_int_ind[i] - 1 for i in range(len(test_int_ind))]].to_numpy(na_value = self.na_integer),
                             test_fac_ind, test_df.iloc[:,[test_fac_ind[i] - 1 for i in range(len(test_fac_ind))]].to_numpy(na_value = self.na_factor),
                             n, p, self.get_param())

        # read the scores
        scores = pd.read_csv(self.tmp_preddata, sep=' ')
        if(type == 'score'):
            if self._problem_type == 'classification':
                return(scores)
            else:
                head_vec = scores.columns.to_numpy().astype(np.float64)
                pred_mat = scores.to_numpy()
                pred_val = np.dot(pred_mat, head_vec)
                return(list(pred_val))
        else:
            if self._problem_type == 'classification':  
                return(list(scores.idxmax(axis=1)))
            else:
                head_vec = scores.columns.to_numpy().astype(np.float64)
                pred_mat = scores.to_numpy()
                pred_val = np.dot(pred_mat, head_vec)
                return(list(pred_val))                
            
            
    
            
        
        