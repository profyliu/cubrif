#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "brif.h"

static void del_Model(PyObject *obj){
    delete_model((rf_model_t *)PyCapsule_GetPointer(obj,"rf_model_t"));
}

static void delete_data_py(data_frame_t *df){
    if(df == NULL) return;
    for(int j = 0; j <= df->p; j++){
        if(df->var_types[j] == 'f'){
            if(df->data[j] != NULL)
                delete_factor((factor_t*)df->data[j]);
        }
        if(df->var_labels[j] != NULL) free(df->var_labels[j]);
    }
    free(df->var_types);
    free(df->var_labels);
    free(df->data);
    free(df);
}

static PyObject *py_fit(PyObject *self, PyObject *args){
    PyObject *py_var_labels = Py_None;
    PyObject *py_var_types = Py_None;
    PyObject *num_ind = Py_None;
    PyObject *num_obj = Py_None;
    PyObject *int_ind = Py_None;
    PyObject *int_obj = Py_None;
    PyObject *fac_ind = Py_None;
    PyObject *fac_obj = Py_None;
    int n = 0;
    int p = 0;
    PyObject *param = Py_None;
    if(!PyArg_ParseTuple(args, "OOOOOOOOiiO", &py_var_labels, &py_var_types, &num_ind, &num_obj, &int_ind, &int_obj, &fac_ind, &fac_obj, &n, &p, &param)){
        return Py_None;
    }
    // check input validity
    if(!PyDict_Check(param)){
        PyErr_SetString(PyExc_ValueError, "param must be a dictionary.");
        return Py_None;
    }
    if(n < 16){
        PyErr_SetString(PyExc_ValueError, "Too few data points.");
        return Py_None;
    }
    if(p <= 0){
        PyErr_SetString(PyExc_ValueError, "Too few predictors.");
        return Py_None;
    }
    if(!PyList_Check(py_var_labels)){
        PyErr_SetString(PyExc_ValueError, "py_var_labels must be a list.");
        return Py_None;        
    }
    if(!PyList_Check(py_var_types)){
        PyErr_SetString(PyExc_ValueError, "py_var_types must be a list.");
        return Py_None;        
    }
    Py_ssize_t n_num_vars = PyList_Size(num_ind);
    Py_ssize_t n_int_vars = PyList_Size(int_ind);
    Py_ssize_t n_fac_vars = PyList_Size(fac_ind);
    for(int k = 0; k < n_num_vars; k++){
        PyObject *this_value = PyList_GetItem(num_ind, k);
        int j = (int) PyLong_AsLong(this_value);  
        if(j > p){
            PyErr_SetString(PyExc_ValueError, "Numeric column index out of bound.");
            return Py_None;              
        }
    }
    for(int k = 0; k < n_int_vars; k++){
        PyObject *this_value = PyList_GetItem(int_ind, k);
        int j = (int) PyLong_AsLong(this_value);  
        if(j > p){
            PyErr_SetString(PyExc_ValueError, "Integer index out of bound.");
            return Py_None;              
        }
    }
    for(int k = 0; k < n_fac_vars; k++){
        PyObject *this_value = PyList_GetItem(fac_ind, k);
        int j = (int) PyLong_AsLong(this_value);  
        if(j > p){
            PyErr_SetString(PyExc_ValueError, "Factor index out of bound.");
            return Py_None;              
        }
    }

    Py_buffer num_view;
    Py_buffer int_view;
    Py_buffer fac_view;
    if(PyObject_GetBuffer(num_obj, &num_view, PyBUF_ANY_CONTIGUOUS)){
        PyErr_SetString(PyExc_ValueError, "num_obj is not a buffer.");
        return Py_None;          
    }
    if(PyObject_GetBuffer(int_obj, &int_view, PyBUF_ANY_CONTIGUOUS)){
        PyBuffer_Release(&num_view);
        PyErr_SetString(PyExc_ValueError, "int_obj is not a buffer.");
        return Py_None;          
    }
    if(PyObject_GetBuffer(fac_obj, &fac_view, PyBUF_ANY_CONTIGUOUS)){
        PyBuffer_Release(&num_view);
        PyBuffer_Release(&int_view);
        PyErr_SetString(PyExc_ValueError, "fac_obj is not a buffer.");
        return Py_None;          
    }
    if(num_view.shape[0] != n || num_view.shape[1] != n_num_vars
        || int_view.shape[0] != n || int_view.shape[1] != n_int_vars
        || fac_view.shape[0] != n || fac_view.shape[1] != n_fac_vars){
        PyBuffer_Release(&num_view);
        PyBuffer_Release(&int_view);
        PyBuffer_Release(&fac_view);
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch.");
        return Py_None;         
    }

    int max_integer_classes = 20;
    int n_numeric_cuts = 31;
    int n_integer_cuts = 31;
    int ps = 0;
    int max_depth = 20;
    int min_node_size = 1;
    int ntrees = 200;
    int nthreads = 2;
    int blocksize = 128;
    int ubthresh = 640;
    int GPU = 0;  // 0: CPU; 1: CPU + GPU
    //int bagging_method = 0;
    //double bagging_proportion = 0.9;
    //int split_search = 4;
    //int search_radius = 5;
    int seed = 0;

    // parse parameters
    if(PyDict_Check(param)){
        PyObject *this_value = Py_None;
        if((this_value = PyDict_GetItemString(param, "max_integer_classes")) != NULL){
            max_integer_classes = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "n_numeric_cuts")) != NULL){
            n_numeric_cuts = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "n_integer_cuts")) != NULL){
            n_integer_cuts = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "ps")) != NULL){
            ps = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "max_depth")) != NULL){
            max_depth = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "min_node_size")) != NULL){
            min_node_size = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "ntrees")) != NULL){
            ntrees = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "blocksize")) != NULL){
            blocksize = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "nthreads")) != NULL){
            nthreads = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "GPU")) != NULL){
            GPU = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "n_lb_GPU")) != NULL){
            ubthresh = (int) PyLong_AsLong(this_value) / sizeof(bitblock_t);
        }
        /*
        if((this_value = PyDict_GetItemString(param, "bagging_method")) != NULL){
            bagging_method = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "bagging_proportion")) != NULL){
            bagging_proportion = (double) PyFloat_AsDouble(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "split_search")) != NULL){
            split_search = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "search_radius")) != NULL){
            search_radius = (int) PyLong_AsLong(this_value);
        }
        */
        if((this_value = PyDict_GetItemString(param, "seed")) != NULL){
            seed = (int) PyLong_AsLong(this_value);
        }
    }
    
    // make var_labels and var_types
    char **var_labels = (char **)malloc((p+1)*sizeof(char*));  // var name at most 50 characters
    char *var_types = (char *)malloc((p+1)*sizeof(char));  // integer, numeric, factor
    for(int j = 0; j <= p; j++){
        var_labels[j] = (char*)malloc(MAX_VAR_NAME_LEN*sizeof(char));
        PyObject* v_pyStr = PyUnicode_AsEncodedString(PyList_GetItem(py_var_labels, j), "utf-8", "Error ~");
        strncpy(var_labels[j], PyBytes_AsString(v_pyStr), MAX_VAR_NAME_LEN-1);
        v_pyStr = PyUnicode_AsEncodedString(PyList_GetItem(py_var_types, j), "utf-8", "Error ~");
        if(!strcmp(PyBytes_AsString(v_pyStr), "numeric")){
            var_types[j] = 'n';
        } else if(!strcmp(PyBytes_AsString(v_pyStr), "integer")){
            var_types[j] = 'i';
        } else {
            var_types[j] = 'f';
        }
    }

    // construct training data frame
    void **data = (void**)malloc((p+1)*sizeof(void*));
    for(int j = 0; j <= p; j++){
        data[j] = NULL;  // initialize
    }
    // process numeric columns
    for(int k = 0; k < n_num_vars; k++){
        PyObject *this_value = PyList_GetItem(num_ind, k);
        int j = (int) PyLong_AsLong(this_value);  
        data[j] = (numeric_t *) num_view.buf + k*num_view.shape[0];
    }
    // process integer columns
    for(int k = 0; k < n_int_vars; k++){
        PyObject *this_value = PyList_GetItem(int_ind, k);
        int j = (int) PyLong_AsLong(this_value);  
        data[j] = (integer_t *) int_view.buf + k*int_view.shape[0];
    }
    // process factor columns
    for(int k = 0; k < n_fac_vars; k++){
        PyObject *this_value = PyList_GetItem(fac_ind, k);
        int j = (int) PyLong_AsLong(this_value);  
        data[j] = (factor_t *) create_factor(n);
        PyObject **this_fac = (PyObject **)fac_view.buf + k*fac_view.shape[0];
        PyObject* v_pyStr;
        char *this_name;
        for(int i = 0; i < fac_view.shape[0]; i++){
            v_pyStr = PyUnicode_AsEncodedString(this_fac[i], "utf-8", "Error ~");
            this_name = PyBytes_AsString(v_pyStr);
            add_element((factor_t *)data[j], i, this_name);
        }
    }

    rf_model_t *model = create_empty_model();
    model->p = p;
    model->var_types = (char*)malloc((p+1)*sizeof(char));
    memcpy(model->var_types, var_types, (p+1)*sizeof(char));
    model->var_labels = (char**)malloc((p+1)*sizeof(char*));
    for(int j = 0; j <= p; j++){
        model->var_labels[j] = (char*)malloc(MAX_VAR_NAME_LEN*sizeof(char));
        strncpy(model->var_labels[j], var_labels[j], MAX_VAR_NAME_LEN-1);
    }

    data_frame_t *train_df = (data_frame_t*)malloc(sizeof(data_frame_t));
    train_df->n = n;
    train_df->p = p;
    train_df->var_types = var_types;  
    train_df->var_labels = var_labels;  
    train_df->data = data;

    make_cuts(train_df, &model, n_numeric_cuts, n_integer_cuts); 
    bx_info_t *bx_train = make_bx(train_df, &model, nthreads);
    ycode_t *yc_train = make_yc(train_df, &model, max_integer_classes, nthreads);
    delete_data_py(train_df);  
    PyBuffer_Release(&num_view);
    PyBuffer_Release(&int_view);
    PyBuffer_Release(&fac_view);   
    if(ps <= 0){
        ps = (int)(round(sqrt(model->p)));
    }

    if(GPU == 0)
        build_forest(bx_train, yc_train, &model, ps, max_depth, min_node_size, ntrees, nthreads, seed);
    else if(GPU == 1)
        build_forest_cuda(bx_train, yc_train, &model, ps, max_depth, min_node_size, ntrees, nthreads, blocksize, seed);
    else
        build_forest_hybrid(bx_train, yc_train, &model, ps, max_depth, min_node_size, ntrees, nthreads, blocksize, ubthresh, seed);
    flatten_model(&model, nthreads);
    delete_bx(bx_train, model);
    delete_yc(yc_train);
    return PyCapsule_New(model, "rf_model_t", del_Model);
}

static PyObject *py_predict(PyObject *self, PyObject *args){
    PyObject *scorefile_obj = Py_None, *scorefile_bytes;
    PyObject *model_capsule;
    char * scorefile;

    PyObject *py_var_labels = Py_None;
    PyObject *py_var_types = Py_None;
    PyObject *num_ind = Py_None;
    PyObject *num_obj = Py_None;
    PyObject *int_ind = Py_None;
    PyObject *int_obj = Py_None;
    PyObject *fac_ind = Py_None;
    PyObject *fac_obj = Py_None;
    int n = 0;
    int p = 0;
    PyObject *param = Py_None;
    if(!PyArg_ParseTuple(args, "OOOOOOOOOOiiO", &model_capsule, &scorefile_obj, &py_var_labels, &py_var_types, &num_ind, &num_obj, &int_ind, &int_obj, &fac_ind, &fac_obj, &n, &p, &param)){
        return Py_None;
    }

    rf_model_t *model = (rf_model_t *) PyCapsule_GetPointer(model_capsule, "rf_model_t");
    if(model == NULL){
        printf("model is NULL\n");
        return Py_None;
    }

    // check input validity
    if(!PyDict_Check(param)){
        PyErr_SetString(PyExc_ValueError, "param must be a dictionary.");
        return Py_None;
    }
    if(p <= 0){
        PyErr_SetString(PyExc_ValueError, "Too few predictors.");
        return Py_None;
    }
    if(!PyList_Check(py_var_labels)){
        PyErr_SetString(PyExc_ValueError, "py_var_labels must be a list.");
        return Py_None;        
    }
    if(!PyList_Check(py_var_types)){
        PyErr_SetString(PyExc_ValueError, "py_var_types must be a list.");
        return Py_None;        
    }
    Py_ssize_t n_num_vars = PyList_Size(num_ind);
    Py_ssize_t n_int_vars = PyList_Size(int_ind);
    Py_ssize_t n_fac_vars = PyList_Size(fac_ind);
    for(int k = 0; k < n_num_vars; k++){
        PyObject *this_value = PyList_GetItem(num_ind, k);
        int j = (int) PyLong_AsLong(this_value);  
        if(j > p){
            PyErr_SetString(PyExc_ValueError, "Numeric column index out of bound.");
            return Py_None;              
        }
    }
    for(int k = 0; k < n_int_vars; k++){
        PyObject *this_value = PyList_GetItem(int_ind, k);
        int j = (int) PyLong_AsLong(this_value);  
        if(j > p){
            PyErr_SetString(PyExc_ValueError, "Integer index out of bound.");
            return Py_None;              
        }
    }
    for(int k = 0; k < n_fac_vars; k++){
        PyObject *this_value = PyList_GetItem(fac_ind, k);
        int j = (int) PyLong_AsLong(this_value);  
        if(j > p){
            PyErr_SetString(PyExc_ValueError, "Factor index out of bound.");
            return Py_None;              
        }
    }

    Py_buffer num_view;
    Py_buffer int_view;
    Py_buffer fac_view;
    if(PyObject_GetBuffer(num_obj, &num_view, PyBUF_ANY_CONTIGUOUS)){
        PyErr_SetString(PyExc_ValueError, "num_obj is not a buffer.");
        return Py_None;          
    }
    if(PyObject_GetBuffer(int_obj, &int_view, PyBUF_ANY_CONTIGUOUS)){
        PyBuffer_Release(&num_view);
        PyErr_SetString(PyExc_ValueError, "int_obj is not a buffer.");
        return Py_None;          
    }
    if(PyObject_GetBuffer(fac_obj, &fac_view, PyBUF_ANY_CONTIGUOUS)){
        PyBuffer_Release(&num_view);
        PyBuffer_Release(&int_view);
        PyErr_SetString(PyExc_ValueError, "fac_obj is not a buffer.");
        return Py_None;          
    }
    if(num_view.shape[0] != n || num_view.shape[1] != n_num_vars
        || int_view.shape[0] != n || int_view.shape[1] != n_int_vars
        || fac_view.shape[0] != n || fac_view.shape[1] != n_fac_vars){
        PyBuffer_Release(&num_view);
        PyBuffer_Release(&int_view);
        PyBuffer_Release(&fac_view);
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch.");
        return Py_None;         
    }

    int vote_method = 1;
    int nthreads = 2;
    // parse parameters
    if(PyDict_Check(param)){
        PyObject *this_value = Py_None;
        if((this_value = PyDict_GetItemString(param, "vote_method")) != NULL){
            vote_method = (int) PyLong_AsLong(this_value);
        }
        if((this_value = PyDict_GetItemString(param, "nthreads")) != NULL){
            nthreads = (int) PyLong_AsLong(this_value);
        }
    }

    // make var_labels and var_types
    char **var_labels = (char **)malloc((p+1)*sizeof(char*));  // var name at most 50 characters
    char *var_types = (char *)malloc((p+1)*sizeof(char));  // integer, numeric, factor
    var_types[0] = '0';
    var_labels[0] = NULL;
    for(int j = 1; j <= p; j++){
        var_labels[j] = (char*)malloc(MAX_VAR_NAME_LEN*sizeof(char));
        PyObject* v_pyStr = PyUnicode_AsEncodedString(PyList_GetItem(py_var_labels, j-1), "utf-8", "Error ~");
        strncpy(var_labels[j], PyBytes_AsString(v_pyStr), MAX_VAR_NAME_LEN-1);
        v_pyStr = PyUnicode_AsEncodedString(PyList_GetItem(py_var_types, j-1), "utf-8", "Error ~");
        if(!strcmp(PyBytes_AsString(v_pyStr), "numeric")){
            var_types[j] = 'n';
        } else if(!strcmp(PyBytes_AsString(v_pyStr), "integer")){
            var_types[j] = 'i';
        } else {
            var_types[j] = 'f';
        }
    }

    // check consistency between model and new data
    int data_ok = 1;
    for(int j = 1; j <= p; j++){
        if(strcmp(model->var_labels[j], var_labels[j])){
            data_ok = 0;
            break;
        }
        if(model->var_types[j] != var_types[j]){
            data_ok = 0;
            break;
        }
    }
    if(!data_ok){
        // clean up and quit
        PyBuffer_Release(&num_view);
        PyBuffer_Release(&int_view);
        PyBuffer_Release(&fac_view);
        free(var_types);
        for(int j = 1; j <= p; j++){
            free(var_labels[j]);
        }
        free(var_labels);
        PyErr_SetString(PyExc_ValueError, "Column name or type mismatch between model and newdata.");
        return Py_None;  
    }
    
    // construct training data frame
    void **data = (void**)malloc((p+1)*sizeof(void*));
    for(int j = 0; j <= p; j++){
        data[j] = NULL;  // initialize
    }
    // process numeric columns
    for(int k = 0; k < n_num_vars; k++){
        PyObject *this_value = PyList_GetItem(num_ind, k);
        int j = (int) PyLong_AsLong(this_value);  
        data[j] = (numeric_t *) num_view.buf + k*num_view.shape[0];
    }
    // process integer columns
    for(int k = 0; k < n_int_vars; k++){
        PyObject *this_value = PyList_GetItem(int_ind, k);
        int j = (int) PyLong_AsLong(this_value);  
        data[j] = (integer_t *) int_view.buf + k*int_view.shape[0];
    }
    // process factor columns
    for(int k = 0; k < n_fac_vars; k++){
        PyObject *this_value = PyList_GetItem(fac_ind, k);
        int j = (int) PyLong_AsLong(this_value);  
        int which_factor = model->index_in_group[j];
        data[j] = copy_factor(n, model->factor_cuts[which_factor]);
        PyObject **this_fac = (PyObject **)fac_view.buf + k*fac_view.shape[0];
        PyObject* v_pyStr;
        char *this_name;
        for(int i = 0; i < fac_view.shape[0]; i++){
            v_pyStr = PyUnicode_AsEncodedString(this_fac[i], "utf-8", "Error ~");
            this_name = PyBytes_AsString(v_pyStr);
            find_add_element((factor_t*)data[j], i, this_name);
        }
    }

    data_frame_t *test_df = (data_frame_t *)malloc(sizeof(data_frame_t));
    test_df->n = n;
    test_df->p = p;
    test_df->var_types = var_types;  
    test_df->var_labels = var_labels;  
    test_df->data = data;


    if (scorefile_obj != Py_None) {
        if (!PyUnicode_FSConverter(scorefile_obj, &scorefile_bytes))
            return NULL;
        scorefile = PyBytes_AsString(scorefile_bytes);
    } else {
        scorefile_bytes = NULL;
        scorefile = NULL;
    }

    int test_n = test_df->n;
    bx_info_t *bx_test = make_bx(test_df, &model, nthreads);
    delete_data_py(test_df);
    PyBuffer_Release(&num_view);
    PyBuffer_Release(&int_view);
    PyBuffer_Release(&fac_view);

    double **score = (double**)malloc(model->yc->nlevels*sizeof(double*));
    for(int k = 0; k < model->yc->nlevels; k++){
        score[k] = (double*)malloc(test_n*sizeof(double));
        memset(score[k], 0, test_n*sizeof(double));
    }
    predict(model, bx_test, score, vote_method, nthreads);
    delete_bx(bx_test, model);

    FILE *outfile = fopen(scorefile, "w");
    // print header row
    for(int k = 0; k < model->yc->nlevels - 1; k++){
        if(model->yc->type == REGRESSION){
            fprintf(outfile, "%f ", model->yc->yavg[k]);
        } else if(model->yc->type == CLASSIFICATION && model->yc->yvalues_num != NULL){
            fprintf(outfile, "%f ", model->yc->yvalues_num[k]);
        } else if(model->yc->type == CLASSIFICATION && model->yc->yvalues_int != NULL){
            if(model->yc->level_names != NULL){
                fprintf(outfile, "%s ", model->yc->level_names[k]);
            } else {
                fprintf(outfile, "%lld ", model->yc->yvalues_int[k]);
            }
        }
    }
    // print the last one without space
    if(model->yc->type == REGRESSION){
        fprintf(outfile, "%f", model->yc->yavg[model->yc->nlevels - 1]);
    } else if(model->yc->type == CLASSIFICATION && model->yc->yvalues_num != NULL){
        fprintf(outfile, "%f", model->yc->yvalues_num[model->yc->nlevels - 1]);
    } else if(model->yc->type == CLASSIFICATION && model->yc->yvalues_int != NULL){
        if(model->yc->level_names != NULL){
            fprintf(outfile, "%s", model->yc->level_names[model->yc->nlevels - 1]);
        } else {
            fprintf(outfile, "%lld", model->yc->yvalues_int[model->yc->nlevels - 1]);
        }
    }
    fprintf(outfile, "\n");
    
    for(int i = 0; i < test_n; i++){
        for(int k = 0; k < model->yc->nlevels - 1; k++){
            fprintf(outfile, "%f ", score[k][i]); 
        }
        fprintf(outfile, "%f\n", score[model->yc->nlevels - 1][i]);  // no space after last 
    }
    fclose(outfile);
    for(int k = 0; k < model->yc->nlevels; k++){
        free(score[k]);
    }
    free(score);
    return Py_BuildValue("i", test_n);
}


static PyMethodDef brifMethods[] = {
    {"fit", py_fit, METH_VARARGS, "Build and return a brif model"},
    {"predict", py_predict, METH_VARARGS, "Make predictions"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef brifmodule = {
    PyModuleDef_HEAD_INIT,
    "cubrifc",
    "The brif C module",
    -1,
    brifMethods
};

PyMODINIT_FUNC PyInit_cubrifc(void){
    return PyModule_Create(&brifmodule);
}