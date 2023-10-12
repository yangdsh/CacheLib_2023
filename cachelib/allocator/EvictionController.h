#pragma once

#include <iostream>
#include <sstream>
#include <stdint.h>
#include <LightGBM/c_api.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <map>
#include <queue>
#include <shared_mutex>
#include <random>
#include <time.h>
#include <tsl/sparse_map.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <folly/system/ThreadName.h>
#include <sys/syscall.h> 
#include <sys/resource.h>
#include "cachelib/common/concurrentqueue.h"
#include "cachelib/allocator/CacheItem.h"
#include "cachelib/allocator/CacheTraits.h"
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::regression;
using namespace arma;
using namespace std;


namespace facebook {

namespace cachelib {


class MLModel {
public:
    int n_in_batch = 0;
    bool for_training = false;
    int feature_cnt = 32;

    virtual void* get_model() =0;

    virtual vector<double> get_model_importances() =0;

    virtual int set_model(void* model) =0;

    virtual int emplace_back_training_sample(uint32_t, int window_idx, uint32_t future_interval) =0;

    virtual void emplace_back(uint32_t, int window_idx) =0;

    virtual bool train() =0;

    virtual float info(bool to_print) =0;

    virtual void delete_model() =0;

    virtual vector<double> predict() =0;

    virtual void clear() =0;

    virtual ~MLModel() {};
};

class Regression: public MLModel {
public:
    vector<double> vec_x;
    vector<double> vec_y;
    double learning_rate = 0.00001;
    int num_iters = 1;
    int regression_model = 0;
    int batch_size;
    LinearRegression* lr = NULL;
    mat* theta_pointer;

    void init_with_params(const std::map<std::string, std::string> &params) {
        for (auto &it: params) {
            if (it.first == "regression_iters") {
                num_iters = stoi(it.second);
            } else if (it.first == "regression_model") {
                regression_model = stoi(it.second);
            } else if (it.first == "regression_learning_rate") {
                learning_rate = stof(it.second);
            } else if (it.first == "batch_size") {
                batch_size = stoi(it.second);
            } else if (it.first == "window_cnt") {
                feature_cnt = stoi(it.second);
            }
        }
    }

    Regression(const std::map<std::string, std::string> &params) {
        init_with_params(params);
        theta_pointer = new mat;
        *theta_pointer = arma::zeros<vec>(feature_cnt);
        // cout << "using regression model" << endl;
    }

    void* get_model() {
        if (regression_model == 1)
            return theta_pointer;
        else
            return lr;
    };

    void delete_model() {
        if (lr != NULL)
            delete lr;
    };

    int set_model(void* model) {
        // cout << "set regression model" << endl;
        if (regression_model == 1)
            *theta_pointer = *(mat*)model;
        else {
            // delete lr;
            lr = (LinearRegression*)model;
        }
        return 0;
    };

    mat computeCost(const mat& X, const mat& y, const mat& theta)
    {
        mat J;
        int m = y.n_rows;
        J = arma::sum((pow(((X*theta)-y), 2))/(2*m)) ;
        return J;
    }

    void gradientDescent(const mat& X, const mat& y) {
        mat delta;
        int iter;
        int m = y.n_rows;
        mat theta = *theta_pointer;
        //vec J_history = arma::zeros<vec>(num_iters) ;
        for (iter = 0; iter < num_iters; iter++)
        {
            delta = arma::trans(X)*(X*theta-y)/m;
            // delta += theta;
            // std::cout << X*(*theta_pointer)-y << std::endl;
            // std::cout << delta << std::endl;
            theta = theta - learning_rate * delta;
        }
        auto sq_error = computeCost(X, y, theta)(0);
        if (sq_error < 1e10) {
            *theta_pointer = theta;
        } else {
            *theta_pointer = arma::zeros<vec>(feature_cnt);
        }
    }

    bool train() {
        if (regression_model == 1) {
            mat mat_x = conv_to<mat>::from(vec_x);
            mat_x.reshape(feature_cnt, n_in_batch);
            mat_x = mat_x.t();
            mat mat_y = conv_to<mat>::from(vec_y);
            mat_y.reshape(n_in_batch, 1);
            gradientDescent(mat_x, mat_y);
        } else {
            lr = new LinearRegression();
            mat mat_x = conv_to<mat>::from(vec_x);
            mat_x.reshape(feature_cnt, n_in_batch);
            // cout << mat_x.n_rows << ' ' << mat_x.n_cols << endl;
            // cout << vec_y.size() << endl;
            lr->Train(mat_x, conv_to<rowvec>::from(vec_y));
        }
        return 1;
    }

    vector<double> predict() {
        mat mat_x = conv_to<mat>::from(vec_x);
        mat_x.reshape(feature_cnt, n_in_batch);
        // cout << mat_x.n_rows << ' ' << mat_x.n_cols << endl;
        rowvec predictions;
        if (regression_model == 1) {
            predictions = mat_x.t() * (*theta_pointer);
        } else {
            lr->Predict(mat_x, predictions);
        }
        return conv_to<vector<double>>::from(predictions);
    }

    void clear() {
        n_in_batch = 0;
        vec_x.clear();
        vec_y.clear();
    }

    float info(bool to_print) {
        if (!to_print)
            return 1;
        if (regression_model == 1) {
            std::cout << (*theta_pointer).t() << std::endl;
        } else {
            std::cout << lr->Parameters().t() << std::endl;
        }
        return 1;
    }

    vector<double> get_model_importances() {
        return vector<double>();
    }

    inline float my_faster_logf (float x)
    {
        return 31 - __builtin_clz(uint32_t(x));
    }

    int emplace_back_training_sample(uint32_t access_in_windows, int window_idx, uint32_t future_interval) {
        vector<double> x;
        std::bitset<32> feats(access_in_windows);
        for (int j = window_idx; j > window_idx - feature_cnt; --j) {
            if (j >= 0) {
                x.emplace_back(feats[j % feature_cnt]);
            } else {
                x.emplace_back(0);
            }
        }
        vec_x.insert(vec_x.end(), x.begin(), x.end());
        n_in_batch ++;
        vec_y.push_back(my_faster_logf(future_interval+1));
        return 1;
    }

    void emplace_back(uint32_t access_in_windows, int window_idx) {
        emplace_back_training_sample(access_in_windows, window_idx, 0);
    }
};

class DNN: public MLModel {
public:
    vector<double> vec_x;
    vector<double> vec_y;
    int num_iters = 1;
    int batch_size;

    FFN<mlpack::ann::MeanSquaredError<>, mlpack::ann::RandomInitialization>* model = NULL;

    void init_with_params(const std::map<std::string, std::string> &params) {
        for (auto &it: params) {
            if (it.first == "window_cnt") {
                feature_cnt = stoi(it.second);
            }
        }
    }

    DNN(const std::map<std::string, std::string> &params) {
        init_with_params(params);
        model = new FFN<mlpack::ann::MeanSquaredError<>, mlpack::ann::RandomInitialization>();
        model->Add<Linear<> >(feature_cnt, 64);
        model->Add<BatchNorm<> >(64);
        model->Add<ReLULayer<> >();
        model->Add<Linear<> >(64, 1);
    }

    float info(bool to_print) {
        return 1;
    }

    vector<double> get_model_importances() {
        return vector<double>();
    }

    void* get_model() {
        return model;
    };

    void delete_model() {
        if (model != NULL)
            delete model;
    };

    int set_model(void* model_) {
        model = (FFN<mlpack::ann::MeanSquaredError<>, mlpack::ann::RandomInitialization>*)model_;
        return 0;
    };

    bool train() {
        mat mat_x = conv_to<mat>::from(vec_x);
        mat_x.reshape(feature_cnt, n_in_batch);
        mat mat_y = conv_to<vec>::from(vec_y);
        mat_y.reshape(1, n_in_batch);
        ens::Adam opt(0.02, 1024, 0.9, 0.999, 1e-8, n_in_batch, 1e-8, true);
        // std::cout << "*train" << std::endl;
        model->Train(mat_x, mat_y, opt); //, ens::PrintLoss(), ens::ProgressBar());
        // std::cout << "*train end" << std::endl;
        return 1;
    }

    vector<double> predict() {      
        mat mat_x = conv_to<mat>::from(vec_x);
        mat_x.reshape(feature_cnt, n_in_batch);
        rowvec predictions;
        int mat_n_row = mat_x.n_rows;
        int mat_n_col = mat_x.n_cols;
        model->Predict(mat_x, predictions);
        return conv_to<vector<double>>::from(predictions);
    }

    void clear() {
        n_in_batch = 0;
        vec_x.clear();
        vec_y.clear();
    }

    inline float my_faster_logf (float x)
    {
        return 31 - __builtin_clz(uint32_t(x));
    }
    
    int emplace_back_training_sample(uint32_t access_in_windows, int window_idx, uint32_t future_interval) {
        vector<double> x;
        std::bitset<32> feats(access_in_windows);
        for (int j = window_idx; j > window_idx - feature_cnt; --j) {
            if (j >= 0) {
                x.emplace_back(feats[j % feature_cnt]);
            } else {
                x.emplace_back(0);
            }
        }
        
        vec_x.insert(vec_x.end(), x.begin(), x.end());
        vec_y.push_back(my_faster_logf(future_interval+1));
        n_in_batch ++;
        return 1;
    }

    void emplace_back(uint32_t access_in_windows, int window_idx) {
        emplace_back_training_sample(access_in_windows, window_idx, 0);
    }


};

class LightGBM: public MLModel {
public:
    class Data {
        public:
        vector<float> labels;
        vector<int32_t> indptr;
        vector<int32_t> indices;
        vector<float> data;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & labels;
            ar & indptr;
            ar & indices;
            ar & data;
        }
    };
    Data input_data;
    uint32_t train_feats[130000];
    int32_t train_window_idxs[130000];
    char lgbm_str[400000];
    int lgbm_buffer_len = 400000;
    int64_t lgbm_len;
    int batch_size;
    int belady_boundary_exp;
    int belady_boundary_range;
    bool include_size_in_label = false;
    bool log_size_feature = false;
    bool variance_feature = false;
    bool only_use_edc = false;
    bool save_to_file = false;
    int debug_mode = 0;

    default_random_engine _generator = default_random_engine();
    uniform_int_distribution<std::size_t> _distribution = uniform_int_distribution<std::size_t>();
    BoosterHandle booster = nullptr;

    // feature cnts
    int n_extra_fields = 1;

    unordered_map<string, string> training_params = {
        //don't use alias here. C api may not recongize
        {"boosting",         "gbdt"},
        {"objective",        "regression"}, //"quantile"},
        {"num_iterations",   "32"},
        {"num_leaves",       "32"},
        //{"feature_fraction", "0.8"},
        //{"bagging_freq",     "5"},
        {"bagging_fraction", "0.8"},
        {"learning_rate",    "0.1"},
        {"verbosity",        "-1"},
        {"force_row_wise",   "true"}
    // {"pred_early_stop",  "true"},
    // {"pred_early_stop_freq", "0"},
    // {"pred_early_stop_margin", "0.0"},
    };
    unordered_map<string, string> inference_params;
    map<string, string> model_params;

    float avg_label = 0;
    
    void init_with_params(map<string, string> &params) {
        model_params = params;
        string inference_threads = "1";
        for (auto &it: params) {
            if (it.first == "debug_mode") {
                debug_mode = stoi(it.second);
            }
            if (it.first == "batch_size") {
                batch_size = stoi(it.second);
            } 
            if (it.first == "n_extra_fields") {
                n_extra_fields = stoi(it.second);
            }
            if (it.first == "num_threads") {
                inference_threads = it.second;
            }
            if (it.first == "num_iterations") {
                training_params["num_iterations"] = it.second;
            }
            if (it.first == "num_leaves") {
                training_params["num_leaves"] = it.second;
            }
            if (it.first == "learning_rate") {
                training_params["learning_rate"] = it.second;
            }
            if (it.first == "save_to_file") {
                save_to_file = stoi(it.second);
            }
            if (it.first == "window_cnt") {
                feature_cnt = stoi(it.second);
            }
        }
        // cout << "ML config: " << int(n_deltas) << endl;

        /*if (n_extra_fields) {
            string categorical_feature = to_string(n_deltas + 1);
            for (uint i = 0; i < n_extra_fields - 1; ++i) {
                categorical_feature += "," + to_string(n_deltas + 2 + i);
            }
            training_params["categorical_feature"] = categorical_feature;
        }*/
        inference_params = training_params;
        // can set number of threads, however the inference time will increase a lot (2x~3x)
        inference_params["num_threads"] = inference_threads;
        training_params["num_threads"] = inference_threads;
    }
    
    inline float my_faster_logf (float x)
    {
        return 31 - __builtin_clz(uint32_t(x));
    }

    LightGBM(map<string, string> &params) {
        init_with_params(params);
        input_data.labels.reserve(batch_size);
        input_data.indptr.reserve(batch_size + 1);
        input_data.indptr.emplace_back(0);
        // cout << "using lightgbm model" << endl;
        //input_data.indices.reserve(batch_size * feature_cnt);
        //input_data.data.reserve(batch_size * feature_cnt);
    }

    int set_model(void* lgbm_str_) {
        // cout << "set lightgbm model" << endl;
        int num_iterations;
        return LGBM_BoosterLoadModelFromString((char*)lgbm_str_, &num_iterations, &booster);
    }

    void delete_model() {
        if (booster)
            LGBM_BoosterFree(booster);
    }

    void* get_model() {
        return lgbm_str;
    }

    vector<double> get_model_importances() {
        int res;
        vector<double> importances = vector<double>(feature_cnt, 0);
        res = LGBM_BoosterFeatureImportance(booster,
                                            0,
                                            1,
                                            importances.data());
        if (res == -1) {
            cerr << "error: get model importance fail" << endl;
        }
        for (int i = 0; i < feature_cnt; i ++) {
            cout << i << ' ' << importances[i] << endl;
        }
        return importances;
    }

    int emplace_back_training_sample(uint32_t access_in_windows, int window_idx, uint32_t future_interval) {
        train_feats[n_in_batch] = access_in_windows;
        train_window_idxs[n_in_batch] = window_idx;
        input_data.labels.push_back(my_faster_logf(future_interval+1));
        n_in_batch ++;
        return 0;
    }

    void emplace_back(uint32_t access_in_windows, int window_idx) {
        std::bitset<32> feats(access_in_windows);
        int32_t counter = input_data.indptr.back();
        for (int j = window_idx; j > window_idx - feature_cnt && j >= 0; --j) {
            if (feats[j % feature_cnt]) {
                input_data.data.emplace_back(1);
                input_data.indices.emplace_back(window_idx - j);
                counter ++;
            }
        }
        input_data.indptr.push_back(counter);
        n_in_batch ++;
    }
    
    bool train() {
        auto timeBegin = std::chrono::system_clock::now();
        std::string param_str;
        for (auto it = training_params.cbegin(); it != training_params.cend(); it++) {
            param_str += (it->first) + "=" + (it->second) + " ";
        }
        int32_t counter = input_data.indptr.back();
        for (int i = 0; i < n_in_batch; i ++) {
            std::bitset<32> feats(train_feats[i]);
            for (int j = train_window_idxs[i]; j > train_window_idxs[i] - feature_cnt && j >= 0; --j) {
                if (feats[j % feature_cnt]) {
                    input_data.data.emplace_back(1);
                    input_data.indices.emplace_back(train_window_idxs[i] - j);
                    counter ++;
                    avg_label ++;
                }
            }
            input_data.indptr.push_back(counter);
        }

        // create training dataset
        DatasetHandle trainData;
        LGBM_DatasetCreateFromCSR(
                static_cast<void *>(input_data.indptr.data()),
                C_API_DTYPE_INT32,
                input_data.indices.data(),
                static_cast<void *>(input_data.data.data()),
                C_API_DTYPE_FLOAT32,
                input_data.indptr.size(),
                input_data.data.size(),
                feature_cnt,  //remove future t
                param_str.c_str(),
                nullptr,
                &trainData);

        LGBM_DatasetSetField(trainData,
                            "label",
                            static_cast<void *>(input_data.labels.data()),
                            input_data.labels.size(),
                            C_API_DTYPE_FLOAT32);

        // init booster
        LGBM_BoosterCreate(trainData, param_str.c_str(), &booster);
        // train
        for (int i = 0; i < stoi(training_params["num_iterations"]); i++) {
            int isFinished;
            LGBM_BoosterUpdateOneIter(booster, &isFinished);
            if (isFinished) {
                break;
            }
        }
        bool res = LGBM_BoosterSaveModelToString(booster, 0, 0, C_API_FEATURE_IMPORTANCE_SPLIT, 
            lgbm_buffer_len, &lgbm_len, lgbm_str);
        if (save_to_file) {
            LGBM_BoosterSaveModel(booster, 0, 0, C_API_FEATURE_IMPORTANCE_SPLIT, "model.txt");
        }
        // get_model_importances();
        LGBM_DatasetFree(trainData);
        if (debug_mode >= 1) {
            cout << "model size: " << lgbm_len << endl;
            cout << "number of 1s in feats: " << avg_label / n_in_batch << endl;
            cout << "train time: " << std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now() - timeBegin).count() << endl;
        }
        clear();
        return res;
    }
        
    float info(bool to_print) {
        if (!to_print)
            return 1;
        auto importances = vector<double>(feature_cnt, 0);
        if (booster) {
            LGBM_BoosterFeatureImportance(booster,
                                                0,
                                                1,
                                                importances.data());
        }
        float sum = 0;
        for (int i = 0; i < feature_cnt; i ++) {
            cout << i << ' ' << importances[i] << endl;
            sum += importances[i];
        }
        clear();
        for (int i = 0; i < 32; i ++) {
            emplace_back(1, i);
        }
        auto scores = predict();
        for (int i = 0; i < scores.size(); i ++) {
            double TTA = exp2(scores[i]);
            cout << "feature bit id: " <<  i << ' ' << TTA << endl;
        }
        clear();
        return sum;
    }

    vector<double> predict() {
        auto scores = vector<double>(n_in_batch, 0);

        int64_t len;
        std::string param_str;
        for (auto it = inference_params.cbegin(); it != inference_params.cend(); it++) {
            param_str += (it->first) + "=" + (it->second) + " ";
        }
        LGBM_BoosterPredictForCSR(booster,
                                static_cast<void *>(input_data.indptr.data()),
                                C_API_DTYPE_INT32,
                                input_data.indices.data(),
                                static_cast<void *>(input_data.data.data()),
                                C_API_DTYPE_FLOAT32,
                                input_data.indptr.size(),
                                input_data.data.size(),
                                feature_cnt,  //remove future t
                                C_API_PREDICT_NORMAL,
                                0,
                                0,
                                param_str.c_str(),
                                &len,
                                scores.data());
        return scores;
    }

    void clear() {
        input_data.labels.clear();
        input_data.indptr.resize(1);
        input_data.indices.clear();
        input_data.data.clear();
        n_in_batch = 0;
        avg_label = 0;
        return;
        
    }
};

template <typename CacheTrait>
class EvictionController {
 public:
    using Item = CacheItem<CacheTrait>;
    vector<string> split_string(const string &s, char delim) {
        vector<string> result;
        stringstream ss(s);
        string item;
        while (getline(ss, item, delim)) {
            result.push_back(item);
        }
        return result;
    }

    ~EvictionController() {
        for (auto& p:prediction_threads) {
            p.join();
        }
        for (auto& p:train_threads) {
            p.join();
        }
    }

    MLModel* build_ml_model() {
        if (model_type == 0) {
            return new LightGBM(params);
        } else if (model_type == 1) {
            return new Regression(params);
        } else if (model_type == 2) {
            return new DNN(params);
        }
        return new LightGBM(params);
    }

    EvictionController(string MLConfig) {
        vector<string> config_kv = split_string(MLConfig, ',');
        for (int i = 0; i < config_kv.size(); i += 2) {
            params.insert({config_kv[i], config_kv[i+1]});
        }
        for (auto &it: params) {
            if (it.first == "debug_mode") {
                debug_mode = stoi(it.second);
            }
            if (it.first == "model_type") {
                model_type = stoi(it.second);
            }
            if (it.first == "async_mode") {
                async_mode = stoi(it.second);
            }
            if (it.first == "ml_mess_mode") {
                ml_mess_mode = stoi(it.second);
            }
            if (it.first == "force_ml_mess_mode") {
                force_ml_mess_mode = stoi(it.second);
            }
            if (it.first == "force_run") {
                force_run = stoi(it.second);
            }
            if (it.first == "batch_size_factor") {
                batch_size_factor = stoi(it.second);
            }
            if (it.first == "window_size_factor") {
                window_size_factor = stof(it.second);
            }
            if (it.first == "memory_window_size") {
                memory_window_size = stoi(it.second);
            }
            if (it.first == "use_single_EC") {
                use_single_EC = static_cast<bool>(stoi(it.second));
            }
            if (it.first == "use_global_clock") {
                use_global_clock = static_cast<bool>(stoi(it.second));
            }
            if (it.first == "use_oracle") {
                use_oracle = static_cast<bool>(stoi(it.second));
            }
            if (it.first == "use_admission_control") {
                use_admission_control = static_cast<bool>(stoi(it.second));
            }
            if (it.first == "use_eviction_control") {
                use_eviction_control = static_cast<bool>(stoi(it.second));
            }
            if (it.first == "use_fifo") {
                use_fifo = static_cast<bool>(stoi(it.second));
            }
            if (it.first == "use_admission_threshold") {
                use_admission_threshold = static_cast<bool>(stoi(it.second));
            }
            if (it.first == "training_batch_size") {
                training_batch_size = stoi(it.second);
            }
            if (it.first == "prediction_batch_size") {
                prediction_batch_size = stoi(it.second);
            }
            if (it.first == "max_reinsertions") {
                max_reinsertions = stoi(it.second);
            }
            if (it.first == "rpe_target") {
                rpe_target = stoi(it.second);
                reinsertion_per_eviction = rpe_target;
            }
            if (it.first == "rpe_change_rate") {
                rpe_change_rate = stof(it.second);
            }
            if (it.first == "use_adaptive_rpe") {
                use_adaptive_rpe = stoi(it.second);
            }
            if (it.first == "freq_scaling") {
                freq_scaling = stof(it.second);
            }
            if (it.first == "block_pred_in_nano") {
                block_pred_in_nano = stoi(it.second);
            }
            if (it.first == "sample_rate") {
                sample_rate = stoi(it.second);
            }
            if (it.first == "prediction_size_threshold") {
                prediction_size_threshold = stoi(it.second);
            }
            if (it.first == "process_id") {
                process_id = stoi(it.second);
            }
        }
        params["batch_size"] = to_string(training_batch_size);
        training_model = build_ml_model();
        training_model->for_training = true;
        params["batch_size"] = to_string(prediction_batch_size);
        prediction_model = build_ml_model();
        if (async_mode && use_eviction_control) {
            prediction_threads.push_back(
                std::thread(&EvictionController::prediction_worker, this));
        }
        // std::cout << "done make EC: " << MLConfig << std::endl;
    }

    bool trained() {
        if (use_oracle)
            return false;
        // std::shared_lock lock(model_mutex_);
        return isTrained;
    }

    void prediction_worker() {
        //const pid_t tid = syscall(SYS_gettid); 
        //int ret = setpriority(PRIO_PROCESS, tid, 19);
        folly::setThreadName("pred" + std::to_string(cid));
        struct timespec tim, tim2;
        tim.tv_sec = 0;
        tim.tv_nsec = block_pred_in_nano;
        // std::cout << "pred worker start" << std::endl;
        std::mutex mtx;
        std::unique_lock<std::mutex> lck(mtx);
        while(cv.wait_for(lck, std::chrono::seconds(1000))!=std::cv_status::timeout) {
            Item** items_for_prediction = new Item*[prediction_batch_size];
            int cnt = candidate_queue.try_dequeue_bulk(items_for_prediction, prediction_batch_size);
            // std::cout << "weak up" << std::endl;
            if (!trained()) {
                for (int i = 0; i < cnt; i ++)
                    evict_queue.enqueue(items_for_prediction[i]);
                delete[] items_for_prediction;
                continue;
            }
            prediction_running = true;

            auto timeBegin = std::chrono::system_clock::now();
            std::shared_lock lock(model_mutex_);
            float avg_score = 0, avg_TTA = 0, avg_delta0 = 0, avg_freq = 0;
            // predict TTA in a batch
            for (int i = 0; i < cnt; i ++) {
                Item* item = items_for_prediction[i];
                int window_idx = item->past_timestamp / window_size;
                prediction_model->emplace_back(item->access_in_windows, window_idx);
            }
            build_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now() - timeBegin).count();
            
            // cout << cid << ' ' << logical_time << endl;
            auto scores = prediction_model->predict();
            
            if (block_pred_in_nano != 0) {
                nanosleep(&tim , &tim2);
            }
            for (int i = 0; i < scores.size(); i ++) {
                double TTA = exp2(scores[i]);
                double past_timestamp = items_for_prediction[i]->past_timestamp;
                //avg_score += TTA / scores.size();
                // a hack to retrieve local time temporarily
                int32_t item_logical_time = logical_time;
                if (TTA > item_logical_time - past_timestamp)
                    TTA = TTA - (item_logical_time - past_timestamp);
                else
                    TTA = (item_logical_time - past_timestamp - TTA)
                        * TTA_diff_scaling;

                bool to_reinsert = ifReinsertItem(TTA);
                if (ml_mess_mode && is_mess_period) {
                    to_reinsert = !to_reinsert;
                }
                if(!to_reinsert) {
                    evict_queue.enqueue(items_for_prediction[i]);
                } else {
                    items_for_prediction[i]->set_is_reinserted(1);// already reinserted
                    if (evict_all_mode && !force_ml_mess_mode)
                        evict_queue.enqueue(items_for_prediction[i]);
                }
            }
            prediction_running = false;
            delete[] items_for_prediction;
            predict_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now()
            - timeBegin).count();
            if (debug_mode >= 4 && rand() % 10 == 0)
                cout << "score: " << avg_score 
                     << ", TTA: " << avg_TTA
                     << ", delta0: " << avg_delta0
                     << ", freq: " << avg_freq
                     << ' ' << reinserted_cnt << ' ' << examined_cnt
                     << ' ' << n_loose << ' ' << n_tight << ' ' << threshold
                     << endl;
            prediction_model->clear();
        }
    }

    void adjust_threshold() {
        if (float(reinserted_cnt) / (examined_cnt-reinserted_cnt) < reinsertion_per_eviction)
            threshold *= (1 + threshold_changing_rate * 0.5);
        else
            threshold *= (1 - threshold_changing_rate * 0.5);
    }
    
    void loosen_threshold() {
        threshold *= (1 + threshold_changing_rate);
        n_loose ++;
    }

    void tighten_threshold() {
        threshold *= (1 - threshold_changing_rate);
        n_tight ++;
    }
    
    bool ifReinsertItem(uint32_t predicted_tta) {
        // do not reinsert if the item is in the Tiny List
        // if (!itr.evictMain())
        //     return false;
        // do not need to lock because the corresponding LRU list is locked
        if (repeat == max_reinsertions) {
          repeat = 0;
          return false;
        }
        examined_cnt ++;

        if (!trained()) {
            return false;
        }
        // decide whether to reinsert or evict and adjust the threshold
        if (predicted_tta < threshold) {
            repeat ++;
            reinserted_cnt ++;
            if (repeat > reinsertion_per_eviction &&
                (repeat + 1) % std::max(int(reinsertion_per_eviction), 1) == 0) {
                tighten_threshold();
            }
            return true;
        } else {
            if (repeat < reinsertion_per_eviction)
                loosen_threshold();
            adjust_threshold();
            repeat = 0;
            // generate negative training data
            /* uint32_t current_time;
            if (use_logical_clock) 
                current_time = logical_time; 
            else    
                current_time = util::getCurrentTimeSec();
            _generateTrainingData(meta, current_time + threshold);*/
            return false;
        }
    }

    void recordAccess(Item* item, bool enable_training) {
        if (!use_eviction_control)
            return;
        uint32_t current_time;
        if (use_logical_clock) {
            current_time = logical_time++; //.fetch_add(1, std::memory_order_relaxed);
        } else {
            current_time = util::getCurrentTimeSec();
        }
        if (window_size != 0) {
            if (enable_training) {
               std::lock_guard<std::mutex> guard(training_data_mutex_);
               _generateTrainingData(item, current_time);
            }
            // clear prev round features
            bitset<32> w(item->access_in_windows);
            int past_window_idx = item->past_timestamp / window_size;
            int window_idx = current_time / window_size;
            for (int i = past_window_idx + 1; i < window_idx; i ++)
                w[i % feature_cnt] = 0;
            // update features
            w[window_idx % feature_cnt] = 1;
            item->access_in_windows = w.to_ulong();
        } else {
            item->access_in_windows = 1;
        }
        item->set_past_timestamp(current_time);
        // cout << "after record: " << current_time << ' ' << item->past_timestamp << endl;
    }

    void _generateTrainingData(Item* item, uint32_t current_time) {
        if (item->get_is_sampled() > 0 && window_size > 0) {
            item->set_is_sampled(0);
            training_batch_cnt += 1;
            uint32_t sample_time = item->past_timestamp;
            uint32_t tta_label = current_time - sample_time;
            if (item->get_is_reinserted()) {
                reinsert_cnt ++;
                reinsert_tta += tta_label;
            } else {
                victim_cnt ++;
                victim_tta += tta_label;
            }
            item->set_is_reinserted(0);
            training_model->emplace_back_training_sample(item->access_in_windows,
                sample_time / window_size, tta_label);

            maybe_train();
        }
    }

    void maybe_train() {
        if (training_batch_cnt % training_batch_size == 0) {
            if (training_in_progress) {
                cout << "overlap with prev training: " << int(cid) << ',' << training_batch_cnt << endl;
                training_model->clear();
                training_batch_cnt = 0;
                return;
            }
            temp_training_model = training_model;
            training_model = build_ml_model();
            cout << int(cid) << endl;
            // cout << "train size: " << training_batch_cnt << endl;
            if (async_mode && model_type != 1) {
                train_threads.push_back(std::thread(&EvictionController::train_and_migrate, this));
                // train_threads[train_threads.size()-1].detach();
            } else {
                train_and_migrate();
            }
        }
    }

    void train_and_migrate() {
        //const pid_t tid = syscall(SYS_gettid); 
        //int ret = setpriority(PRIO_PROCESS, tid, 19);
        folly::setThreadName("train" + std::to_string(cid));
        auto timeBegin = std::chrono::system_clock::now();
        training_in_progress = 1;
        // cout << "-------train-------" << endl;
        if (temp_training_model->train() == -1) {
            cout << "training fails!" << endl;
            delete temp_training_model; 
            training_in_progress = 0;
            return;
        }
        double train_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now() - timeBegin).count();
        //cout << "training time: " << train_time << endl;
        
        //cout << "cid: " << cid << endl;
        if (temp_training_model->info(debug_mode >= 3) == 0) {
            cout << "training ends with error!" << endl;
            delete temp_training_model; 
            training_in_progress = 0;
            return;
        }

        if (temp_training_model->set_model(temp_training_model->get_model()) == 0) {
            std::unique_lock lock(model_mutex_);
            prediction_model->delete_model();
            // cout << cid << " is trained" << endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            prediction_model->set_model(temp_training_model->get_model());
        } else {
            cout << "training model has error!" << endl;
        }
        delete temp_training_model;
        isTrained = true;
        training_in_progress = 0;
        return;
    }

    // parameters
    map<string, string> params;
    int debug_mode = 0;
    int model_type = 0;
    bool use_fifo = 0;
    int process_id = -1;
    bool async_mode = 1;
    bool use_single_EC = false;
    bool use_global_clock = false;
    bool use_logical_clock = true;
    bool use_admission_control = false;
    bool use_eviction_control = true;
    bool use_admission_threshold = true;
    bool use_oracle = false;
    int max_reinsertions = 100;
    int training_batch_size = 130000;
    int prediction_batch_size = 128;
    float reinsertion_per_eviction = 3;
    float rpe_target = 3;
    float rpe_change_rate = 0.5;
    bool use_adaptive_rpe = false;
    float threshold_changing_rate = 0.0001;
    float TTA_diff_scaling = 1;
    float freq_scaling = 2;
    int block_pred_in_nano = 0;
    int memory_window_size = 1000000000;
    int forget_num = 0;
    int sample_rate = 64;
    int prediction_size_threshold = 0;
    int batch_size_factor = 10;
    bool prediction_running = false;
    bool ml_mess_mode = 0;
    bool force_ml_mess_mode = 0;
    bool force_run = 0;
    int feature_cnt = 32;
    float window_size_factor = 10;
    int training_sample_rate = 10;
    
    // running variables
    MLModel* training_model = NULL;
    MLModel* temp_training_model = NULL;
    MLModel* prediction_model = NULL;
    bool training_in_progress = 0;
    bool isTrained = false;
    vector<std::thread> train_threads;
    vector<std::thread> prediction_threads;
    std::condition_variable cv;
    moodycamel::ConcurrentQueue<Item*> evict_queue, candidate_queue;
    //std::atomic<bool> ready_for_predict = false;
    // mutable folly::cacheline_aligned<folly::DistributedMutex> mutex_;
    mutable std::mutex training_data_mutex_;
    mutable std::shared_mutex model_mutex_;
    int training_batch_cnt = 0;
    float threshold = 10000;
    float threshold_summary, tta_summary, delta_summary, freq_summary, summary_cnt;
    float candidate_ratio = 1;
    int num_in_cache = 0;
    uint32_t repeat = 0;
    int n_loose = 0, n_tight = 0;
    std::atomic<uint32_t> logical_time = 0;
    int cid;
    bool evict_all_mode = 0;
    bool is_mess_period = 0;
    int window_size = 0;

    // stat
    uint32_t examined_cnt = 0;
    uint32_t reinserted_cnt = 0;
    int tiny_win_cnt = 0;
    int main_win_cnt = 0;
    double train_data_time;
    double training_time;
    double data_dump_time;
    double data_load_time;
    double update_time;
    double build_time = 0;
    double predict_time = 0;
    double victim_tta = 0;
    double reinsert_tta = 0;
    double victim_cnt = 0;
    double reinsert_cnt = 0;
};

} // namespace cachelib
} // namespace facebook