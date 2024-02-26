#pragma once

#include <iostream>
#include <sstream>
#include <stdint.h>
#include <LightGBM/c_api.h>
#include <math.h>
#include <string.h>
#include <bitset>
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
#include <Eigen/Dense>

#define FOLLY_XLOG_MIN_LEVEL INFO
#define LABEL_AS_FEATURE 1

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

    static float my_faster_logf (float val)
    {
        union { float val; int32_t x; } u = { val };
        float log_2 = (float)(((u.x >> 23) & 255) - 128);
        u.x   &= ~(255 << 23);
        u.x   += 127 << 23;
        log_2 += ((-0.3358287811f) * u.val + 2.0f) * u.val  -0.65871759316667f;
        return (log_2);
        // return 31 - __builtin_clz(uint32_t(x));
    }

    static float my_faster_logf (float val, float base)
    {
        return my_faster_logf(val) / my_faster_logf(base);
        // return 31 - __builtin_clz(uint32_t(x));
    }

    virtual void* get_model() =0;

    virtual vector<double> get_model_importances() =0;

    virtual int set_model(void* model) =0;

    virtual int emplace_back_training_sample(uint32_t, int window_idx, uint32_t future_interval) =0;

    virtual void emplace_back(uint32_t, int window_idx) =0;

    virtual bool train() =0;

    virtual float info(bool to_print) =0;

    virtual void delete_model() =0;

    virtual vector<double> predict(int mode) =0;

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

    vector<double> predict(int mode) {
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

    vector<double> predict(int mode) {      
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
    uint32_t* train_feats;
    int32_t* train_window_idxs;
    // char lgbm_str[400000];
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
    uint64_t tta_bucket[100];
    uint32_t tta_feat[100];
    
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


    LightGBM(map<string, string> &params) {
        init_with_params(params);
        train_feats = new uint32_t[batch_size];
        train_window_idxs = new int32_t[batch_size];
        input_data.labels.reserve(batch_size);
        input_data.indptr.emplace_back(0);
    }

    ~LightGBM() {
        delete[] train_feats;
        delete[] train_window_idxs;
    }

    void* get_model() {
        return booster;
    }

    int set_model(void* lgbm) {
        booster = lgbm;
        return 1;
    }

    void delete_model() {
        if (booster) {
            LGBM_BoosterFree(booster);
        }
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
        input_data.labels.push_back(future_interval);
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
            //if (feats[train_window_idxs[i] % feature_cnt] == 0 && train_feats[i] != 0) {
            //    XLOG(INFO) << feats << ' ' << train_window_idxs[i];
            //}
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

        for (int i = 0; i < input_data.labels.size(); i ++) {
            input_data.labels[i] = my_faster_logf(input_data.labels[i]);
        }
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
        /*
        bool res = LGBM_BoosterSaveModelToString(booster, 0, 0, C_API_FEATURE_IMPORTANCE_SPLIT, 
            lgbm_buffer_len, &lgbm_len, lgbm_str);
        if (save_to_file) {
            LGBM_BoosterSaveModel(booster, 0, 0, C_API_FEATURE_IMPORTANCE_SPLIT, "model.txt");
        }
        */
        if (input_data.labels.size() >= 130000 && debug_mode >= 2) {
            cout << "label distribution: " << endl;
            clear_tta();
            get_dist<float>(input_data.labels);
            clear_tta();
            cout << "run label distribution: " << endl;
            auto scores = predict(0);
            get_dist<double>(scores);
            info(1);
        }
        LGBM_DatasetFree(trainData);
        XLOG_EVERY_MS(INFO, 1000) << "<train size=" <<  n_in_batch
            << "> number of 1s in feats: " << avg_label / n_in_batch;
        if (debug_mode >= 2) {
            cout << "train time: " << std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now() - timeBegin).count() << endl;
        }
        clear();
        return 1;
    }

    template <typename T>
    T rotate_left(T x, T amount){
        const unsigned bits=sizeof(T)*8;
        if (amount)
            return (x<<amount)|(x>>(bits-amount));
        return x;
    }

    void update_tta_bucket(double tta, int idx) {
        int bucket_idx = my_faster_logf(tta / 100, 1.2);
        bucket_idx = bucket_idx > 0 ? bucket_idx : 0;
        tta_bucket[bucket_idx] ++;
        if (tta_feat[bucket_idx] == 0) {
            tta_feat[bucket_idx] = rotate_left<uint32_t>(train_feats[idx],
                31 - (train_window_idxs[idx] % feature_cnt));
        }
    }

    void clear_tta() {
        for (int i = 0; i <= 99; i ++) {
            tta_bucket[i] = 0;
            tta_feat[i] = 0;
        }
    }

    void print_tta_bucket() {
        for (int i = 0; i <= 99; i ++) {
            if (tta_bucket[i] < 10)
                continue;
            uint64_t threshold = pow(1.2, i) * 100;
            std::bitset<32> feat(tta_feat[i]);
            cout << threshold << ':' << tta_bucket[i] << ' ' << feat << endl;
        }
    }

    template <class T>
    void get_dist(vector<T> scores) {
        for (int i = 0; i <= 99; i ++) {
            tta_bucket[i] = 0;
        }
        for (int i = 0; i < scores.size(); i ++) {
            double tta = exp2(scores[i]);
            update_tta_bucket(tta, i);
        }
        print_tta_bucket();
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
        return sum;
    }

    vector<double> predict(int mode) {
        auto scores = vector<double>(n_in_batch, 0);
        if (!booster || mode == 2) {
            // XLOG_EVERY_MS(INFO, 1000) << "predict: model not trained";
            return scores;
        }

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
        should_terminate = true;
        cv.notify_all();
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
            if (it.first == "evict_all_mode") {
                evict_all_mode = stoi(it.second);
            }
            if (it.first == "force_run") {
                force_run = stoi(it.second);
            }
            if (it.first == "batch_size_factor") {
                batch_size_factor = stof(it.second);
            }
            if (it.first == "window_size_factor") {
                window_size_factor = stof(it.second);
            }
            if (it.first == "bfRatio") {
                bfRatio = stof(it.second);
            }
            if (it.first == "pRatio") {
                pRatio = stof(it.second);
            }
            if (it.first == "meta_update_ssd") {
                meta_update_ssd = stoi(it.second);
            }
            if (it.first == "reinsert_sample_rate") {
                reinsert_sample_rate = stof(it.second);
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
                use_eviction_control = stoi(it.second);
            }
            if (it.first == "use_fifo") {
                use_fifo = static_cast<bool>(stoi(it.second));
            }
            if (it.first == "use_admission_threshold") {
                use_admission_threshold = static_cast<bool>(stoi(it.second));
            }
            if (it.first == "training_batch_size") {
                training_batch_size = stoi(it.second);
                max_batch_size = stoi(it.second);
            }
            if (it.first == "prediction_batch_size") {
                prediction_batch_size = stoi(it.second);
            }
            if (it.first == "max_reinsertions") {
                max_reinsertions = stoi(it.second);
            }
            if (it.first == "use_largest_tta") {
                use_largest_tta = stoi(it.second);
            }
            if (it.first == "rpe_target") {
                rpe_target = stof(it.second);
                reinsertion_per_eviction = rpe_target;
                if (max_reinsertions < rpe_target * 2)
                    max_reinsertions = rpe_target * 2;
            }
            if (it.first == "threshold_changing_rate") {
                threshold_changing_rate = stof(it.second);
            }
            if (it.first == "use_rpe_running_average") {
                use_rpe_running_average = stoi(it.second);
            }
            if (it.first == "use_rpe_oracle") {
                use_rpe_oracle = stoi(it.second);
            }
            if (it.first == "freq_scaling") {
                freq_scaling = stof(it.second);
            }
            if (it.first == "time_unit") {
                time_unit = stoi(it.second);
            }
            if (it.first == "block_pred_in_nano") {
                block_pred_in_nano = stoi(it.second);
            }
            if (it.first == "sample_rate") {
                sample_rate = stoi(it.second);
            }
            if (it.first == "training_sample_rate") {
                training_sample_rate = stoi(it.second);
            }
            if (it.first == "prediction_size_threshold") {
                prediction_size_threshold = stoi(it.second);
            }
            if (it.first == "process_id") {
                process_id = stoi(it.second);
            } 
            if (it.first == "heuristic_mode") {
                heuristic_mode = stoi(it.second);
            }
            if (it.first == "heuristic_aided") {
                heuristic_aided = stoi(it.second);
            }
        }
        params["batch_size"] = to_string(prediction_batch_size);
        prediction_model = build_ml_model();
        if (async_mode && use_eviction_control) {
            prediction_threads.push_back(
                std::thread(&EvictionController::prediction_worker, this));
        }
        params["batch_size"] = to_string(training_batch_size);
        training_model = build_ml_model();
        training_model->for_training = true;
        // std::cout << "done make EC: " << MLConfig << std::endl;
    }

    bool trained() {
        if (use_oracle)
            return false;
        // std::shared_lock lock(model_mutex_);
        return isTrained;
    }

    void info() {
        std::cout << "<cid=" << int(cid) << ' '
            << "> ml reinserted/evicted: " << reinserted_cnt << ' ' << examined_cnt - reinserted_cnt
            << ' ' << avg_repeat / (examined_cnt - reinserted_cnt)
            << ' ' << "heur: " << n_evict_by_heur
            << ' ' << "threshold: " << threshold
            << " size: " <<  window_size 
            << " train: " << training_batch_cnt
            << " neg:" << neg_cnt
            << std::endl;
        /*if (cid == 1) {
            for (auto bb: beladyBoundaries) {
                cout << bb << ' ';
            }
            cout << endl << endl;
        }*/
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
        while(true) {
            if (should_terminate)
                return;
            if (cv.wait_for(lck, std::chrono::seconds(1000))==std::cv_status::timeout)
                continue;
            Item** items_for_prediction = new Item*[prediction_batch_size];
            Item** items_for_eviction = new Item*[prediction_batch_size];
            int eviction_cnt = 0;
            int cnt = candidate_queue.try_dequeue_bulk(items_for_prediction, prediction_batch_size);
            if (cnt == 0)
                continue;
            candidate_queue_size -= cnt;
            if (evict_all_mode) {
                evict_queue.enqueue_bulk(items_for_prediction, cnt);
                evict_queue_size += cnt;
                delete[] items_for_prediction;
                continue;
            }

            auto timeBegin = std::chrono::system_clock::now();
            std::shared_lock lock(model_mutex_);
            float avg_score = 0, avg_TTA = 0, avg_delta0 = 0, avg_freq = 0;
            // predict TTA in a batch
            for (int i = 0; i < cnt; i ++) {
                Item* item = items_for_prediction[i];
                if (item == NULL) {
                    XLOG(INFO) << "node is null at prediction";
                    break;
                }
                int window_idx = item->get_past_timestamp() / window_size;
                bitset<32> w(item->access_in_windows);
#ifdef LABEL_AS_FEATURE
                if (item->next_timestamp - current_timestamp_global / time_unit > 1e7 && heuristic_aided == 4)
                    w[window_idx] = 0;
#endif
                prediction_model->emplace_back(w.to_ulong(), window_idx);
            }
            build_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now() - timeBegin).count();
            
            auto scores = prediction_model->predict(heuristic_aided);
            
            if (block_pred_in_nano != 0) {
                nanosleep(&tim , &tim2);
            }
            for (int i = 0; i < scores.size(); i ++) {
                double TTA = exp2(scores[i]);
                double past_timestamp = items_for_prediction[i]->get_past_timestamp();
                //avg_score += TTA / scores.size();
                // a hack to retrieve local time temporarily
                int32_t item_logical_time = logical_time;
                if (TTA > item_logical_time - past_timestamp)
                    TTA = TTA - (item_logical_time - past_timestamp);
                else
                    TTA = (item_logical_time - past_timestamp - TTA)
                        * TTA_diff_scaling;

                bool to_reinsert = 0;
                if (rpe_target == 0) {
                    if (items_for_prediction[i]->get_item_flag2())
                        heur_evicted_cnt ++;
                    else
                        heur_reinserted_cnt ++;
                    heur_reinserted_cnt *= (1-1e-5);
                    heur_evicted_cnt *= (1-1e-5);
                    reinsertion_per_eviction = float(heur_reinserted_cnt) / (heur_evicted_cnt + 1);
                }
                if (heuristic_aided == 1) {
                    to_reinsert = 1-items_for_prediction[i]->get_item_flag2();
                }
#ifdef TRUE_TTA
                else if (heuristic_aided == 2) {
                    TTA = items_for_prediction[i]->next_timestamp - current_timestamp_global / time_unit;
                    if (TTA < 0) {
                        neg_tta_cnt ++;
                        XLOG_EVERY_MS(INFO, 1000) << cid << ' ' << items_for_prediction[i]->next_timestamp 
                        << ' ' << current_timestamp_global << ' ' << neg_tta_cnt;
                        TTA = -TTA * 10;
                    }
                    to_reinsert = ifReinsertItem(TTA * 1.01, items_for_prediction[i]);
                }
#endif
                else if (ml_mess_mode) {
                    to_reinsert = ml_mess_mode-1;
                }
                else if (heuristic_aided == 3) {
                    TTA = 5000 * items_for_prediction[i]->get_item_flag2() 
                        + 4000 * (1 - float(i) / scores.size()) + 1000;
                    to_reinsert = ifReinsertItem(TTA, items_for_prediction[i]);
                }
                else {
                    if (trained()) {
                        to_reinsert = ifReinsertItem(TTA, items_for_prediction[i]);
                    } else {
                        to_reinsert = 1-items_for_prediction[i]->get_item_flag2();
                    }
                    
                    if (heuristic_aided == 4) {
                        if (items_for_prediction[i]->next_timestamp > current_timestamp_global / time_unit) {
                            neg_cnt ++;
                            _generateTrainingData(items_for_prediction[i], 
                                items_for_prediction[i]->next_timestamp - current_timestamp_global / time_unit);
                        } else if (items_for_prediction[i]->next_timestamp == 0) {
                            XLOG_EVERY_MS(INFO, 100) << cid << ' ' << items_for_prediction[i]->next_timestamp
                                << ' ' << current_timestamp_global << ' ' << items_for_prediction[i]->getKey();
                        }
                    } else if (heuristic_aided == 5) {
                        if (logical_time - past_timestamp > threshold_of_inactive) {
                            neg_cnt ++;
                            _generateTrainingData(items_for_prediction[i], threshold_of_inactive * 3);
                            items_for_prediction[i]->set_is_sampled(1);
                        }
                    }
                }

                if (items_for_prediction[i]->get_is_sampled() == 0 && heuristic_aided != 4) { 
                // was not accessed again
                    to_reinsert = true;
                    reinserted_by_reuse ++;
                    XLOG_EVERY_MS(INFO, 1000) << "reused candidate cnt: " << reinserted_by_reuse;
                }

                if(!to_reinsert) {
                    //if (trained())
                    //    adjust_threshold_inactive(logical_time - past_timestamp);
                    examined_cnt ++;
                    e_cnt ++;
                    avg_repeat += repeat;
                    if (repeat == max_reinsertions) {
                        n_evict_by_max ++;
                        if (use_largest_tta) {
                            TTA = largest_tta;
                            items_for_eviction[eviction_cnt++] = largest_tta_item;
                            largest_tta = 0;
                            largest_tta_item = nullptr;
                        } else {
                            items_for_eviction[eviction_cnt++] = items_for_prediction[i];
                        }
                        XLOG_EVERY_MS(INFO, 1000) << " pass max reinsertion: " << cid << ' ' << n_evict_by_max;
                    } else {
                        items_for_eviction[eviction_cnt++] = items_for_prediction[i];
                    }
                    repeat = 0;
                } else {
                    reinserted_cnt ++;
                    examined_cnt ++;
                    e_cnt ++;
                    r_cnt ++;
                    repeat ++;
                    //items_for_prediction[i]->set_is_reinserted(1);// already reinserted
                }
            }
            adjust_threshold();
            evict_queue.enqueue_bulk(items_for_eviction, eviction_cnt);
            evict_queue_size += eviction_cnt;
            XLOG_EVERY_MS(INFO, 1000) << "enqueue eviction cnt: " << eviction_cnt;

            delete[] items_for_eviction;
            delete[] items_for_prediction;
            predict_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now()
            - timeBegin).count();
            prediction_model->clear();
        }
    }

    void rebalanceTrueTTA() {
        if (top2node.size() > low2node.size() / rpe_target) {
            node2low[top2node.begin()->second] = 
                low2node.insert({top2node.begin()->first, top2node.begin()->second});
            node2top.erase(top2node.begin()->second);
            top2node.erase(top2node.begin());
        }
        if (top2node.size() < low2node.size() / rpe_target) {
            node2top[low2node.rbegin()->second] = 
                top2node.insert({low2node.rbegin()->first, low2node.rbegin()->second});
            node2low.erase(low2node.rbegin()->second);
            low2node.erase(std::next(low2node.rbegin()).base());
        }
    }

    void removeTrueTTA(Item* node) {
        uint64_t t = node->next_timestamp;
        std::unique_lock lock(oracle_mutex_);
        if (node2low.find(node) != node2low.end()) {
            auto rank_iter = node2low[node];
            low2node.erase(rank_iter);
            node2low.erase(node);
            rebalanceTrueTTA();
        }
        else if (node2top.find(node) != node2top.end()) {
            auto rank_iter = node2top[node];
            top2node.erase(rank_iter);
            node2top.erase(node);
            rebalanceTrueTTA();
        }
/*
        if (node2rank.find(node) != node2rank.end()) {
            auto rank_iter = node2rank[node];
            rank2node.erase(rank_iter);
            node2rank.erase(node);
        }
*/
    }

    void addTrueTTA(Item* node) {
        uint64_t t = 0;
        t = node->next_timestamp;
        std::unique_lock lock(oracle_mutex_);
        if (node2low.find(node) != node2low.end()) {
            auto rank_iter = node2low[node];
            low2node.erase(rank_iter);
            node2low.erase(node);
        }
        if (node2top.find(node) != node2top.end()) {
            auto rank_iter = node2top[node];
            top2node.erase(rank_iter);
            node2top.erase(node);
        }
/*
        if (node2rank.find(node) != node2rank.end()) {
            auto rank_iter = node2rank[node];
            rank2node.erase(rank_iter);
            node2rank.erase(node);
        }
        node2rank[node] = rank2node.insert({t, node});
*/

        auto smallest_in_top = top2node.begin()->first;
        if (t > smallest_in_top) {
            node2top[node] = top2node.insert({t, node});
        } else {
            node2low[node] = low2node.insert({t, node});
        }
        rebalanceTrueTTA();
    }

    void adjust_threshold_oracle() {
        std::unique_lock lock(oracle_mutex_);
        threshold = top2node.begin()->first - current_timestamp_global / time_unit;
/*        int cnt = 0;
        auto it = rank2node.rbegin();
        for (; cnt < rank2node.size() / (1+rpe_target); cnt ++, it ++);
        XLOG_EVERY_MS(INFO, 1000) << threshold << ' ' << it->first - current_timestamp_global / time_unit;
        threshold = it->first - current_timestamp_global / time_unit;
*/
    }

    void update_tta_bucket(double tta) {
        if (use_rpe_running_average || use_rpe_oracle) {
            return;
        }
        int bucket_idx = MLModel::my_faster_logf(tta / 100, 1.2);
        if (bucket_idx < 0) {
            bucket_idx = 0;
        }
        tta_bucket[bucket_idx] ++;
    }

    void adjust_threshold() {
        if (use_rpe_running_average || use_rpe_oracle) {
            return;
        }
        double total = 0;
        for (int i = 0; i <= 99; i ++) {
            total += tta_bucket[i];
        }

        double num_below_threshold = total * reinsertion_per_eviction / (reinsertion_per_eviction + 1);
        double sum = 0;
        threshold = 0;
        for (int i = 0; i <= 99; i ++) {
            sum += tta_bucket[i];
            if (sum >= num_below_threshold && threshold == 0) {
                threshold = pow(1.2, i) * 100;
                if (sum < total * 0.95)
                    threshold *= (1 
                        + 0.2 * (num_below_threshold - (sum-tta_bucket[i])) / tta_bucket[i]);
            }
            tta_bucket[i] *= (1 - (float)prediction_batch_size / window_size);
        }
    }

    void adjust_threshold_inactive(float t) {
        if (cid == 1)
            XLOG_EVERY_MS(INFO, 1000) << "threshold of inactive: " 
                << threshold_of_inactive << ' ' << s_cnt / (s_cnt + l_cnt);
        if (t < threshold_of_inactive) {
            s_cnt ++;
        } else {
            l_cnt ++;
        }
        if (s_cnt / (s_cnt + l_cnt) < 0.95) {    
            threshold_of_inactive *= (1 + threshold_changing_rate);
        } else {
            threshold_of_inactive *= (1 - threshold_changing_rate);
        }
        s_cnt *= (1 - 1e-3);
        l_cnt *= (1 - 1e-3);
    }

    void adjust_threshold_() {
        if (!use_rpe_running_average || use_rpe_oracle) {
            return;
        }
        // XLOG_EVERY_MS(INFO, 1000) << "<cid=" << int(cid) << ' ' << "> threshold: " << threshold
        //    << " rpe: " << float(reinserted_cnt) / (examined_cnt-reinserted_cnt) 
        //    << " r&e: " << reinserted_cnt << ' ' << examined_cnt;
        if ((r_cnt / (e_cnt-r_cnt) < reinsertion_per_eviction - 1
            || r_cnt / (e_cnt-r_cnt) < reinsertion_per_eviction * 0.5)
            && r_cnt / (e_cnt-r_cnt) < reinsertion_per_eviction * 0.9) {
            threshold *= 2;
            r_cnt = 100;
            e_cnt = r_cnt * (1+reinsertion_per_eviction) / reinsertion_per_eviction;
        } else if ((r_cnt / (e_cnt-r_cnt) > reinsertion_per_eviction + 1
            || r_cnt / (e_cnt-r_cnt) > reinsertion_per_eviction * 1.5)
            && r_cnt / (e_cnt-r_cnt) > reinsertion_per_eviction * 1.1) {    
            threshold *= 0.5;
            r_cnt = 100;
            e_cnt = r_cnt * (1+reinsertion_per_eviction) / reinsertion_per_eviction;
        }
        if (r_cnt / (e_cnt-r_cnt) < reinsertion_per_eviction) {    
            threshold *= (1 + threshold_changing_rate);
        } else {
            threshold *= (1 - threshold_changing_rate);
        }
        r_cnt *= (1 - 1e-5);
        e_cnt *= (1 - 1e-5);
    }
    
    bool ifReinsertItem(double predicted_tta, Item* item) {
        update_tta_bucket(predicted_tta);
        if (repeat == max_reinsertions) {
          return false;
        }
        adjust_threshold_();
        // decide whether to reinsert or evict and adjust the threshold
        if (predicted_tta < threshold) {
            // maintain the largest tta of the current search attempt
            if (use_largest_tta) {
                if (predicted_tta > largest_tta) {
                    largest_tta = predicted_tta;
                    largest_tta_item = item;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    void remove(Item* item) {
        if (use_rpe_oracle) {
            removeTrueTTA(item);
            XLOG_EVERY_MS(INFO, 1000) << "True TTA heap size: "
                << node2low.size() << ' ' << node2top.size() << ' ' << window_size << ' ' << node2rank.size();
        }
    }

    void recordAccess(Item* item, bool enable_training) {
        if (!use_eviction_control || evict_all_mode)
            return;
        uint32_t current_time;
        if (use_logical_clock) {
            current_time = logical_time++; //.fetch_add(1, std::memory_order_relaxed);
            current_time = current_time - (current_time&15);
        } else {
            current_time = util::getCurrentTimeSec();
        }
        if (window_size != 0) {
            if (use_rpe_oracle)
                adjust_threshold_oracle();
            if (logical_time % (window_size/ 10) == 0) {
                /*
                if (cid == 1) {
                    std::unique_lock lock(oracle_mutex_);
                    beladyBoundaries.push_back(threshold);
                }
                */
            }
            if (enable_training) {
               _generateTrainingData(item, current_time - item->get_past_timestamp());
            }
            // clear prev round features
            /*if (item->getKey() == "544637") {
                std::cout << "FEATURE of " << item->getKey() << ':' << item->access_in_windows << std::endl;
            }
            */
            bitset<32> w(item->access_in_windows);
            int window_idx = current_time / window_size;
            if (item->get_past_timestamp() != 0) {
                int past_window_idx = item->get_past_timestamp() / window_size;
                if (window_idx - past_window_idx >= feature_cnt) {
                    w.reset();
                } else {
                    for (int i = past_window_idx + 1; i < window_idx && i <= past_window_idx + feature_cnt; i ++)
                        w[i % feature_cnt] = 0;
                }
            }
            // update features
            w[window_idx % feature_cnt] = 1;
            item->access_in_windows = w.to_ulong();
            /*if (item->getKey() == "544637") {
                std::cout << "PASTTIME " << item->past_timestamp << ' ' << item->access_in_windows << std::endl;
            }
            */
        } else {
            item->access_in_windows = 0;
        }
        item->set_past_timestamp(current_time);
        if (use_rpe_oracle)
            addTrueTTA(item);
    }

    void _generateTrainingData(Item* item, uint32_t tta_label) {
        if (item->get_is_sampled() > 0 && window_size > 0 && item->get_past_timestamp() != 0 
                && generate_in_progress.test_and_set() == 0) {
            if (item->get_is_sampled() == 0) {
                generate_in_progress.clear();
                return;
            }
            /*
            if (tta_label < 100) {
                small_label_cnt += 1;
                XLOG_EVERY_MS(INFO, 1000) << "cid: " << cid << "small label: " << small_label_cnt;
            }
            */
            item->set_is_sampled(0);
            uint32_t sample_time = item->get_past_timestamp();

            bitset<32> w(item->access_in_windows);
            int window_idx = sample_time / window_size;
#ifdef LABEL_AS_FEATURE
            if (tta_label > 1e7 && heuristic_aided == 4)
                w[window_idx] = 0;
#endif
            training_model->emplace_back_training_sample(w.to_ulong(), window_idx, tta_label);
            training_batch_cnt += 1;
            maybe_train();
            generate_in_progress.clear();
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
            // cout << int(cid) << endl;
            // cout << "train size: " << training_batch_cnt << endl;
            // cout << "neg_cnt: " << neg_cnt << endl;
            // neg_cnt = 0;
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
        
        // cout << "cid: " << int(cid) << endl;
        if (temp_training_model->info(debug_mode >= 3) == 0) {
            cout << "training ends with error!" << endl;
            delete temp_training_model; 
            training_in_progress = 0;
            return;
        }

        update_model();
        delete temp_training_model;
        isTrained = true;
        training_in_progress = 0;
        return;
    }

    void update_model() {
        std::unique_lock lock(model_mutex_);
        for (int i = 0; i <= 99; i ++)
            tta_bucket[i] *= 0.1;
        prediction_model->delete_model();
        XLOG(INFO) << cid << " is trained" << endl;
        prediction_model->set_model(temp_training_model->get_model());
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
    int use_eviction_control = true;
    bool use_admission_threshold = true;
    bool use_oracle = false;
    int max_reinsertions = 10;
    int training_batch_size = 131072;
    int max_batch_size = 131072;
    int prediction_batch_size = 128;
    float reinsertion_per_eviction = 3;
    float rpe_target = 3;
    bool use_rpe_running_average = false;
    bool use_rpe_oracle = false;
    bool use_largest_tta = false;
    float threshold_changing_rate = 1e-4;
    float TTA_diff_scaling = 1;
    float freq_scaling = 2;
    int block_pred_in_nano = 0;
    int memory_window_size = 1000000000;
    int forget_num = 0;
    int sample_rate = 64;
    int prediction_size_threshold = 0;
    float batch_size_factor = 10;
    bool ml_mess_mode = 0;
    float bfRatio = 0;
    float pRatio = 0.05;
    int meta_update_ssd = 0;
    bool force_run = 0;
    int heuristic_mode = 0;
    int heuristic_aided = 0;
    int feature_cnt = 32;
    float window_size_factor = 10;
    int training_sample_rate = 5;
    int reinsert_sample_rate = 5;
    int time_unit = 1;
    
    // running variables
    MLModel* training_model = NULL;
    MLModel* temp_training_model = NULL;
    MLModel* prediction_model = NULL;
    bool training_in_progress = 0;
    bool enqueue_in_progress = 0;
    std::atomic_flag generate_in_progress = 0;
    bool isTrained = false;
    vector<std::thread> train_threads;
    vector<std::thread> prediction_threads;
    double tta_bucket[100];
    std::condition_variable cv;
    moodycamel::ConcurrentQueue<Item*> evict_queue, candidate_queue;
    std::atomic<int> candidate_queue_size = 0, evict_queue_size = 0;
    mutable std::mutex oracle_mutex_;
    mutable std::shared_mutex model_mutex_;
    int training_batch_cnt = 0;
    int neg_cnt = 0;
    float threshold = 100000;
    float threshold_of_inactive = 10000000;
    float threshold_summary, tta_summary, delta_summary, freq_summary, summary_cnt;
    float candidate_ratio = 1;
    int num_in_cache = 0;
    uint32_t repeat = 0;
    double largest_tta = 0;
    Item* largest_tta_item = nullptr; 
    double avg_repeat = 0;
    bool should_terminate = false;
    int reinserted_by_reuse = 0;
    int small_label_cnt = 0;
    int n_loose = 0, n_tight = 0;
    std::atomic<uint32_t> logical_time = 0;
    int cid;
    bool evict_all_mode = 0;
    bool is_mess_period = 0;
    int window_size = 0;
    int64_t current_timestamp_global;
    std::multimap<int64_t, Item*> low2node, top2node, rank2node;
    std::map<Item*, typename std::multimap<int64_t, Item*>::iterator> node2low, node2top, node2rank;
    std::vector<int64_t> beladyBoundaries;

    // stat
    uint32_t examined_cnt = 0;
    uint32_t reinserted_cnt = 0;
    uint32_t n_evict_by_heur = 0;
    uint32_t n_evict_by_max = 0;
    double heur_evicted_cnt = 0;
    double heur_reinserted_cnt = 0;
    uint64_t neg_tta_cnt = 0;
    float e_cnt = 0;
    float r_cnt = 0;
    float s_cnt = 0;
    float l_cnt = 0;
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