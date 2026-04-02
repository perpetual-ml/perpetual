import argparse
import json
import re
import warnings
from functools import partial
from pathlib import Path
from time import process_time, time

import numpy as np
import optuna
import pandas as pd
import requests
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.datasets import fetch_openml
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# python package-python/examples/benchmark_comparison.py
# python package-python/examples/benchmark_comparison.py --no-optuna
# python package-python/examples/benchmark_comparison.py --skip-catboost
# python package-python/examples/benchmark_comparison.py --catboost-only
# python package-python/examples/benchmark_comparison.py --catboost-only --catboost-results-path tmp/catboost_results.csv

# https://www.openml.org/search?type=study&study_type=task&id=457&sort=tasks_included


"""
All PB (tuned) tabarena performance (v2.1.0):
                                   dataset problem_type   metric     method  metric_error  rank  total_methods dataset_id performance
                                       MIC   multiclass log_loss PB (tuned)      1.238202  18.0             18 46980
                             seismic-bumps       binary  roc_auc PB (tuned)      0.258542  18.0             18 46956      3-2
           taiwanese_bankruptcy_prediction       binary  roc_auc PB (tuned)      0.160871  18.0             18 46962      4-1
     students_dropout_and_academic_success   multiclass log_loss PB (tuned)      1.125762  18.0             18
                   Is-this-a-good-customer       binary  roc_auc PB (tuned)      0.279860  17.0             18
          blood-transfusion-service-center       binary  roc_auc PB (tuned)      0.277322  17.0             18 46913
                                     heloc       binary  roc_auc PB (tuned)      0.212887  17.0             18
                                    anneal   multiclass log_loss PB (tuned)      0.067830  17.0             18 46906
                                     churn       binary  roc_auc PB (tuned)      0.136474  17.0             18
                                  diabetes       binary  roc_auc PB (tuned)      0.175348  17.0             18 46921
                        kddcup09_appetency       binary  roc_auc PB (tuned)      0.369296  17.0             17
               coil2000_insurance_policies       binary  roc_auc PB (tuned)      0.267649  17.0             18
                                    splice   multiclass log_loss PB (tuned)      0.308573  17.0             18
                             Diabetes130US       binary  roc_auc PB (tuned)      0.411460  17.0             17
                    Amazon_employee_access       binary  roc_auc PB (tuned)      0.186921  17.0             17
                              Fitness_Club       binary  roc_auc PB (tuned)      0.194381  16.0             18
                                APSFailure       binary  roc_auc PB (tuned)      0.013972  16.0             17
HR_Analytics_Job_Change_of_Data_Scientists       binary  roc_auc PB (tuned)      0.210702  16.0             17
                     E-CommereShippingData       binary  roc_auc PB (tuned)      0.263426  16.0             18
                             hiva_agnostic   multiclass log_loss PB (tuned)      0.309314  16.0             17
                          website_phishing   multiclass log_loss PB (tuned)      0.294513  16.0             18
                        airfoil_self_noise   regression     rmse PB (tuned)      2.417587  16.0             18
                        QSAR_fish_toxicity   regression     rmse PB (tuned)      0.899660  15.0             18 46954
                 online_shoppers_intention       binary  roc_auc PB (tuned)      0.069808  15.0             18
                               qsar-biodeg       binary  roc_auc PB (tuned)      0.075866  15.0             18
               credit_card_clients_default       binary  roc_auc PB (tuned)      0.225365  15.0             17
                                    SDSS17   multiclass log_loss PB (tuned)      0.131018  15.0             17 46955
                               QSAR-TID-11   regression     rmse PB (tuned)      0.813204  15.0             17
                            bank-marketing       binary  roc_auc PB (tuned)      0.248461  15.0             17
                                  credit-g       binary  roc_auc PB (tuned)      0.218830  14.0             18 46918
                      maternal_health_risk   multiclass log_loss PB (tuned)      0.564076  14.0             18
                          GiveMeSomeCredit       binary  roc_auc PB (tuned)      0.142017  14.0             17 46929
                       Bank_Customer_Churn       binary  roc_auc PB (tuned)      0.139907  14.0             18
                        Marketing_Campaign       binary  roc_auc PB (tuned)      0.096731  14.0             18
                                       jm1       binary  roc_auc PB (tuned)      0.254839  13.0             18
                              wine_quality   regression     rmse PB (tuned)      0.629992  13.0             18
             concrete_compressive_strength   regression     rmse PB (tuned)      4.928130  13.0             18
                    physiochemical_protein   regression     rmse PB (tuned)      3.562011  13.0             17
               polish_companies_bankruptcy       binary  roc_auc PB (tuned)      0.063708  13.0             18
             healthcare_insurance_expenses   regression     rmse PB (tuned)   4625.943316  12.0             18
                             miami_housing   regression     rmse PB (tuned)  85758.029216  11.0             18
     hazelnut-spread-contaminant-detection       binary  roc_auc PB (tuned)      0.022465  11.0             18
          Another-Dataset-on-used-Fiat-500   regression     rmse PB (tuned)    746.753007  11.0             18
                         superconductivity   regression     rmse PB (tuned)      9.575404  11.0             17
          customer_satisfaction_in_airline       binary  roc_auc PB (tuned)      0.005844  11.0             17 46920
                        Food_Delivery_Time   regression     rmse PB (tuned)      7.488482  10.0             17
                                  diamonds   regression     rmse PB (tuned)    531.637650  10.0             17
                                    houses   regression     rmse PB (tuned)      0.217172  10.0             17
                              NATICUSdroid       binary  roc_auc PB (tuned)      0.014906   9.0             18
          in_vehicle_coupon_recommendation       binary  roc_auc PB (tuned)      0.166644   9.0             18
                               Bioresponse       binary  roc_auc PB (tuned)      0.123246   2.0             17

PerpetualBooster 3.0.0rc1 results (tuned):

dataset,method,pb_metric_error,pb_rank,n_methods_compared,tid,did,task_type,problem_type,target_feature,NumberOfClasses,NumberOfFeatures,NumberOfInstances
Another-Dataset-on-used-Fiat-500,PB (tuned),719.7110393314281,1.0,18,363615,46907,Supervised Regression,regression,price,0.0,8.0,1538.0
Bioresponse,PB (tuned),0.1221640488656196,1.0,18,363620,46912,Supervised Classification,binary,MoleculeElicitsResponse,2.0,1777.0,3751.0
Marketing_Campaign,PB (tuned),0.07171086180520136,1.0,18,363684,46940,Supervised Classification,binary,Response,2.0,26.0,2240.0
healthcare_insurance_expenses,PB (tuned),4186.912613037499,1.0,18,363675,46931,Supervised Regression,regression,charges,0.0,7.0,1338.0
splice,PB (tuned),0.10279068474808135,2.0,18,363702,46958,Supervised Classification,multiclass,SiteType,3.0,61.0,3190.0
taiwanese_bankruptcy_prediction,PB (tuned),0.051936488169364914,3.0,18,363706,46962,Supervised Classification,binary,Bankrupt,2.0,95.0,6819.0
churn,PB (tuned),0.07280673702163953,4.0,18,363623,46915,Supervised Classification,binary,CustomerChurned,2.0,20.0,5000.0
E-CommereShippingData,PB (tuned),0.25774848488219604,5.0,18,363632,46924,Supervised Classification,binary,ArrivedLate,2.0,11.0,10999.0
jm1,PB (tuned),0.24378392741904875,5.0,18,363712,46979,Supervised Classification,binary,defects,2.0,22.0,10885.0
Bank_Customer_Churn,PB (tuned),0.13038893465243284,6.0,18,363619,46911,Supervised Classification,binary,churn,2.0,11.0,10000.0
polish_companies_bankruptcy,PB (tuned),0.03585059978189742,6.0,18,363694,46950,Supervised Classification,binary,company_bankrupt,2.0,65.0,5910.0
in_vehicle_coupon_recommendation,PB (tuned),0.16609287882768498,8.0,18,363681,46937,Supervised Classification,binary,AcceptCoupon,2.0,25.0,12684.0
Food_Delivery_Time,PB (tuned),7.531697842070671,10.0,18,363672,46928,Supervised Regression,regression,Time_taken(min),0.0,10.0,45451.0
houses,PB (tuned),0.2189612400514908,10.0,18,363678,46934,Supervised Regression,regression,LnMedianHouseValue,0.0,9.0,20640.0
customer_satisfaction_in_airline,PB (tuned),0.006049132292364923,11.0,18,363628,46920,Supervised Classification,binary,satisfaction,2.0,22.0,129880.0
diamonds,PB (tuned),540.1534852944271,11.0,18,363631,46923,Supervised Regression,regression,price,0.0,10.0,53940.0
superconductivity,PB (tuned),9.553816255683303,11.0,18,363705,46961,Supervised Regression,regression,critical_temp,0.0,82.0,21263.0
wine_quality,PB (tuned),0.6227992745117511,12.0,18,363708,46964,Supervised Regression,regression,median_wine_quality,0.0,13.0,6497.0
anneal,PB (tuned),0.035889218714893645,13.0,18,363614,46906,Supervised Classification,multiclass,classes,5.0,39.0,898.0
miami_housing,PB (tuned),87784.37215632183,13.0,18,363686,46942,Supervised Regression,regression,SALE_PRC,0.0,16.0,13776.0
NATICUSdroid,PB (tuned),0.01707047293553332,14.0,18,363689,46969,Supervised Classification,binary,Malware,2.0,87.0,7491.0
SDSS17,PB (tuned),0.08762948766631677,14.0,18,363699,46955,Supervised Classification,multiclass,ObjectType,3.0,12.0,78053.0
coil2000_insurance_policies,PB (tuned),0.2558389753582997,14.0,18,363624,46916,Supervised Classification,binary,MobileHomePolicy,2.0,86.0,9822.0
hazelnut-spread-contaminant-detection,PB (tuned),0.027149999999999896,14.0,18,363674,46930,Supervised Classification,binary,Contaminated,2.0,31.0,2400.0
maternal_health_risk,PB (tuned),0.5294716401552494,14.0,18,363685,46941,Supervised Classification,multiclass,RiskLevel,3.0,7.0,1014.0
physiochemical_protein,PB (tuned),3.5898867185082426,14.0,18,363693,46949,Supervised Regression,regression,ResidualSize,0.0,10.0,45730.0
HR_Analytics_Job_Change_of_Data_Scientists,PB (tuned),0.20224224218715603,15.0,18,363679,46935,Supervised Classification,binary,LookingForJobChange,2.0,13.0,19158.0
MIC,PB (tuned),0.4785837728960037,15.0,18,363711,46980,Supervised Classification,multiclass,LET_IS,8.0,112.0,1699.0
bank-marketing,PB (tuned),0.24631949598167324,15.0,18,363618,46910,Supervised Classification,binary,SubscribeTermDeposit,2.0,14.0,45211.0
credit_card_clients_default,PB (tuned),0.2218481497941378,15.0,18,363627,46919,Supervised Classification,binary,DefaultOnPaymentNextMonth,2.0,24.0,30000.0
Fitness_Club,PB (tuned),0.19390880217785844,16.0,18,363671,46927,Supervised Classification,binary,attended,2.0,7.0,1500.0
GiveMeSomeCredit,PB (tuned),0.15410723943371973,16.0,18,363673,46929,Supervised Classification,binary,FinancialDistressNextTwoYears,2.0,11.0,150000.0
QSAR-TID-11,PB (tuned),0.8578547523515985,16.0,18,363697,46953,Supervised Regression,regression,MEDIAN_PXC50,0.0,1025.0,5742.0
airfoil_self_noise,PB (tuned),1.9171906454706062,16.0,18,363612,46904,Supervised Regression,regression,scaled-sound-pressure,0.0,6.0,1503.0
concrete_compressive_strength,PB (tuned),5.364084848050503,16.0,18,363625,46917,Supervised Regression,regression,ConcreteCompressiveStrength,0.0,9.0,1030.0
credit-g,PB (tuned),0.22303418803418795,16.0,18,363626,46918,Supervised Classification,binary,good_or_bad_customer,2.0,21.0,1000.0
online_shoppers_intention,PB (tuned),0.07809676917116559,16.0,18,363691,46947,Supervised Classification,binary,Revenue,2.0,18.0,12330.0
website_phishing,PB (tuned),0.2911696798264711,16.0,18,363707,46963,Supervised Classification,multiclass,WebsiteType,3.0,10.0,1353.0
Amazon_employee_access,PB (tuned),0.1807549825397392,17.0,18,363613,46905,Supervised Classification,binary,ResourceApproved,2.0,10.0,32769.0
Is-this-a-good-customer,PB (tuned),0.28269929154015594,17.0,18,363682,46938,Supervised Classification,binary,bad_client_target,2.0,14.0,1723.0
QSAR_fish_toxicity,PB (tuned),0.9265520335204859,17.0,18,363698,46954,Supervised Regression,regression,LC50,0.0,7.0,907.0
hiva_agnostic,PB (tuned),0.3285674240514629,17.0,18,363677,46933,Supervised Classification,multiclass,CompoundActivity,3.0,1618.0,3845.0
seismic-bumps,PB (tuned),0.24282445243543638,17.0,18,363700,46956,Supervised Classification,binary,HighEnergySeismicBump,2.0,16.0,2584.0
APSFailure,PB (tuned),0.021148312367939903,18.0,18,363616,46908,Supervised Classification,binary,AirPressureSystemFailure,2.0,171.0,76000.0
Diabetes130US,PB (tuned),0.3803649352985119,18.0,18,363630,46922,Supervised Classification,binary,EarlyReadmission,2.0,48.0,71518.0
blood-transfusion-service-center,PB (tuned),0.35298245614035084,18.0,18,363621,46913,Supervised Classification,binary,DonatedBloodInMarch2007,2.0,5.0,748.0
diabetes,PB (tuned),0.1771419009370816,18.0,18,363629,46921,Supervised Classification,binary,TestedPositiveForDiabetes,2.0,9.0,768.0
heloc,PB (tuned),0.21382097206932238,18.0,18,363676,46932,Supervised Classification,binary,RiskPerformance,2.0,24.0,10459.0
kddcup09_appetency,PB (tuned),0.2647601035811177,18.0,18,363683,46939,Supervised Classification,binary,appetency,2.0,213.0,50000.0
qsar-biodeg,PB (tuned),0.09030908500739354,18.0,18,363696,46952,Supervised Classification,binary,Biodegradable,2.0,42.0,1054.0
students_dropout_and_academic_success,PB (tuned),0.9426793317606776,18.0,18,363704,46960,Supervised Classification,multiclass,AcademicOutcome,3.0,37.0,4424.0

"""


# Dataset IDs with the most number of rows
# TASK_IDS = [46955, 46920, 46929]
# Dataset IDs with the least number of rows (max 1000 rows)
# TASK_IDS = [46913, 46921, 46906, 46954, 46918]
# All 11 datasets
TASK_IDS = [46913, 46921, 46906, 46954, 46918, 46955, 46920, 46929, 46980, 46956, 46962]
WEAKEST_RC1_TASK_IDS = [
    46960,
    46922,
    46913,
    46939,
    46932,
    46952,
    46908,
    46954,
    46933,
    46938,
]

PERPETUAL_BOOSTER_CLS = None

TABARENA_REFERENCE_RANKS = {
    46913: {
        "dataset": "blood-transfusion-service-center",
        "v2_rank": 17.0,
        "v3_rank": 18.0,
    },
    46921: {"dataset": "diabetes", "v2_rank": 17.0, "v3_rank": 18.0},
    46906: {"dataset": "anneal", "v2_rank": 17.0, "v3_rank": 13.0},
    46954: {"dataset": "QSAR_fish_toxicity", "v2_rank": 15.0, "v3_rank": 17.0},
    46918: {"dataset": "credit-g", "v2_rank": 14.0, "v3_rank": 16.0},
    46955: {"dataset": "SDSS17", "v2_rank": 15.0, "v3_rank": 14.0},
    46920: {
        "dataset": "customer_satisfaction_in_airline",
        "v2_rank": 11.0,
        "v3_rank": 11.0,
    },
    46929: {"dataset": "GiveMeSomeCredit", "v2_rank": 14.0, "v3_rank": 16.0},
    46980: {"dataset": "MIC", "v2_rank": 18.0, "v3_rank": 15.0},
    46956: {"dataset": "seismic-bumps", "v2_rank": 18.0, "v3_rank": 17.0},
    46962: {
        "dataset": "taiwanese_bankruptcy_prediction",
        "v2_rank": 18.0,
        "v3_rank": 3.0,
    },
}

cb_iterations = 100
n_trials = 100
PERPETUAL_BUDGET = 2.0
use_optuna = True  # Set to False to use default CatBoost parameters (skip Optuna)
skip_catboost = False  # Set to True to skip CatBoost completely
n_splits = 3


def resolve_task(task_id):
    """
    Fetches the source dataset ID and target feature for a given OpenML task ID.
    """
    url = f"https://www.openml.org/api/v1/xml/task/{task_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            text = response.text
            did_match = re.search(r"<oml:data_set_id>(\d+)</oml:data_set_id>", text)
            target_match = re.search(
                r"<oml:target_feature>(.*?)</oml:target_feature>", text
            )

            did = int(did_match.group(1)) if did_match else task_id
            target = target_match.group(1) if target_match else None
            return did, target
    except Exception as e:
        print(f"Warning: Could not resolve task {task_id}: {e}")
    return task_id, None


def prepare_data(task_id, seed):
    # Resolve Task ID to Dataset ID and target feature
    data_id, target = resolve_task(task_id)

    print(f"Loading Dataset ID {data_id} (from Task {task_id})...")
    try:
        # Fetch the dataset
        data = fetch_openml(
            data_id=data_id, return_X_y=True, as_frame=True, parser="auto"
        )
        X, y = data
        print(f"Data shape: {X.shape}")
    except Exception as e:
        print(f"Error fetching data for data_id {data_id}: {e}")
        raise

    # If y is a DataFrame (multi-target) or if we have a specific target name
    if isinstance(y, pd.DataFrame):
        if target and target in y.columns:
            y = y[target]
        else:
            y = y.iloc[:, 0]

    # Simple task type detection
    if (
        y.dtype == "object"
        or y.dtype.name == "category"
        or len(np.unique(y.dropna())) < 20
    ):
        n_classes = len(np.unique(y.dropna()))
        if n_classes == 2:
            task_type = "binary"
            scoring = "roc_auc"
            metric_function = roc_auc_score
        else:
            task_type = "multiclass"
            scoring = "neg_log_loss"
            metric_function = log_loss
    else:
        task_type = "regression"
        scoring = "neg_mean_squared_error"
        metric_function = mean_squared_error

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y if task_type in {"binary", "multiclass"} else None,
    )

    return (X_train, X_test, y_train, y_test, task_type, scoring, metric_function)


def build_cv(task_type, seed):
    if task_type in {"binary", "multiclass"}:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def score_predictions(task_type, y_true, y_pred):
    if task_type == "binary":
        return roc_auc_score(y_true, y_pred)
    if task_type == "multiclass":
        return -log_loss(y_true, y_pred)
    return -mean_squared_error(y_true, y_pred)


def evaluate_predictions(task_type, y_true, y_pred):
    if task_type == "binary":
        return roc_auc_score(y_true, y_pred)
    if task_type == "multiclass":
        return log_loss(y_true, y_pred)
    return mean_squared_error(y_true, y_pred)


def parse_task_ids(raw_task_ids):
    if not raw_task_ids:
        return TASK_IDS
    if raw_task_ids == "weakest-rc1-10":
        return WEAKEST_RC1_TASK_IDS
    return [
        int(task_id.strip()) for task_id in raw_task_ids.split(",") if task_id.strip()
    ]


def get_perpetual_booster_cls():
    global PERPETUAL_BOOSTER_CLS
    if PERPETUAL_BOOSTER_CLS is None:
        from perpetual import PerpetualBooster

        PERPETUAL_BOOSTER_CLS = PerpetualBooster
    return PERPETUAL_BOOSTER_CLS


def metric_higher_is_better(task_type):
    return task_type == "binary"


def count_perpetual_wins(task_type, pb_scores, cb_scores):
    if metric_higher_is_better(task_type):
        return sum(
            pb_score > cb_score for pb_score, cb_score in zip(pb_scores, cb_scores)
        )
    return sum(pb_score < cb_score for pb_score, cb_score in zip(pb_scores, cb_scores))


def summarize_task_result(result):
    task_type = result.get("task_type")
    reference = TABARENA_REFERENCE_RANKS.get(result["data_id"], {})
    higher_is_better = metric_higher_is_better(task_type) if task_type else None
    catboost_mean = np.mean(result["CatBoost"]) if "CatBoost" in result else np.nan
    pb_mean = np.mean(result["PB_test"]) if "PB_test" in result else np.nan
    wins = None
    pb_advantage = None
    if "CatBoost" in result and "PB_test" in result and task_type is not None:
        wins = count_perpetual_wins(task_type, result["PB_test"], result["CatBoost"])
        pb_advantage = (
            pb_mean - catboost_mean if higher_is_better else catboost_mean - pb_mean
        )

    return {
        "task_id": result["data_id"],
        "dataset": reference.get("dataset"),
        "task_type": task_type,
        "catboost_mean": catboost_mean,
        "pb_mean": pb_mean,
        "pb_advantage": pb_advantage,
        "pb_wins": wins,
        "seed_count": len(result.get("PB_test", result.get("CatBoost", []))),
        "reference_v2_rank": reference.get("v2_rank"),
        "reference_v3_rank": reference.get("v3_rank"),
        "reference_rank_delta": (
            None
            if "v2_rank" not in reference or "v3_rank" not in reference
            else reference["v2_rank"] - reference["v3_rank"]
        ),
    }


def predict_for_metric(model, X_data, task_type):
    if task_type == "binary":
        return model.predict_proba(X_data)[:, 1]
    if task_type == "multiclass":
        return model.predict_proba(X_data)
    return model.predict(X_data)


def prepare_catboost_frame(X_data):
    X_catboost = X_data.copy()
    cat_features = X_catboost.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for feature in cat_features:
        values = X_catboost[feature].astype("object")
        X_catboost[feature] = values.where(values.notna(), "__nan__").astype(str)

    return X_catboost, cat_features


def objective_catboost(trial, X_train, y_train, task_type, seed):
    X_train_cb, cat_features = prepare_catboost_frame(X_train)
    params = {
        "iterations": cb_iterations,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-6, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-6, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "od_type": "Iter",
        "od_wait": 20,
        "verbose": False,
        "random_seed": seed,
        "allow_writing_files": False,
    }

    cv = build_cv(task_type, seed)

    scores = []
    for train_idx, val_idx in cv.split(X_train_cb, y_train):
        X_tr, X_val = X_train_cb.iloc[train_idx], X_train_cb.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        if task_type in {"binary", "multiclass"}:
            model = CatBoostClassifier(**params)
        else:
            model = CatBoostRegressor(**params)

        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=20,
            cat_features=cat_features,
        )

        preds = predict_for_metric(model, X_val, task_type)
        score = score_predictions(task_type, y_val, preds)
        scores.append(score)

    return np.mean(scores)


def run_catboost(X_train, y_train, X_test, y_test, task_type, seed):
    start_cpu = process_time()
    start_wall = time()
    X_train_cb, cat_features = prepare_catboost_frame(X_train)
    X_test_cb, _ = prepare_catboost_frame(X_test)

    if use_optuna:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        obj = partial(
            objective_catboost,
            X_train=X_train_cb,
            y_train=y_train,
            task_type=task_type,
            seed=seed,
        )

        study.optimize(obj, n_trials=n_trials)
        best_params = study.best_params
        best_params["iterations"] = cb_iterations
    else:
        best_params = {}

    best_params["verbose"] = False
    best_params["random_seed"] = seed
    best_params["allow_writing_files"] = False

    cv = build_cv(task_type, seed)

    ensemble_preds = []

    for train_idx, val_idx in cv.split(X_train_cb, y_train):
        X_tr, y_tr = X_train_cb.iloc[train_idx], y_train.iloc[train_idx]
        if task_type in {"binary", "multiclass"}:
            model = CatBoostClassifier(**best_params)
        else:
            model = CatBoostRegressor(**best_params)

        model.fit(X_tr, y_tr, cat_features=cat_features)

        ensemble_preds.append(predict_for_metric(model, X_test_cb, task_type))

    y_pred = np.mean(ensemble_preds, axis=0)
    stop_cpu = process_time()
    stop_wall = time()

    metric = evaluate_predictions(task_type, y_test, y_pred)

    best_cv_score = study.best_value if use_optuna else None

    return {
        "metric": metric,
        "cpu_time": stop_cpu - start_cpu,
        "wall_time": stop_wall - start_wall,
        "best_cv_score": best_cv_score,
        "best_params": dict(best_params),
    }


def run_perpetual(
    X_train,
    y_train,
    X_test,
    y_test,
    task_type,
    seed,
    budget,
    stopping_rounds=None,
    iteration_limit=None,
):
    objective = "LogLoss" if task_type in {"binary", "multiclass"} else "SquaredLoss"
    model = get_perpetual_booster_cls()(
        objective=objective,
        budget=budget,
        max_cat=1000,
        seed=seed,
        stopping_rounds=stopping_rounds,
        iteration_limit=iteration_limit,
    )

    start_cpu = process_time()
    start_wall = time()
    model.fit(X_train, y_train)
    stop_cpu = process_time()
    stop_wall = time()

    y_pred = predict_for_metric(model, X_test, task_type)
    metric = evaluate_predictions(task_type, y_test, y_pred)

    return metric, stop_cpu - start_cpu, stop_wall - start_wall


def persist_catboost_results(rows, path):
    if path is None or not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates(subset=["task_id", "seed"], keep="last").to_csv(
        path, index=False
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PB vs CatBoost. Use --no-optuna to skip CatBoost tuning "
            "or --catboost-only to tune CatBoost and save its results without "
            "running PerpetualBooster."
        )
    )
    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Skip CatBoost Optuna optimization and use default parameters",
    )
    parser.add_argument(
        "--skip-catboost",
        action="store_true",
        help="Skip CatBoost completely",
    )
    parser.add_argument(
        "--catboost-only",
        action="store_true",
        help=(
            "Run only CatBoost (with Optuna unless --no-optuna is set), skip "
            "PerpetualBooster, and save CatBoost results to CSV"
        ),
    )
    parser.add_argument(
        "--catboost-results-path",
        default=None,
        help=(
            "Optional CSV path for saved CatBoost results. Defaults to "
            "catboost_results.csv when --catboost-only is used. When used with --skip-catboost, reads results from this file to compare."
        ),
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=PERPETUAL_BUDGET,
        help="PerpetualBooster fixed benchmark budget",
    )
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated OpenML task IDs to run, or 'weakest-rc1-10'. Defaults to the built-in 11-task subset.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=5,
        help="Number of random seeds to evaluate per task",
    )
    parser.add_argument(
        "--perpetual-stopping-rounds",
        type=int,
        default=None,
        help="Optional explicit stopping_rounds override for PerpetualBooster",
    )
    parser.add_argument(
        "--perpetual-iteration-limit",
        type=int,
        default=None,
        help="Optional explicit iteration_limit override for PerpetualBooster",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional CSV path for the aggregated benchmark summary",
    )
    args = parser.parse_args()

    # Initialize execution flags with default values.
    use_optuna = True
    skip_catboost = False
    skip_perpetual = False

    if args.catboost_only and args.skip_catboost:
        parser.error("--catboost-only cannot be combined with --skip-catboost")

    if args.no_optuna:
        use_optuna = False
        print("Optuna optimization skipped. Using default CatBoost parameters.")

    PERPETUAL_BUDGET = args.budget
    task_ids = parse_task_ids(args.task_ids)
    seed_count = max(args.seed_count, 1)

    print(
        f"Benchmark config: tasks={task_ids}, seeds={seed_count}, "
        f"pb_budget={PERPETUAL_BUDGET:.2f}, "
        f"pb_stopping_rounds={args.perpetual_stopping_rounds}, "
        f"pb_iteration_limit={args.perpetual_iteration_limit}"
    )

    if args.skip_catboost:
        skip_catboost = True
        print("CatBoost evaluation skipped completely.")

    catboost_results_path = None
    if args.catboost_only:
        skip_perpetual = True
        catboost_results_path = Path(
            args.catboost_results_path or "catboost_results.csv"
        )
        print(
            "Running CatBoost-only mode. PerpetualBooster evaluation skipped; "
            f"results will be saved to {catboost_results_path}."
        )
    elif args.catboost_results_path:
        catboost_results_path = Path(args.catboost_results_path)

    catboost_saved_rows = []

    loaded_catboost_results = {}
    if catboost_results_path and catboost_results_path.exists():
        cb_df = pd.read_csv(catboost_results_path)
        catboost_saved_rows = cb_df.to_dict("records")
        for _, row in cb_df.iterrows():
            loaded_catboost_results[(int(row["task_id"]), int(row["seed"]))] = row[
                "metric"
            ]

    all_results = []

    summary_path = Path(args.summary_path) if args.summary_path else None

    for data_id in task_ids:
        print(f"\nEvaluating Task ID: {data_id}")
        task_results = {"data_id": data_id}

        # Methodology: 5 seeds
        seeds = range(seed_count)

        for seed in seeds:
            try:
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    task_type,
                    scoring,
                    metric_function,
                ) = prepare_data(data_id, seed)
                task_results["task_type"] = task_type

                # CatBoost
                if not skip_catboost and (data_id, seed) in loaded_catboost_results:
                    cb_metric = loaded_catboost_results[(data_id, seed)]
                    print(
                        f"Seed {seed} - CatBoost: reusing saved result {cb_metric:.4f}"
                    )
                    if "CatBoost" not in task_results:
                        task_results["CatBoost"] = []
                    task_results["CatBoost"].append(cb_metric)
                elif not skip_catboost:
                    cb_result = run_catboost(
                        X_train, y_train, X_test, y_test, task_type, seed
                    )
                    print(
                        "Seed "
                        f"{seed} - CatBoost: {cb_result['metric']:.4f}, CPU: "
                        f"{cb_result['cpu_time']:.2f}s, Wall: "
                        f"{cb_result['wall_time']:.2f}s"
                    )
                    if "CatBoost" not in task_results:
                        task_results["CatBoost"] = []
                    task_results["CatBoost"].append(cb_result["metric"])

                    if catboost_results_path is not None and not skip_catboost:
                        catboost_saved_rows.append(
                            {
                                "task_id": data_id,
                                "seed": seed,
                                "task_type": task_type,
                                "metric": cb_result["metric"],
                                "cpu_time": cb_result["cpu_time"],
                                "wall_time": cb_result["wall_time"],
                                "use_optuna": use_optuna,
                                "best_cv_score": cb_result["best_cv_score"],
                                "best_params": json.dumps(
                                    cb_result["best_params"], sort_keys=True
                                ),
                            }
                        )
                        persist_catboost_results(
                            catboost_saved_rows, catboost_results_path
                        )
                elif skip_catboost and (data_id, seed) in loaded_catboost_results:
                    cb_metric = loaded_catboost_results[(data_id, seed)]
                    if "CatBoost" not in task_results:
                        task_results["CatBoost"] = []
                    task_results["CatBoost"].append(cb_metric)

                # PerpetualBooster fixed-budget benchmark run.
                if not skip_perpetual:
                    pb_metric, pb_cpu, pb_wall = run_perpetual(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        task_type,
                        seed,
                        PERPETUAL_BUDGET,
                        stopping_rounds=args.perpetual_stopping_rounds,
                        iteration_limit=args.perpetual_iteration_limit,
                    )
                    print(
                        "Seed "
                        f"{seed} - Perpetual budget={PERPETUAL_BUDGET:.1f}: "
                        f"Test={pb_metric:.4f}, CPU: "
                        f"{pb_cpu:.2f}s, Wall: {pb_wall:.2f}s"
                    )

                    if "PB_budget" not in task_results:
                        task_results["PB_budget"] = []
                    task_results["PB_budget"].append(PERPETUAL_BUDGET)

                    if "PB_test" not in task_results:
                        task_results["PB_test"] = []
                    task_results["PB_test"].append(pb_metric)

            except Exception as e:
                print(f"Error evaluating Dataset ID {data_id} with seed {seed}: {e}")
                continue

        all_results.append(task_results)

    # Print final summary table
    print("\n" + "=" * 50)
    print(f"FINAL SUMMARY (Averaged over {seed_count} seeds)")
    print("=" * 50)
    summary_rows = []
    for res in all_results:
        summary_row = summarize_task_result(res)
        summary_rows.append(summary_row)
        summary = f"Dataset {res['data_id']}: "
        if "CatBoost" in res:
            summary += f"CatBoost={np.mean(res['CatBoost']):.4f}, "

        if "PB_test" in res:
            wins = None
            if "CatBoost" in res:
                wins = count_perpetual_wins(
                    res["task_type"], res["PB_test"], res["CatBoost"]
                )
            summary += (
                f"PB(test)={np.mean(res['PB_test']):.4f}, "
                f"PB(budget)={PERPETUAL_BUDGET:.1f}"
            )
            if wins is not None:
                summary += f", PB wins={wins}/{seed_count}"

        if summary_row["reference_v3_rank"] is not None:
            summary += (
                ", rank v2->v3="
                f"{summary_row['reference_v2_rank']:.0f}->{summary_row['reference_v3_rank']:.0f}"
            )

        print(summary)

    if catboost_results_path is not None and catboost_saved_rows:
        persist_catboost_results(catboost_saved_rows, catboost_results_path)
        print(f"Saved CatBoost results to {catboost_results_path}")

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Saved benchmark summary to {summary_path}")

"""
==================================================
FINAL SUMMARY (Averaged over 5 seeds)
==================================================
Catboost with default parameters (no Optuna):
Dataset 46956: CatBoost=0.7690, PB_0.5=0.7269, PB_1.0=0.7531, PB_1.5=0.7525, PB_2.0=0.7530, PB_2.5=0.7563

((self.cfg.budget.max(1.0) * cols.max(1) as f32) * 10.0).round().clamp(256.0, ITER_LIMIT as f32) as usize
Dataset 46956:                  PB_0.5=0.7491, PB_1.0=0.7608, PB_1.5=0.7523, PB_2.0=0.7547, PB_2.5=0.7707

# DO NOT SET ITERATION_LIMIT
StructuralStopState - budget < 2.0:
Dataset 46956:                  PB_0.5=0.7491, PB_1.0=0.7608, PB_1.5=0.7523, PB_2.0=0.7624, PB_2.5=0.7713

# DO NOT SET ITERATION_LIMIT
StructuralStopState - budget < 0.5: 
Dataset 46956:                  PB_0.5=0.7521, PB_1.0=0.7593, PB_1.5=0.7561, PB_2.0=0.7624, PB_2.5=0.7713


Evaluating Task ID: 46956
Loading Dataset ID 46956 (from Task 46956)...
Data shape: (2584, 15)
Seed 0 - CatBoost: 0.7339, CPU: 5193.06s, Wall: 2657.54s
Seed 0 - Perpetual CV (budget=0.5): 0.7519
Seed 0 - Perpetual CV (budget=1.0): 0.7721
Seed 0 - Perpetual CV (budget=1.5): 0.7736
Seed 0 - Perpetual CV (budget=2.0): 0.7720
Seed 0 - Perpetual CV (budget=2.5): 0.7708
Seed 0 - Perpetual best budget=1.5: CV=0.7736, Test=0.6891, CPU: 17.52s, Wall: 6.78s
Loading Dataset ID 46956 (from Task 46956)...
Data shape: (2584, 15)
Seed 1 - CatBoost: 0.7813, CPU: 3778.50s, Wall: 2313.68s
Seed 1 - Perpetual CV (budget=0.5): 0.7270
Seed 1 - Perpetual CV (budget=1.0): 0.7411
Seed 1 - Perpetual CV (budget=1.5): 0.7394
Seed 1 - Perpetual CV (budget=2.0): 0.7500
Seed 1 - Perpetual CV (budget=2.5): 0.7457
Seed 1 - Perpetual best budget=2.0: CV=0.7500, Test=0.7895, CPU: 18.23s, Wall: 6.19s
Loading Dataset ID 46956 (from Task 46956)...
Data shape: (2584, 15)
Seed 2 - CatBoost: 0.7926, CPU: 3518.80s, Wall: 2120.20s
Seed 2 - Perpetual CV (budget=0.5): 0.7430
Seed 2 - Perpetual CV (budget=1.0): 0.7671
Seed 2 - Perpetual CV (budget=1.5): 0.7611
Seed 2 - Perpetual CV (budget=2.0): 0.7641
Seed 2 - Perpetual CV (budget=2.5): 0.7719
Seed 2 - Perpetual best budget=2.5: CV=0.7719, Test=0.8183, CPU: 19.28s, Wall: 6.75s
Loading Dataset ID 46956 (from Task 46956)...
Data shape: (2584, 15)
Seed 3 - CatBoost: 0.7282, CPU: 3405.52s, Wall: 1843.29s
Seed 3 - Perpetual CV (budget=0.5): 0.7425
Seed 3 - Perpetual CV (budget=1.0): 0.7713
Seed 3 - Perpetual CV (budget=1.5): 0.7685
Seed 3 - Perpetual CV (budget=2.0): 0.7711
Seed 3 - Perpetual CV (budget=2.5): 0.7689
Seed 3 - Perpetual best budget=1.0: CV=0.7713, Test=0.7717, CPU: 19.77s, Wall: 6.93s
Loading Dataset ID 46956 (from Task 46956)...
Data shape: (2584, 15)
Seed 4 - CatBoost: 0.7539, CPU: 2376.50s, Wall: 1441.93s
Seed 4 - Perpetual CV (budget=0.5): 0.7460
Seed 4 - Perpetual CV (budget=1.0): 0.7610
Seed 4 - Perpetual CV (budget=1.5): 0.7658
Seed 4 - Perpetual CV (budget=2.0): 0.7744
Seed 4 - Perpetual CV (budget=2.5): 0.7731
Seed 4 - Perpetual best budget=2.0: CV=0.7744, Test=0.7538, CPU: 19.03s, Wall: 5.77s

==================================================
FINAL SUMMARY (Averaged over 5 seeds)
==================================================
Dataset 46956: CatBoost=0.7580, PB(test)=0.7645, PB(CV score)=0.7683, PB(best budget avg)=1.80


Evaluating Task ID: 46962
Loading Dataset ID 46962 (from Task 46962)...
Data shape: (6819, 94)
Seed 0 - CatBoost: 0.9294, CPU: 26482.53s, Wall: 4417.47s
Seed 0 - Perpetual CV (budget=0.5): 0.9315
Seed 0 - Perpetual CV (budget=1.0): 0.9507
Seed 0 - Perpetual CV (budget=1.5): 0.9503
Seed 0 - Perpetual CV (budget=2.0): 0.9501
Seed 0 - Perpetual CV (budget=2.5): 0.9484
Seed 0 - Perpetual best budget=1.0: CV=0.9507, Test=0.9338, CPU: 13021.89s, Wall: 2424.28s
Loading Dataset ID 46962 (from Task 46962)...
Data shape: (6819, 94)
Seed 1 - CatBoost: 0.9528, CPU: 8262.08s, Wall: 1073.73s
Seed 1 - Perpetual CV (budget=0.5): 0.9317
Seed 1 - Perpetual CV (budget=1.0): 0.9382
Seed 1 - Perpetual CV (budget=1.5): 0.9379
Seed 1 - Perpetual CV (budget=2.0): 0.9425
Seed 1 - Perpetual CV (budget=2.5): 0.9391
Seed 1 - Perpetual best budget=2.0: CV=0.9425, Test=0.9535, CPU: 7514.67s, Wall: 1250.49s
Loading Dataset ID 46962 (from Task 46962)...
Data shape: (6819, 94)
Seed 2 - CatBoost: 0.9546, CPU: 8852.00s, Wall: 1229.24s
Seed 2 - Perpetual CV (budget=0.5): 0.9225
Seed 2 - Perpetual CV (budget=1.0): 0.9381
Seed 2 - Perpetual CV (budget=1.5): 0.9387
Seed 2 - Perpetual CV (budget=2.0): 0.9389
Seed 2 - Perpetual CV (budget=2.5): 0.9378
Seed 2 - Perpetual CV (budget=2.5): 0.9378
Seed 2 - Perpetual best budget=2.0: CV=0.9389, Test=0.9590, CPU: 9437.67s, Wall: 1530.59s
Loading Dataset ID 46962 (from Task 46962)...
Data shape: (6819, 94)
Seed 3 - CatBoost: 0.9359, CPU: 13050.34s, Wall: 1737.22s
Seed 3 - Perpetual CV (budget=0.5): 0.9196
Seed 3 - Perpetual CV (budget=1.0): 0.9444
Seed 3 - Perpetual CV (budget=1.5): 0.9426
Seed 3 - Perpetual CV (budget=2.0): 0.9445
Seed 3 - Perpetual CV (budget=2.5): 0.9436
Seed 3 - Perpetual best budget=2.0: CV=0.9445, Test=0.9298, CPU: 10488.19s, Wall: 1712.49s
Loading Dataset ID 46962 (from Task 46962)...
Data shape: (6819, 94)
Seed 4 - CatBoost: 0.9444, CPU: 67732.55s, Wall: 10211.18s
Seed 4 - Perpetual CV (budget=0.5): 0.9271
Seed 4 - Perpetual CV (budget=1.0): 0.9441
Seed 4 - Perpetual CV (budget=1.5): 0.9381
Seed 4 - Perpetual CV (budget=2.0): 0.9379
Seed 4 - Perpetual CV (budget=2.5): 0.9368
Seed 4 - Perpetual best budget=1.0: CV=0.9441, Test=0.9458, CPU: 9515.03s, Wall: 2417.26s

==================================================
FINAL SUMMARY (Averaged over 5 seeds)
==================================================
Dataset 46962: CatBoost=0.9434, PB(test)=0.9444, PB(CV score)=0.9441, PB(best budget avg)=1.60



row subsampling and max_cat=1000:

Evaluating Task ID: 46955
Loading Dataset ID 46955 (from Task 46955)...
Data shape: (78053, 11)
Seed 0 - CatBoost: 0.0769, CPU: 16382.72s, Wall: 3170.04s
Seed 0 - Perpetual CV (budget=0.5): -0.1410
Seed 0 - Perpetual CV (budget=1.0): -0.1125
Seed 0 - Perpetual CV (budget=1.5): -0.1413
Seed 0 - Perpetual CV (budget=2.0): -0.3378
Seed 0 - Perpetual CV (budget=2.5): -0.8828
Seed 0 - Perpetual best budget=1.0: CV=-0.1125, Test=0.1134, CPU: 152.23s, Wall: 50.51s


no row subsampling and max_cat=9000:

Evaluating Task ID: 46955
Loading Dataset ID 46955 (from Task 46955)...
Data shape: (78053, 11)
Seed 0 - CatBoost: 0.0769, CPU: 23710.47s, Wall: 4679.17s
Seed 0 - Perpetual CV (budget=0.5): -0.3614
Seed 0 - Perpetual CV (budget=1.0): -0.2878
Seed 0 - Perpetual CV (budget=1.5): -0.2289
Seed 0 - Perpetual CV (budget=2.0): -0.1945
Seed 0 - Perpetual CV (budget=2.5): -0.1684
Seed 0 - Perpetual best budget=2.5: CV=-0.1684, Test=0.1782, CPU: 1871.59s, Wall: 1351.24s
Loading Dataset ID 46955 (from Task 46955)...
Data shape: (78053, 11)


no row subsampling and max_cat=1000:

Evaluating Task ID: 46955
Loading Dataset ID 46955 (from Task 46955)...
Data shape: (78053, 11)
Seed 0 - CatBoost: 0.0769, CPU: 20905.67s, Wall: 3693.78s
Seed 0 - Perpetual CV (budget=0.5): -0.1410
Seed 0 - Perpetual CV (budget=1.0): -0.1190
Seed 0 - Perpetual CV (budget=1.5): -0.1037
Seed 0 - Perpetual CV (budget=2.0): -0.1687
Seed 0 - Perpetual CV (budget=2.5): -0.1788
Seed 0 - Perpetual best budget=1.5: CV=-0.1037, Test=0.1558, CPU: 325.73s, Wall: 167.95s
"""
