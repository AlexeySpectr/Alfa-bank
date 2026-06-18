import joblib
import pandas as pd
model = joblib.load("model.pkl")
df_feature=pd.read_csv('test_apps.csv',sep=',')
idx=df_feature['front_id']
df_feature['cnt_deb_ul_ip_90'] = df_feature['cnt_deb_ul_ip_90'].fillna(df_feature['cnt_deb_ul_ip_90'].median())
df_feature['cnt_cred_loan_90_flag']=df_feature['cnt_cred_loan_90'].isna().astype(int)
df_feature['cnt_cred_loan_90'] = df_feature['cnt_cred_loan_90'].fillna(df_feature['cnt_cred_loan_90'].median())
df_feature['fl_hdb_bki_total_active_products'] = df_feature['fl_hdb_bki_total_active_products'].fillna(df_feature['fl_hdb_bki_total_active_products'].median())
df_feature['balance_rur_amt_30_min_flag']=df_feature['balance_rur_amt_30_min'].isna().astype(int)
df_feature['balance_rur_amt_30_min']=df_feature['balance_rur_amt_30_min'].fillna(0)
df_feature['offered_rate'] = df_feature['offered_rate'].mask(df_feature['offered_rate'] < -50,-9999)
df_feature['offered_rate'] = df_feature['offered_rate'].mask(df_feature['offered_rate'] > 50,9999)
df_feature['decision_day'] = pd.to_datetime(df_feature['decision_day'],errors='coerce')
df_feature['decision_day_flag'] = (df_feature['decision_day'].dt.day >= 25).astype(int)
df_feature['decision_weekend'] = (df_feature['decision_day'].dt.dayofweek >= 5).astype(int)
df_feature['count_all_corp_dashboard_events'] = df_feature['count_all_corp_dashboard_events'].fillna(0)
df_feature=df_feature.drop(columns=['front_id','decision_day',
                                    'corp_credit_products','sum_deb_ul_90',
                                    'sum_deb_ul_30','cnt_deb_loan_90',
                                    'sum_deb_ul_90','cnt_deb_ul_ip_30',
                                    'loan_rev_max_start_non_fin','loan_rev_min_start_fin',
                                    'app_term_mean_360','overdraft_app_term_max_360','days_from_authperson_registration',
                                    'corp_list',
                                    'p75_time_spent_minutes','sum_deb_investment_90'],axis=1)

num_features = [
    'loan_amount_last', 'overdraft_limit_min', 'overdraft_limit_max',
    'offered_rate', 'cb_rate', 'cnt_deb_ul_ip_90', 'balance_rur_amt_30_min',
    'cnt_cred_loan_90', 'fl_hdb_bki_total_active_products',
    'count_all_corp_dashboard_events'
]
cat_features = ['fl_adminarea', 'db_group_last']
flag_features = ['decision_weekend']

X = df_feature[num_features + cat_features + flag_features]

df_out = pd.DataFrame()
df_out['front_id'] = idx
df_out['target_value'] = model.predict_proba(df_feature)[:, 1]
df_out.to_csv('submission.csv', index=False)
print("Готово!")