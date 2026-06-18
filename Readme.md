# Практическое задание от Альфа-Банка.          Отклик на кредитный оффер
## Данный репозитории содержит решение задачи от Альфа-Банка.
### Суть задачи состоит в том, чтобы  предсказать вероятность того, что корпоративный клиент согласится на предложенные условия кредитного продукта. Необходимо разработать модель, которая на основании параметров заявки, условий оффера, характеристик клиента и его финансовой активности поможет банку выбрать наиболее привлекательное кредитное предложение 

Начнем с данных ,которые нам предоставили для разработки. Это `train_apps.csv` для обучения нашей модели,`test_apps.csv` для проверки нашей модели ,а также `sample_submission.csv` ,который мы должны получить на выходе нашего приложения.Также нам данны описание наших колонок ,взглянем же на них.



## Таблица переменных датасета

### 1. Идентификатор и дата
| Название | Описание |
|----------|----------|
| `front_id` | Уникальный идентификатор заявки |
| `decision_day` | День принятия решения |

### 2. Параметры кредитного предложения
| Название | Описание |
|----------|----------|
| `loan_amount_last` | Запрошенный лимит по кредиту в текущей заявке |
| `overdraft_limit_min` | Минимальный лимит овердрафта по скорингу |
| `overdraft_limit_max` | Максимальный лимит овердрафта по скорингу |
| `offered_rate` | Предложенная процентная ставка |
| `cb_rate` | Ключевая ставка ЦБ на момент заявки |

### 3. Финансовые потоки (дебетовые операции)
| Название | Описание |
|----------|----------|
| `sum_deb_ul_90` | Сумма переводов юрлицам за 90 дней |
| `sum_deb_ul_30` | Сумма переводов юрлицам за 30 дней |
| `cnt_deb_loan_90` | Количество выплат по кредитам за 90 дней |
| `cnt_deb_ul_ip_90` | Количество переводов юрлицам и ИП за 90 дней |
| `cnt_deb_ul_ip_30` | Количество переводов юрлицам и ИП за 30 дней |
| `sum_deb_investment_90` | Сумма инвестиций и вкладов за 90 дней |

### 4. Состояние счетов и активность
| Название | Описание |
|----------|----------|
| `balance_rur_amt_30_min` | Минимальный остаток на рублёвых счетах за месяц |
| `corp_credit_products` | Количество событий в корпоративных кредитных продуктах |
| `corp_list` | Количество событий в `corp_list` |
| `count_all_corp_dashboard_events` | Количество действий в корпоративном интернет-банке |
| `p75_time_spent_minutes` | 75-й перцентиль времени в мобильном/интернет-банке за месяц |
| `days_from_authperson_registration` | Дней с регистрации как управляющего лица |

### 5. Кредитная история и сроки
| Название | Описание |
|----------|----------|
| `cnt_cred_loan_90` | Количество полученных кредитов/займов за 90 дней |
| `fl_hdb_bki_total_active_products` | Количество активных продуктов по данным БКИ |
| `loan_rev_max_start_non_fin` | Месяцев до макс. даты начала по действующим оборотным кредитам |
| `loan_rev_min_start_fin` | Месяцев до мин. даты начала по закрытым оборотным кредитам |
| `app_term_mean_360` | Средний срок заявок за 360 дней |
| `overdraft_app_term_max_360` | Максимальный срок заявок на овердрафт за 360 дней |

### 6. Категориальные признаки
| Название | Описание |
|----------|----------|
| `db_group_last` | Тип последнего кредитного продукта в заявке |
| `fl_adminarea` | Регион регистрации клиента |

### 7. Целевая переменная
| Название | Описание |
|----------|----------|
| `target_value` | 1 – клиент согласился, 0 – отказался |

Также стоит предупредить ,что в данном репозитории присутствует различные ноутбки ,поскольку для меня это как холст и когда что-то приходило в голову начинал с нового холста-ноутбука ,прощу строго не судить за такое безобразие,каждый раз буду пытаться приводить все в более приятный вид. Лирическое отступление закончилось,пожалуй приступим к делу.

Первым на очереди файл `App.ipynb` <sup>(про название ноутбуков лучше и не спрашивать)</sup>

Сначала мы прочитали csv файл и посмотрели информацию об это датасете.

```python
df=pd.read_csv('train_apps.csv',sep=',')
df.info()
df.dtypes
```

Мы видим ,что все колонки кроме  `fl_adminarea` , `db_group_last` и `decision_day` являются числовыми в отличие от приведённых выше.Это означает ,что эти признаки придется подготовить ,чтобы продолжить дальше.Глядя чуть вперед могу сказать ,что изначально `decision_day` не был мною использовал и не учитывался на обучение ,но вдругом ноутбуке его решено было использовать ,позже объясню почему.

Дальше мы берем необходимые метки для нашей задачи классификации.
```python
y=df['target_value']
y.value_counts()
```
Также при выводе комнады `value_counts()` замечаем,что наша выборка несбалансирована ,тут мы все оставим как есть ,но в другом ноутбуке попробуем это изменить.

Пора заняться нашими категориальными признаки ,мы не можем их просто так отдать на обучение нашему алгоритму ,кроме некоторых исключений об этом поговорим позже.Тут же бы используем `OneHotEncoder` для кодирования данных признаков и заполнения пропусков значением `Unknown`

```python
X_features_str=df[['db_group_last']]
X_features_str.value_counts()
X_features_str_db_group_last=df[['db_group_last']].fillna('Unknown')
Encoder=OneHotEncoder(drop='first',handle_unknown='ignore')
Encoder.fit(X_features_str_db_group_last)
X_features_str_db_group_last=Encoder.transform(X_features_str_db_group_last)
table_db_group_last=pd.DataFrame.sparse.from_spmatrix(data=X_features_str_db_group_last,columns=Encoder.get_feature_names_out())
table_db_group_last
X_features_str_fl_adminarea=df[['fl_adminarea']]
X_features_str_fl_adminarea.value_counts()
X_features_str_fl_adminarea=X_features_str_fl_adminarea[['fl_adminarea']].fillna('Unknown')
Encoder_f1=OneHotEncoder(drop='first',handle_unknown='ignore')
Encoder_f1.fit(X_features_str_fl_adminarea)
X_features_str_fl_adminarea=Encoder_f1.transform(X_features_str_fl_adminarea)
X_features_str_fl_adminarea
table_fl_adminarea=pd.DataFrame.sparse.from_spmatrix(X_features_str_fl_adminarea,columns=Encoder_f1.get_feature_names_out())
table_fl_adminarea

```
Далее мы изучим на графиках выбранные мною числовые признаки и как они влияют на нашу метку.

```python
X=df[['loan_amount_last','overdraft_limit_min','overdraft_limit_max','offered_rate','cb_rate']]
features = ['loan_amount_last','overdraft_limit_min','overdraft_limit_max','offered_rate','cb_rate']
for col in features:
    sns.histplot(data=df,x=col,hue='target_value')
    plt.show()
```

Признаки выбранные мною так или иначе влияют на метку ,но некоторые из них имеют значения ,которые вообще не могу существовать ,а именно речь про `offered_rate`<sup>не знаю насколько я прав,все вопросу к поисковику</sup> ,который не может быть больше 50.Поэтому мы обрабатываем этот признак следующим образом.
 
```python
X['offered_rate'] = X['offered_rate'].mask(X['offered_rate'] < -50,-9999)
X['offered_rate'] = X['offered_rate'].mask(X['offered_rate'] > 50,9999))
```
После чего соединяем все таблицы в одну.
```python
X_features_after=pd.concat([X,table_fl_adminarea,table_db_group_last],axis=1)
```
Далее мы используя `train_test_split` и `StandardScaler` делим наши признаки на тестовые и обучащие и нормализуем их.Конечно не самым лучше образом через цикл,мне было интересно насколько сильно это влияет на результат ,повлияло не особо сильно и в одном из следующих ноутбуков такого уже не было ,поскольку все-таки делать не очень то и правильно.

```python
X_features_after = X_features_after.astype(float)
X_train,X_test,y_train,y_test=train_test_split(X_features_after,y,test_size=0.3)
scaler=StandardScaler()
labels_features=['loan_amount_last','overdraft_limit_min','overdraft_limit_max','cb_rate']
for i in labels_features:
    scaler.fit(X_train[[i]])
    X_train[i]=scaler.transform(X_train[[i]])
    X_test[i]=scaler.transform(X_test[[i]])
```

После чего было выбрано 2 модели ,которые мы протестировали это `GradientBoosting` и `RandomForest` ,чтобы сравить из результаты и потом уже подбирать параметры.`GradientBoosting` показал себя лучше чем `RandomForest` .Подобрав параметры с помощью `RandomizedSearchCV` ,мы сохранили результат в файл и отправили его на проверку ,результат оказался слишком хорошим и после чего снова начали поиски решения для улучшения модели.Перейдем к следующему  ноутбуку.

В `App2.ipynb` было решено расширить число признаков и были выбраны следующие признаки:
`cnt_deb_ul_ip_90`,`cnt_cred_loan_90`,`fl_hdb_bki_total_active_products`

После чего с помощью комнад `isnull().sum()` мы видим пропущенные значения в этих колонках ,нам необходимо их заполнить ,чтобы наша модель смогла нормально обучиться на данных.Мы заполним их `median()` ,который устойчив к выбросам ,а также добавими для некоторых флаг ,чтобы увеличить количество признаков ,а также чтобы модель могла обрщать значение на такие пропуски.


```python
df_feature['cnt_deb_ul_ip_90'] = df_feature['cnt_deb_ul_ip_90'].fillna(df_feature['cnt_deb_ul_ip_90'].median())
df_feature['cnt_cred_loan_90_flag']=df_feature['cnt_cred_loan_90'].isna().astype(int)
df_feature['cnt_cred_loan_90'] = df_feature['cnt_cred_loan_90'].fillna(df_feature['cnt_cred_loan_90'].median())
df_feature['fl_hdb_bki_total_active_products'] = df_feature['fl_hdb_bki_total_active_products'].fillna(df_feature['fl_hdb_bki_total_active_products'].median())
```
Далее все идет точно также как и в предыдущем ноутбуке ,кроме нормализации признаков и oversampling.Нормализаци прошла в этот раз без цикла,уже хорошо , а oversampling был использован с целью выравнивания наших меток для обучения ,чтобы не было перевеса ,но как оказался увелечение большинством ухудшило значние AUC-ROC и поэтому пришлось от него отказаться ,но после прочтения пару статей выяснилось ,что еще существует undersampling ,но он уже будет использован далее. 

```python
scaler=StandardScaler()
labels_features=['loan_amount_last','overdraft_limit_min','overdraft_limit_max','cb_rate',
                 'cnt_deb_ul_ip_90',
                 'cnt_cred_loan_90',
                 'fl_hdb_bki_total_active_products']
X_features_after[labels_features] = scaler.fit_transform(X_features_after[labels_features])
X_features_after
```

Следующим ноутбук у нас это `Feature_enginnering.ipynb`,как понятно из названия здесь мы попытались получить новые признаки путем различных мат.операции.

Но при вызове метода 
```python
df['rate_spread'] = df['offered_rate'] - df['cb_rate']
df['rate_ratio'] = df['offered_rate'] / df['cb_rate']
df['od_limit_avg'] = (df['overdraft_limit_min'] + df['overdraft_limit_max']) / 2
df['loan_to_od_max'] = df['loan_amount_last'] / df['overdraft_limit_max']
df['loan_to_od_avg'] = df['loan_amount_last'] / df['od_limit_avg']
```
И тут впервые начали использовать pipeline ,и стало намного проще.
```python
preprocess = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    
])
pipe = Pipeline([
    ('prep', preprocess),
    ('model', GradientBoostingClassifier(subsample=1.0,n_estimators=300,
                                         min_samples_leaf=1,
                                         max_features='log2',
                                         min_samples_split=5,
                                         max_depth=4,
                                         learning_rate=0.05))
])


param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 4, 5],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 5],
    'model__max_features': ['sqrt', 'log2', None]
}
grid = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=3,
    verbose=2
)


cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    n_iter=30,
    cv=cv,
    scoring='roc_auc',
    n_jobs=4,
    random_state=42,
    verbose=2
)
``` 
Но результат стал хуже чем был и пришлось искать решения дальше. 

Следующий ноутбук `feature.ipynb` это результат изучения довольно старой статьи на Kaggle про [AUC-ROC 0.99](https://www.kaggle.com/code/yejiseoung/building-gradient-boosting-pipeline-0-99-roc-auc).Здесь я и узнал про oversimpling ,который в моей задаче не подошел.Но и данный анализ результатов больших не выдал и пришлось снова искать решения лучшего значени AUC-ROC кривой.Из нового немного другой pipeline,который работает с oversimpling.


```python
model_pipe = imbPipeline([
    
    ("scale", StandardScaler()),
    ("oversampling", ADASYN(sampling_strategy='auto', random_state=0, n_neighbors=5)),
    ("gbm", GradientBoostingClassifier(**best_param))
])
``` 
Перейдем к одноименному ноутбуку и понятному из названия о чем будет в нем речь ,а именно `Catboost.ipynb`.Все правильно поняли ,здесь мы будем работать с библиотекой `Catboost` , которая умеет обрабатывать категориальные признаки.Тут также обработали признаки ,как и раньше кроме обработки категориальных признаков ,как упоминалось ранее данная бибилиотека спокойно работает с данными признаками без их предварительной обработки.

```python

cat_features = ['db_group_last', 'fl_adminarea']


df_feature = df_feature.copy()
df_feature[cat_features] = (
    df_feature[cat_features]
    .astype(str)
    .fillna("missing")
)


model = CatBoostClassifier(
    verbose=0,
    random_seed=42  
)


param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [2, 3, 4, 6],
    'iterations': [300, 500, 800, 1000],
    'l2_leaf_reg': [1, 3, 5],  
    'border_count': [32, 64, 128] 
}


search = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    n_iter=20,  
    cv=5,
    scoring='roc_auc',
    n_jobs=3,
    verbose=2,
    random_state=42
)


search.fit(df_feature, y, cat_features=cat_features)


```

Результат также оказался не впечетляющим ,`GradienBoosting` давал лучше показатель AUC-ROC.

Ну и теперь перейдем к завершающим ноутбукам и файлу.А именно `App-test.ipynb` ,`App-f.ipynb` и `App_Finaly.py`

В `App-test.ipynb` были использованы признаки из датасета , а также новые признаки ,которые мы получали из `Feature_enginnering.ipynb` и применения на них `HistGradientBoostingClassifier`,который может работать с пропущенными значениями и работает быстре чем `GradientBoostingClassifier`.

```python
df['decision_day_flag'] = (df['decision_day'].dt.day >= 25).astype(int)
df['decision_weekend'] = (df['decision_day'].dt.dayofweek >= 5).astype(int)


```

Из даты принятия решения мы извлекли новые признаки ,которые могут оказать влияние на принятие оффера.Также использовали `PolynomialFeatures`,чтобы получить новые признаки ,но особого влияние как мы увидим дальше это не оказало.



```python
from sklearn.ensemble import HistGradientBoostingClassifier
param_grid = {
    'model__max_iter': randint(100, 500),         
    'model__learning_rate': [0.02, 0.03, 0.05],
    'model__max_depth': [3, 4, 5],
    'model__min_samples_leaf': randint(10, 60),
    'model__l2_regularization': [0.0, 0.5, 1.0, 2.0],
    'model__max_leaf_nodes': [None, 31, 63, 127],  
}
num_features = [
    'loan_amount_last','overdraft_limit_min',
    'overdraft_limit_max','offered_rate',
    'cb_rate','cnt_deb_ul_ip_90',
    'cnt_cred_loan_90','fl_hdb_bki_total_active_products','count_all_corp_dashboard_events',
    'loan_to_od_max','weighted_spread','app_term_mean_360'
]

cat_features = [
    'fl_adminarea','db_group_last'
]


X = df_feature[num_features +cat_features]

y = df['target_value']

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    
])
pipe = Pipeline([
    ('prep', preprocess),
    ('model', HistGradientBoostingClassifier(
    learning_rate=0.03,
    max_iter=500,
    max_depth=5,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
    verbose=2
))
])
grid = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=3,
    verbose=2,
    n_iter=30,
    random_state=42,    
    error_score='raise'
)

rus = RandomUnderSampler(random_state=42)
X_res_rus, y_res_rus = rus.fit_resample(X, y)
grid.fit(X_res_rus,y_res_rus)
```

И после этого в голову приходит идея ,а что если посмотреть веса признаков и убрать не значимые признаки ,поскольку они не несут большого значения ,а могу влиять на конечный результат.


```python
from sklearn.inspection import permutation_importance
result = permutation_importance(
    best_model, X_res_rus, y_res_rus,
    n_repeats=5,
    random_state=42,
    scoring='roc_auc'
)

importances = result.importances_mean
feature_names = X_res_rus.columns  

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feat_imp.head(40))
```
| Признак                                | Значение  |
|----------------------------------------|-----------|
| `loan_amount_last`                     | 0.051624  |
| `cb_rate`                              | 0.032069  |
| `cnt_cred_loan_90`                     | 0.028342  |
| `count_all_corp_dashboard_events`      | 0.028214  |
| `db_group_last`                        | 0.019192  |
| `offered_rate`                         | 0.012310  |
| `overdraft_limit_max`                  | 0.012092  |
| `overdraft_limit_min`                  | 0.010587  |
| `balance_rur_amt_30_min`               | 0.010427  |
| `cnt_deb_ul_ip_90`                     | 0.008334  |
| `app_term_mean_360`                    | 0.007916  |
| `loan_to_od_max`                       | 0.006841  |
| `fl_adminarea`                         | 0.004648  |
| `fl_hdb_bki_total_active_products`     | 0.003753  |
| `weighted_spread`                      | 0.002399  |
| `cnt_cred_loan_90_flag`                | 0.001365  |
| `app_term_mean_360^2`                  | 0.000757  |
| `od_range`                             | 0.000728  |
| `is_promo_rate`                        | 0.000243  |
| `balance_rur_amt_30_min_flag`          | 0.000123  |
| `heavy_user`                           | 0.000051  |

Как мы видим,некоторые новые признаки не несут за собой большого значения также как и полученный полиминиальный признак `app_term_mean_360^2`.Далее попробовав их убрать ,решил еще раз обучить модель ,показатель AUC-ROC не вырос,но и не сильно то уменьшился из чего можно сделать вывод,что данные признаки не нужны нашей модели.



А теперь перейдем к `App-f.ipynb`.В нем из изменений это использование нового бустинга , а именно `XGboost`.


```python
num_features = [
    'loan_amount_last', 'overdraft_limit_min', 'overdraft_limit_max',
    'offered_rate', 'cb_rate', 'cnt_deb_ul_ip_90', 'balance_rur_amt_30_min',
    'cnt_cred_loan_90', 'fl_hdb_bki_total_active_products',
    'count_all_corp_dashboard_events'
]
cat_features = ['fl_adminarea', 'db_group_last']
flag_features = ['decision_weekend']

X = df_feature[num_features + cat_features + flag_features]
y = df['target_value']





preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('flag', 'passthrough', flag_features)
])


model = xgb.XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.1,     
    max_depth=6,            
    n_estimators=100,      
    subsample=0.8,          
    colsample_bytree=0.8,   
    reg_alpha=0.1,       
    reg_lambda=1,         
    gamma=0,              
    booster='gbtree',    
    tree_method='auto',    
    scale_pos_weight=1 ,
    verbosity=2            
)

pipe = Pipeline([
    ('prep', preprocess),
    ('model', model)
])

param_grid = {
    'model__n_estimators': randint(100, 1000),
    'model__max_depth': randint(2, 10),
    'model__learning_rate': uniform(0.01, 0.29),      
    'model__min_child_weight': randint(1, 15),
    'model__subsample': uniform(0.5, 0.5),            
    'model__colsample_bytree': uniform(0.5, 0.5),     
    'model__gamma': uniform(0, 5),
    'model__reg_lambda': uniform(0, 10),
    'model__reg_alpha': uniform(0, 5)
   
}


grid = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=3,               
    verbose=2,
    n_iter=30,               
    random_state=42,
    error_score='raise'
)

rus = RandomUnderSampler(random_state=42)
X_res_rus, y_res_rus = rus.fit_resample(X,y)
grid.fit(X_res_rus,y_res_rus)
```

Здесь мы также подбирали параметры для обучения и результат должен получить лучше чем ,с предыдущими вариантами.Еще не отправлял на результат ,но по внутренним проверкам AUC-ROC выше ,чем у других.

А также здесь мы сохранили лучшие параметры нашей модели для дальшего файла `App_Finaly.py`


```python
import joblib

best_model = grid.best_estimator_

joblib.dump(best_model, "model.pkl")
```

Ну и наконец-то файл `App_Finaly.py`.Здесь мы читаем входной файл (он привязан программно ,но если захотеть можно сделать с помощью ввода конкретного пути),обрабатываем признаки(обработка желает конечно лучшего ,нужно было бы сделать через class или хотя бы через функцию ,но время на решение уже заканчивалось) и выводим данные в файл `submission.csv`.На этом все.

Вывод в ходе решения задачи были просмотреты разные библиотеки бустинга и случайного леса,под них подобраны параметры и сравнив стало понятно ,что лучше всего справляется XGBoost.Но возможно некторые признаки были плохо отобраны или обработаны и случайный лес или Catboost справился бы куда лучше.Пока все что мною так или иначе было изучено использовалось в данной задаче.Также возможно библиотека PyTorch помогла бы достичь AUC-ROC высокого значения и поставить меня в 10 лучших значений AUC-ROC,но я пока не закрпелял навыки в Pytorch.И к сожалению время на решение задачи ограниченно ,чтобы быстро повторить Pytorch и решить задачу,теперь и самому интересно насколько высокий AUC-ROC он бы показал или такой же.Вернусь к этому вопросу позже.Прошу строго не ругать за код ,понимаю что он не очень качественный ,но всему можно научиться ,было бы желание.Стоить добавить,что `test_apps.csv` не открывалось до самого конца,хотя к концу решения было ясно,что в нем есть пропущенные данные ,которые в `train_apps.csv` заполнены и для честности я не стал их обработывать ,хотя желание получить на этом чуть больше значение AUC-ROC были.
