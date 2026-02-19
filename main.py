import pandas as pd
df = pd.read_csv("data/data.csv")
df = pd.get_dummies(df)

from sklearn.model_selection import train_test_split
X = df.drop(columns=["isFraud"])
y = df["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

from xgboost import XGBClassifier
model = XGBClassifier(
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        eval_metric='aucpr',
        random_state=42
)

from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [3,5,7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
}
grid = GridSearchCV(
    model,
    param_grid=param_grid,
    cv=5,
    verbose=2
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_proba[:,1]))

import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)