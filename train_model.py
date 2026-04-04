import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')





import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# ─── Load Data ─────────────────────────────────────────────
base_path = os.path.dirname(__file__)
train = pd.read_csv(os.path.join(base_path, "train.csv"))
test  = pd.read_csv(os.path.join(base_path, "test.csv"))

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Columns: {list(train.columns)}")

print("\nPlacement distribution:")
print(train['Placement_Status'].value_counts())


# ─── Preprocessing ─────────────────────────────────────────
drop_cols = ['Student_ID']
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols, inplace=True, errors='ignore')

cat_cols = ['Gender', 'Degree', 'Branch']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col]  = le.transform(test[col])
    encoders[col] = le

target_enc = LabelEncoder()
train['Placement_Status'] = target_enc.fit_transform(train['Placement_Status'])

print("\nLabel encoding:")
print(dict(zip(target_enc.classes_, target_enc.transform(target_enc.classes_))))


# ─── Feature & Target ─────────────────────────────────────
feature_cols = [c for c in train.columns if c != 'Placement_Status']

X_train = train[feature_cols]
y_train = train['Placement_Status']

if 'Placement_Status' in test.columns:
    test['Placement_Status'] = target_enc.transform(test['Placement_Status'])
    X_test = test[feature_cols]
    y_test = test['Placement_Status']
else:
    X_test = test[feature_cols]
    y_test = None


# ─── Leakage Check (IMPORTANT) ─────────────────────────────
print("\n🔍 Correlation with Target:")
print(train.corr()['Placement_Status'].sort_values(ascending=False))


# ─── Models ───────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
}

results = []
trained_models = {}

print("\n" + "="*60)
print("Training Models...")
print("="*60)


# ─── Training Loop ─────────────────────────────────────────
for name, model in models.items():

    model.fit(X_train, y_train)
    trained_models[name] = model

    # Train Accuracy (Overfitting Check)
    train_acc = accuracy_score(y_train, model.predict(X_train))

    if y_test is not None:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        auc = roc_auc_score(y_test, proba)
    else:
        preds = model.predict(X_train)
        proba = model.predict_proba(X_train)[:, 1]
        acc = train_acc
        f1 = f1_score(y_train, preds, average='weighted')
        auc = roc_auc_score(y_train, proba)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Train_Accuracy': train_acc,
        'F1': f1,
        'AUC': auc,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std()
    })

    print(f"{name:<22} Train: {train_acc:.4f} | Test: {acc:.4f} | CV: {cv_scores.mean():.4f}")


# ─── Results DataFrame ─────────────────────────────────────
results_df = pd.DataFrame(results)

# Smart sorting
results_df = results_df.sort_values(
    by=['Accuracy', 'CV_Mean', 'AUC'],
    ascending=False
)

print("\n📊 Model Comparison:")
print(results_df)


# ─── Smart Best Model Selection ────────────────────────────
top_acc = results_df.iloc[0]['Accuracy']
top_models = results_df[results_df['Accuracy'] == top_acc]

if 'Random Forest' in top_models['Model'].values:
    best_name = 'Random Forest'
else:
    best_name = top_models.iloc[0]['Model']

best_model = trained_models[best_name]

print(f"\n🏆 Best Model Selected: {best_name}")


# ─── Save Files ────────────────────────────────────────────
joblib.dump(best_model, os.path.join(base_path, 'best_model.pkl'))
joblib.dump(encoders, os.path.join(base_path, 'label_encoders.pkl'))
joblib.dump(target_enc, os.path.join(base_path, 'target_encoder.pkl'))
joblib.dump(feature_cols, os.path.join(base_path, 'feature_cols.pkl'))

print("✅ Model saved")


# ─── Feature Importance ───────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
else:
    fi = None


# ─── Plot Report ───────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0d1117')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = ['#4fc3f7','#81c784','#ffb74d','#e57373','#ce93d8','#80cbc4']
TEXT_C = '#e6edf3'
GRID_C = '#30363d'
BG = '#161b22'


def style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT_C)
    ax.tick_params(colors=TEXT_C)
    ax.grid(True, color=GRID_C)


# Accuracy Plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(results_df['Model'], results_df['Accuracy']*100, color=COLORS)
ax1.set_ylim(80, 101)
style(ax1, "Accuracy")


# CV Plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(results_df['Model'], results_df['CV_Mean']*100, color=COLORS)
style(ax2, "Cross Validation")


# Feature Importance
ax3 = fig.add_subplot(gs[0, 2])
if fi is not None:
    ax3.barh(fi.index, fi.values*100, color=COLORS[0])
style(ax3, "Feature Importance")


# Confusion Matrix
ax4 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test if y_test is not None else y_train,
                      best_model.predict(X_test if y_test is not None else X_train))
sns.heatmap(cm, annot=True, fmt='d', ax=ax4)
style(ax4, "Confusion Matrix")


# Save Plot
plt.savefig(os.path.join(base_path, 'ml_report.png'), dpi=150, bbox_inches='tight')
plt.close()

print("✅ Report saved")
