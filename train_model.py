
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')   # ⭐ IMPORTANT FIX

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')





from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# ─── Load Data ───────────────────────────────────────────────────────────────
base_path = os.path.dirname(__file__)  # folder where script is located
train = pd.read_csv(os.path.join(base_path, "train.csv"))
test  = pd.read_csv(os.path.join(base_path, "test.csv"))

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Columns: {list(train.columns)}")
print(f"\nPlacement distribution:\n{train['Placement_Status'].value_counts()}")

# ─── Preprocessing ───────────────────────────────────────────────────────────
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
print(f"\nLabel encoding: {dict(zip(target_enc.classes_, target_enc.transform(target_enc.classes_)))}")

feature_cols = [c for c in train.columns if c != 'Placement_Status']
X_train = train[feature_cols]
y_train = train['Placement_Status']

if 'Placement_Status' in test.columns:
    test['Placement_Status'] = target_enc.transform(test['Placement_Status'])
    X_test = test[feature_cols]
    y_test  = test['Placement_Status']
else:
    X_test = test[feature_cols]
    y_test  = None

# ─── Models ──────────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
}

results = []
trained_models = {}

print("\n" + "="*60)
print("Training Models...")
print("="*60)

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

    if y_test is not None:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average='weighted')
        auc   = roc_auc_score(y_test, proba)
    else:
        preds = model.predict(X_train)
        proba = model.predict_proba(X_train)[:, 1]
        acc   = accuracy_score(y_train, preds)
        f1    = f1_score(y_train, preds, average='weighted')
        auc   = roc_auc_score(y_train, proba)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    results.append({
        'Model': name, 'Accuracy': acc, 'F1': f1,
        'AUC': auc, 'CV_Mean': cv_scores.mean(), 'CV_Std': cv_scores.std()
    })
    print(f"{name:<22} Acc: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}  CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

# Best model
best_name = results_df.iloc[0]['Model']
best_model = trained_models[best_name]
print(f"\n🏆 Best Model: {best_name}")

# Save model & encoders
joblib.dump(best_model, os.path.join(base_path, 'best_model.pkl'))
joblib.dump(encoders,   os.path.join(base_path, 'label_encoders.pkl'))
joblib.dump(target_enc, os.path.join(base_path, 'target_encoder.pkl'))
joblib.dump(feature_cols, os.path.join(base_path, 'feature_cols.pkl'))
print("✅ Model saved: best_model.pkl")

# ─── Feature Importance ──────────────────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
else:
    fi = None

# ─── Plot Report ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = ['#4fc3f7','#81c784','#ffb74d','#e57373','#ce93d8','#80cbc4']
BAR_BG = '#161b22'
TEXT_C = '#e6edf3'
GRID_C = '#30363d'

def style_ax(ax, title):
    ax.set_facecolor(BAR_BG)
    ax.set_title(title, color=TEXT_C, fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT_C, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID_C)
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)

# 1. Accuracy bar
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(results_df['Model'], results_df['Accuracy']*100, color=COLORS)
ax1.set_ylim(80, 101)
ax1.set_ylabel('Accuracy %', color=TEXT_C, fontsize=9)
ax1.set_xticklabels(results_df['Model'], rotation=30, ha='right')
for b, v in zip(bars, results_df['Accuracy']):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f'{v*100:.1f}%',
             ha='center', va='bottom', color=TEXT_C, fontsize=7, fontweight='bold')
style_ax(ax1, 'Model Accuracy')

# 2. AUC bar
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(results_df['Model'], results_df['AUC'], color=COLORS)
ax2.set_ylim(0.85, 1.005)
ax2.set_ylabel('AUC-ROC', color=TEXT_C, fontsize=9)
ax2.set_xticklabels(results_df['Model'], rotation=30, ha='right')
for b, v in zip(bars2, results_df['AUC']):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{v:.3f}',
             ha='center', va='bottom', color=TEXT_C, fontsize=7, fontweight='bold')
style_ax(ax2, 'AUC-ROC Score')

# 3. CV Score with error bars
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(results_df['Model'], results_df['CV_Mean']*100,
        yerr=results_df['CV_Std']*100, color=COLORS, capsize=4,
        error_kw={'ecolor': '#ffffff', 'linewidth': 1.5})
ax3.set_ylim(80, 101)
ax3.set_ylabel('CV Accuracy %', color=TEXT_C, fontsize=9)
ax3.set_xticklabels(results_df['Model'], rotation=30, ha='right')
style_ax(ax3, '5-Fold Cross Validation')

# 4. Feature Importance
ax4 = fig.add_subplot(gs[1, 0])
if fi is not None:
    colors_fi = ['#4fc3f7' if i == 0 else '#30363d' for i in range(len(fi))]
    bars4 = ax4.barh(fi.index[::-1], fi.values[::-1]*100, color=colors_fi[::-1])
    ax4.set_xlabel('Importance %', color=TEXT_C, fontsize=9)
    for b, v in zip(bars4, fi.values[::-1]):
        ax4.text(v*100+0.3, b.get_y()+b.get_height()/2, f'{v*100:.1f}%',
                 va='center', color=TEXT_C, fontsize=7)
    style_ax(ax4, f'Feature Importance ({best_name})')
    ax4.xaxis.grid(True, color=GRID_C, linewidth=0.5)
    ax4.yaxis.grid(False)
else:
    ax4.text(0.5, 0.5, 'N/A for\nLogistic Regression',
             ha='center', va='center', color=TEXT_C, transform=ax4.transAxes)
    style_ax(ax4, 'Feature Importance')

# 5. Confusion Matrix of best model
ax5 = fig.add_subplot(gs[1, 1])
if y_test is not None:
    cm = confusion_matrix(y_test, best_model.predict(X_test))
else:
    cm = confusion_matrix(y_train[:5000], best_model.predict(X_train[:5000]))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=target_enc.classes_, yticklabels=target_enc.classes_,
            annot_kws={'color': '#0d1117', 'fontsize': 11, 'fontweight': 'bold'})
ax5.set_title(f'Confusion Matrix ({best_name})', color=TEXT_C, fontsize=11, fontweight='bold', pad=10)
ax5.set_xlabel('Predicted', color=TEXT_C, fontsize=9)
ax5.set_ylabel('Actual', color=TEXT_C, fontsize=9)
ax5.tick_params(colors=TEXT_C, labelsize=8)

# 6. Summary table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
ax6.set_facecolor(BAR_BG)
table_data = [[r['Model'], f"{r['Accuracy']*100:.1f}%", f"{r['F1']:.3f}", f"{r['AUC']:.3f}"]
              for _, r in results_df.iterrows()]
tbl = ax6.table(cellText=table_data,
                colLabels=['Model', 'Acc', 'F1', 'AUC'],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor('#21262d' if row % 2 == 0 else '#161b22')
    cell.set_text_props(color=TEXT_C)
    cell.set_edgecolor(GRID_C)
    if row == 0:
        cell.set_facecolor('#1f6feb')
        cell.set_text_props(color='white', fontweight='bold')
tbl.scale(1, 1.6)
ax6.set_title('Model Comparison', color=TEXT_C, fontsize=11, fontweight='bold', pad=10)

fig.suptitle('Placement Prediction — ML Report', color='#58a6ff',
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig(os.path.join(base_path, 'ml_report.png'), dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✅ Report saved: ml_report.png")
print("\nDone! Files created: best_model.pkl, label_encoders.pkl, target_encoder.pkl, feature_cols.pkl, ml_report.png")
