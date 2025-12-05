"""
ЗЭЭЛИЙН БАТЛАМЖИЙН ТААМАГЛАЛ (LOAN APPROVAL PREDICTION)
Машин сургалтын төсөл

Эх сурвалж: Kaggle - Loan Approval Prediction Dataset
"""

# ======================================================================
# 1. ШААРДЛАГАТАЙ САНГУУД ТАТАХ
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_curve, auc, 
                            precision_recall_curve, f1_score, roc_auc_score)
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

# Монгол хэл дэмжих тохиргоо
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("ЗЭЭЛИЙН БАТЛАМЖИЙН ТААМАГЛАЛ - МАШИН СУРГАЛТЫН ТӨСӨЛ")
print("=" * 80)

# ======================================================================
# 2. ӨГӨГДӨЛ УНШИЖ АВАХ
# ======================================================================

print("\n📊 АЛХАМ 1: ӨГӨГДӨЛ УНШИЖ АВАХ")
print("-" * 80)

try:
    # Өгөгдөл уншиж авах
    train_df = pd.read_csv('loan_train.csv')
    test_df = pd.read_csv('loan_test.csv')
    
    print(f"✓ Сургалтын өгөгдөл татагдлаа: {train_df.shape[0]} мөр, {train_df.shape[1]} багана")
    print(f"✓ Тестийн өгөгдөл татагдлаа: {test_df.shape[0]} мөр, {test_df.shape[1]} багана")
    
    print(f"\n📋 Өгөгдлийн эхний 5 мөр:")
    print(train_df.head())
    
    print(f"\n📋 Багануудын жагсаалт:")
    for i, col in enumerate(train_df.columns, 1):
        print(f"  {i}. {col:30} - {train_df[col].dtype}")
    
except FileNotFoundError:
    print("❌ АЛДАА: loan_train.csv эсвэл loan_test.csv файл олдсонгүй!")
    print("   Файлуудыг Python скриптийн хажууд байрлуулна уу.")
    exit()

# ======================================================================
# 3. ӨГӨГДЛИЙН АНАЛИЗ
# ======================================================================

print("\n" + "=" * 80)
print("📊 АЛХАМ 2: ӨГӨГДЛИЙН АНАЛИЗ")
print("-" * 80)

print("\n📌 Өгөгдлийн мэдээлэл:")
print(train_df.info())

print("\n📌 Статистик үзүүлэлтүүд (тоон хувьсагчид):")
print(train_df.describe())

print("\n📌 Алга болсон утгууд:")
missing = train_df.isnull().sum()
missing_pct = (missing / len(train_df) * 100).round(2)
missing_df = pd.DataFrame({
    'Багана': missing.index,
    'Алга болсон тоо': missing.values,
    'Хувь (%)': missing_pct.values
})
missing_df = missing_df[missing_df['Алга болсон тоо'] > 0].sort_values('Алга болсон тоо', ascending=False)
if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("Алга болсон утга байхгүй")

# Зорилтот хувьсагчийг олох (Loan_Status буюу төстэй нэртэй)
target_col = None
for col in train_df.columns:
    if 'status' in col.lower() or 'approval' in col.lower() or 'loan_status' in col.lower():
        target_col = col
        break

if target_col is None:
    # Сүүлийн баганыг зорилтот хувьсагч гэж үзэх
    target_col = train_df.columns[-1]
    print(f"\n⚠️ Зорилтот хувьсагчийг автоматаар сонгосон: {target_col}")

print(f"\n📌 Зорилтот хувьсагч: {target_col}")
print(f"📌 Зорилтот хувьсагчийн тархалт:")
if train_df[target_col].dtype == 'object':
    status_counts = train_df[target_col].value_counts()
    print(status_counts)
    for status, count in status_counts.items():
        pct = count / len(train_df) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")
else:
    print(train_df[target_col].value_counts())

# ======================================================================
# 4. ӨГӨГДӨЛ ЦЭВЭРЛЭХ БА БОЛОВСРУУЛАЛТ
# ======================================================================

print("\n" + "=" * 80)
print("⚙️ АЛХАМ 3: ӨГӨГДӨЛ ЦЭВЭРЛЭХ БА БОЛОВСРУУЛАЛТ")
print("-" * 80)

# Өгөгдлийн хуулбар үүсгэх
df = train_df.copy()

# ID баганыг устгах (хэрэв байгаа бол)
id_cols = [col for col in df.columns if 'id' in col.lower()]
if id_cols:
    print(f"\n✓ ID багана(ууд) устгагдлаа: {id_cols}")
    df = df.drop(columns=id_cols)

# Зорилтот хувьсагчийг тоон болгох
if df[target_col].dtype == 'object':
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    print(f"\n✓ Зорилтот хувьсагч кодлогдлоо:")
    for i, label in enumerate(le_target.classes_):
        print(f"  {label} → {i}")

# Категори болон тоон хувьсагчдыг ялгах
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Зорилтот хувьсагчийг тоон хувьсагчдаас хасах
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

print(f"\n📊 Категори хувьсагчид ({len(categorical_cols)}):")
for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"  • {col}: {unique_count} өөр утга")

print(f"\n📊 Тоон хувьсагчид ({len(numeric_cols)}):")
for col in numeric_cols:
    print(f"  • {col}")

# Алга болсон утгуудыг бөглөх
print("\n🔧 Алга болсон утгуудыг бөглөж байна...")

# Тоон хувьсагчдын алга болсон утгыг дундаж утгаар бөглөх
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  ✓ {col}: дундаж утгаар ({median_val:.2f})")

# Категори хувьсагчдын алга болсон утгыг модаар (хамгийн түгээмэл утга) бөглөх
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  ✓ {col}: модаар ({mode_val})")

print("\n✓ Алга болсон утгууд бүгд бөглөгдлөө")

# Категори хувьсагчдыг кодлох
print("\n🔧 Категори хувьсагчдыг кодлож байна...")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  ✓ {col}: {len(le.classes_)} категори кодлогдлоо")

# ======================================================================
# 5. ӨГӨГДЛИЙН ВИЗУАЛИЗАЦИ
# ======================================================================

print("\n" + "=" * 80)
print("📈 АЛХАМ 4: ӨГӨГДЛИЙН ВИЗУАЛИЗАЦИ")
print("-" * 80)

# График 1: Зорилтот хувьсагч + тоон хувьсагчдын тархалт
n_plots = min(6, len(numeric_cols) + 1)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

# Зорилтот хувьсагчийн тархалт
status_counts = df[target_col].value_counts()
colors_map = ['#2ecc71', '#e74c3c']
axes[0].bar(range(len(status_counts)), status_counts.values, color=colors_map[:len(status_counts)])
axes[0].set_title(f'{target_col} - Тархалт', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Ангилал')
axes[0].set_ylabel('Тоо')
axes[0].set_xticks(range(len(status_counts)))
if 'le_target' in locals():
    axes[0].set_xticklabels(le_target.classes_, rotation=0)

# Тоон хувьсагчдын тархалт
plot_colors = ['#3498db', '#e67e22', '#9b59b6', '#1abc9c', '#f39c12']
for idx, col in enumerate(numeric_cols[:5]):
    axes[idx+1].hist(df[col].dropna(), bins=30, color=plot_colors[idx % len(plot_colors)], 
                     edgecolor='black', alpha=0.7)
    axes[idx+1].set_title(f'{col} - Тархалт', fontsize=10, fontweight='bold')
    axes[idx+1].set_xlabel(col)
    axes[idx+1].set_ylabel('Давтамж')

plt.tight_layout()
plt.savefig('loan_distributions.png', dpi=300, bbox_inches='tight')
print("✓ График хадгалагдлаа: loan_distributions.png")
plt.close()

# График 2: Корреляцийн матриц
print("\n🔗 Корреляцийн анализ хийж байна...")
correlation_matrix = df.corr()

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, mask=mask,
            cbar_kws={"shrink": 0.8})
plt.title('Хувьсагчдын хоорондын корреляци', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ График хадгалагдлаа: correlation_matrix.png")
plt.close()

print(f"\n📌 {target_col}-тай хамгийн их холбоотой хувьсагчид:")
target_corr = correlation_matrix[target_col].sort_values(ascending=False)
print(target_corr.head(10))

# График 3: Зорилтот хувьсагчтай хамгийн их холбоотой хувьсагчид
top_features = target_corr.abs().sort_values(ascending=False)[1:6]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, (feature, corr_val) in enumerate(top_features.items()):
    if feature in df.columns:
        # Box plot
        df.boxplot(column=feature, by=target_col, ax=axes[idx])
        axes[idx].set_title(f'{feature}\n(Корреляци: {corr_val:.3f})', fontweight='bold')
        axes[idx].set_xlabel('')
        plt.sca(axes[idx])
        plt.xticks(range(1, len(status_counts)+1), 
                  le_target.classes_ if 'le_target' in locals() else range(len(status_counts)))

# Сүүлийн хоосон графикийг нуух
if len(top_features) < 6:
    for idx in range(len(top_features), 6):
        axes[idx].axis('off')

plt.suptitle('Зорилтот хувьсагчтай хамгийн их холбоотой хувьсагчид', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('feature_relationships.png', dpi=300, bbox_inches='tight')
print("✓ График хадгалагдлаа: feature_relationships.png")
plt.close()

# ======================================================================
# 6. ӨГӨГДЛИЙГ СУРГАЛТ БА ТЕСТЭД ХУВААХ
# ======================================================================

print("\n" + "=" * 80)
print("✂️ АЛХАМ 5: ӨГӨГДЛИЙГ ХУВААХ")
print("-" * 80)

# X (шинж) болон y (зорилтот) салгах
X = df.drop(target_col, axis=1)
y = df[target_col]

print(f"✓ Шинжийн тоо: {X.shape[1]}")
print(f"✓ Өгөгдлийн тоо: {X.shape[0]}")

# Сургалт ба валидацийн өгөгдөлд хуваах (80-20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Сургалтын өгөгдөл: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Валидацийн өгөгдөл: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")

# Стандартчлах
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\n✓ Өгөгдөл стандартчлагдлаа (StandardScaler)")

# ======================================================================
# 7. МАШИН СУРГАЛТЫН ЗАГВАРУУД
# ======================================================================

print("\n" + "=" * 80)
print("🤖 АЛХАМ 6: МАШИН СУРГАЛТЫН ЗАГВАРУУД")
print("=" * 80)

# Загваруудын үр дүнг хадгалах
results = {}
predictions = {}
models = {}

# --------------------------------------------------
# 7.1. ЛОГИСТИК РЕГРЕСС
# --------------------------------------------------
print("\n" + "-" * 80)
print("1️⃣ ЛОГИСТИК РЕГРЕСС (LOGISTIC REGRESSION)")
print("-" * 80)

log_reg = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_val_scaled)
y_pred_proba_log = log_reg.predict_proba(X_val_scaled)[:, 1]

acc_log = accuracy_score(y_val, y_pred_log)
f1_log = f1_score(y_val, y_pred_log, average='weighted')
auc_log = roc_auc_score(y_val, y_pred_proba_log) if len(np.unique(y)) == 2 else 0

# Cross-validation
cv_scores_log = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring='accuracy')

results['Логистик регресс'] = {
    'accuracy': acc_log, 
    'f1_score': f1_log,
    'auc': auc_log,
    'cv_mean': cv_scores_log.mean(),
    'cv_std': cv_scores_log.std()
}
predictions['Логистик регресс'] = (y_pred_log, y_pred_proba_log)
models['Логистик регресс'] = log_reg

print(f"✓ Нарийвчлал (Accuracy): {acc_log:.4f}")
print(f"✓ F1-Score: {f1_log:.4f}")
if len(np.unique(y)) == 2:
    print(f"✓ AUC: {auc_log:.4f}")
print(f"✓ Cross-Validation: {cv_scores_log.mean():.4f} (±{cv_scores_log.std():.4f})")

print("\n📊 Дэлгэрэнгүй тайлан:")
print(classification_report(y_val, y_pred_log, 
                          target_names=le_target.classes_ if 'le_target' in locals() else None))

# --------------------------------------------------
# 7.2. ШИЙДВЭРИЙН МОД
# --------------------------------------------------
print("\n" + "-" * 80)
print("2️⃣ ШИЙДВЭРИЙН МОД (DECISION TREE)")
print("-" * 80)

dt_model = DecisionTreeClassifier(random_state=42, max_depth=8, 
                                  min_samples_split=20, min_samples_leaf=10)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_val_scaled)
y_pred_proba_dt = dt_model.predict_proba(X_val_scaled)[:, 1]

acc_dt = accuracy_score(y_val, y_pred_dt)
f1_dt = f1_score(y_val, y_pred_dt, average='weighted')
auc_dt = roc_auc_score(y_val, y_pred_proba_dt) if len(np.unique(y)) == 2 else 0

cv_scores_dt = cross_val_score(dt_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

results['Шийдвэрийн мод'] = {
    'accuracy': acc_dt, 
    'f1_score': f1_dt,
    'auc': auc_dt,
    'cv_mean': cv_scores_dt.mean(),
    'cv_std': cv_scores_dt.std()
}
predictions['Шийдвэрийн мод'] = (y_pred_dt, y_pred_proba_dt)
models['Шийдвэрийн мод'] = dt_model

print(f"✓ Нарийвчлал (Accuracy): {acc_dt:.4f}")
print(f"✓ F1-Score: {f1_dt:.4f}")
if len(np.unique(y)) == 2:
    print(f"✓ AUC: {auc_dt:.4f}")
print(f"✓ Cross-Validation: {cv_scores_dt.mean():.4f} (±{cv_scores_dt.std():.4f})")

print("\n📊 Дэлгэрэнгүй тайлан:")
print(classification_report(y_val, y_pred_dt,
                          target_names=le_target.classes_ if 'le_target' in locals() else None))

# Хувьсагчдын ач холбогдол
feature_importance_dt = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📌 Хувьсагчдын ач холбогдол (эхний 10):")
print(feature_importance_dt.head(10).to_string(index=False))

# --------------------------------------------------
# 7.3. RANDOM FOREST
# --------------------------------------------------
print("\n" + "-" * 80)
print("3️⃣ RANDOM FOREST")
print("-" * 80)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, 
                                 max_depth=10, min_samples_split=20,
                                 min_samples_leaf=10, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_val_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_val_scaled)[:, 1]

acc_rf = accuracy_score(y_val, y_pred_rf)
f1_rf = f1_score(y_val, y_pred_rf, average='weighted')
auc_rf = roc_auc_score(y_val, y_pred_proba_rf) if len(np.unique(y)) == 2 else 0

cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

results['Random Forest'] = {
    'accuracy': acc_rf, 
    'f1_score': f1_rf,
    'auc': auc_rf,
    'cv_mean': cv_scores_rf.mean(),
    'cv_std': cv_scores_rf.std()
}
predictions['Random Forest'] = (y_pred_rf, y_pred_proba_rf)
models['Random Forest'] = rf_model

print(f"✓ Нарийвчлал (Accuracy): {acc_rf:.4f}")
print(f"✓ F1-Score: {f1_rf:.4f}")
if len(np.unique(y)) == 2:
    print(f"✓ AUC: {auc_rf:.4f}")
print(f"✓ Cross-Validation: {cv_scores_rf.mean():.4f} (±{cv_scores_rf.std():.4f})")

print("\n📊 Дэлгэрэнгүй тайлан:")
print(classification_report(y_val, y_pred_rf,
                          target_names=le_target.classes_ if 'le_target' in locals() else None))

# Хувьсагчдын ач холбогдол
feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📌 Хувьсагчдын ач холбогдол (эхний 10):")
print(feature_importance_rf.head(10).to_string(index=False))

# --------------------------------------------------
# 7.4. NAIVE BAYES
# --------------------------------------------------
print("\n" + "-" * 80)
print("4️⃣ NAIVE BAYES")
print("-" * 80)

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_val_scaled)
y_pred_proba_nb = nb_model.predict_proba(X_val_scaled)[:, 1]

acc_nb = accuracy_score(y_val, y_pred_nb)
f1_nb = f1_score(y_val, y_pred_nb, average='weighted')
auc_nb = roc_auc_score(y_val, y_pred_proba_nb) if len(np.unique(y)) == 2 else 0

cv_scores_nb = cross_val_score(nb_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

results['Naive Bayes'] = {
    'accuracy': acc_nb, 
    'f1_score': f1_nb,
    'auc': auc_nb,
    'cv_mean': cv_scores_nb.mean(),
    'cv_std': cv_scores_nb.std()
}
predictions['Naive Bayes'] = (y_pred_nb, y_pred_proba_nb)
models['Naive Bayes'] = nb_model

print(f"✓ Нарийвчлал (Accuracy): {acc_nb:.4f}")
print(f"✓ F1-Score: {f1_nb:.4f}")
if len(np.unique(y)) == 2:
    print(f"✓ AUC: {auc_nb:.4f}")
print(f"✓ Cross-Validation: {cv_scores_nb.mean():.4f} (±{cv_scores_nb.std():.4f})")

print("\n📊 Дэлгэрэнгүй тайлан:")
print(classification_report(y_val, y_pred_nb,
                          target_names=le_target.classes_ if 'le_target' in locals() else None))

# ======================================================================
# 8. ЗАГВАРУУДЫН ХАРЬЦУУЛАЛТ
# ======================================================================

print("\n" + "=" * 80)
print("📊 АЛХАМ 7: ЗАГВАРУУДЫН ХАРЬЦУУЛАЛТ")
print("=" * 80)

# Үр дүнгийн хүснэгт
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('accuracy', ascending=False)

print("\n📌 Загваруудын үр дүнгийн хүснэгт:")
print(results_df.to_string())
print(f"\n🏆 Хамгийн сайн загвар: {results_df.index[0]} "
      f"(Accuracy: {results_df.iloc[0]['accuracy']:.4f})")

# Харьцуулсан график
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy харьцуулалт
axes[0, 0].barh(results_df.index, results_df['accuracy'], color='steelblue')
axes[0, 0].set_xlabel('Нарийвчлал (Accuracy)', fontweight='bold')
axes[0, 0].set_title('Загваруудын нарийвчлал', fontsize=14, fontweight='bold')
axes[0, 0].set_xlim(0, 1)
for i, v in enumerate(results_df['accuracy']):
    axes[0, 0].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

# F1-Score харьцуулалт
axes[0, 1].barh(results_df.index, results_df['f1_score'], color='coral')
axes[0, 1].set_xlabel('F1-Score', fontweight='bold')
axes[0, 1].set_title('Загваруудын F1-Score', fontsize=14, fontweight='bold')
axes[0, 1].set_xlim(0, 1)
for i, v in enumerate(results_df['f1_score']):
    axes[0, 1].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

# AUC харьцуулалт
if len(np.unique(y)) == 2:
    axes[1, 0].barh(results_df.index, results_df['auc'], color='mediumpurple')
    axes[1, 0].set_xlabel('AUC Score', fontweight='bold')
    axes[1, 0].set_title('Загваруудын AUC', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlim(0, 1)
    for i, v in enumerate(results_df['auc']):
        axes[1, 0].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')
else:
    axes[1, 0].axis('off')

# Cross-validation харьцуулалт
axes[1, 1].barh(results_df.index, results_df['cv_mean'], color='lightgreen')
axes[1, 1].set_xlabel('Cross-Validation Score', fontweight='bold')
axes[1, 1].set_title('Cross-Validation (5-fold)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlim(0, 1)
for i, v in enumerate(results_df['cv_mean']):
    axes[1, 1].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ График хадгалагдлаа: model_comparison.png")
plt.close()

# ======================================================================
# 9. ROC МУРУЙ БА AUC
# ======================================================================

if len(np.unique(y)) == 2:
    print("\n" + "=" * 80)
    print("📈 АЛХАМ 8: ROC МУРУЙ БА AUC")
    print("=" * 80)

    plt.figure(figsize=(10, 8))

    colors = ['blue', 'green', 'red', 'purple']
    for (name, (_, y_pred_proba)), color in zip(predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, 
                 label=f'{name} (AUC = {roc_auc:.3f})')
        print(f"✓ {name}: AUC = {roc_auc:.4f}")

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Санамсаргүй (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    plt.title('ROC Муруй - Загваруудын харьцуулалт', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    print("\n✓ График хадгалагдлаа: roc_curves.png")
    plt.close()

# ======================================================================
# 10. CONFUSION MATRIX
# ======================================================================

print("\n" + "=" * 80)
print("🎯 АЛХАМ 9: CONFUSION MATRIX")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, (name, (y_pred, _)) in enumerate(predictions.items()):
    cm = confusion_matrix(y_val, y_pred)
    
    labels = le_target.classes_ if 'le_target' in locals() else [str(i) for i in range(len(np.unique(y)))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Тоо'})
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Бодит утга', fontweight='bold')
    axes[idx].set_xlabel('Таамагласан утга', fontweight='bold')
    
    # Нарийвчлал нэмж харуулах
    acc = accuracy_score(y_val, y_pred)
    axes[idx].text(0.5, -0.15, f'Accuracy: {acc:.4f}', 
                   transform=axes[idx].transAxes, ha='center',
                   fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ График хадгалагдлаа: confusion_matrices.png")
plt.close()

# ======================================================================
# 11. ХУВЬСАГЧДЫН АЧ ХОЛБОГДОЛ
# ======================================================================

print("\n" + "=" * 80)
print("🔍 АЛХАМ 10: ХУВЬСАГЧДЫН АЧ ХОЛБОГДОЛ")
print("=" * 80)

# Random Forest-ийн хувьсагчдын ач холбогдлыг график болгох
plt.figure(figsize=(12, 8))
top_n = min(15, len(feature_importance_rf))
top_features_rf = feature_importance_rf.head(top_n)

plt.barh(range(len(top_features_rf)), top_features_rf['importance'], 
         color='steelblue', edgecolor='black')
plt.yticks(range(len(top_features_rf)), top_features_rf['feature'])
plt.xlabel('Ач холбогдол', fontweight='bold', fontsize=12)
plt.ylabel('Хувьсагч', fontweight='bold', fontsize=12)
plt.title(f'Хамгийн чухал {top_n} хувьсагч (Random Forest)', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ График хадгалагдлаа: feature_importance.png")
plt.close()

# ======================================================================
# 12. ТЕСТИЙН ӨГӨГДӨЛ ДЭЭР ТААМАГЛАЛ ХИЙХ
# ======================================================================

print("\n" + "=" * 80)
print("🔮 АЛХАМ 11: ТЕСТИЙН ӨГӨГДӨЛ ДЭЭР ТААМАГЛАЛ")
print("=" * 80)

# Хамгийн сайн загварыг сонгох
best_model_name = results_df.index[0]
best_model = models[best_model_name]

print(f"✓ Хамгийн сайн загвар: {best_model_name}")
print(f"✓ Validation accuracy: {results_df.iloc[0]['accuracy']:.4f}")

# Тестийн өгөгдлийг боловсруулах
test_df_processed = test_df.copy()

# ID багана хадгалах (хэрэв байгаа бол)
test_ids = None
for col in test_df.columns:
    if 'id' in col.lower():
        test_ids = test_df_processed[col].copy()
        test_df_processed = test_df_processed.drop(columns=[col])
        break

# Зорилтот хувьсагч байгаа эсэхийг шалгах
if target_col in test_df_processed.columns:
    test_df_processed = test_df_processed.drop(columns=[target_col])
    print(f"⚠️ Тестийн өгөгдлөөс {target_col} устгагдлаа")

# Категори хувьсагчдыг кодлох
for col in categorical_cols:
    if col in test_df_processed.columns:
        if col in label_encoders:
            # Шинэ категори илэрвэл хамгийн түгээмэл утгыг өгөх
            le = label_encoders[col]
            def safe_transform(x):
                if x in le.classes_:
                    return le.transform([x])[0]
                else:
                    return le.transform([le.classes_[0]])[0]
            test_df_processed[col] = test_df_processed[col].apply(safe_transform)

# Алга болсон утгуудыг бөглөх
for col in numeric_cols:
    if col in test_df_processed.columns:
        if test_df_processed[col].isnull().sum() > 0:
            median_val = df[col].median()
            test_df_processed[col].fillna(median_val, inplace=True)

for col in categorical_cols:
    if col in test_df_processed.columns:
        if test_df_processed[col].isnull().sum() > 0:
            mode_val = 0  # Кодлогдсон утга
            test_df_processed[col].fillna(mode_val, inplace=True)

# Багануудыг тохируулах
for col in X.columns:
    if col not in test_df_processed.columns:
        test_df_processed[col] = 0

test_df_processed = test_df_processed[X.columns]

# Стандартчлах
X_test_scaled = scaler.transform(test_df_processed)

# Таамаглал хийх
test_predictions = best_model.predict(X_test_scaled)
test_predictions_proba = best_model.predict_proba(X_test_scaled)

print(f"\n✓ {len(test_predictions)} таамаглал хийгдлээ")

# Таамаглалын тархалт
pred_counts = pd.Series(test_predictions).value_counts().sort_index()
print("\n📊 Таамаглалын тархалт:")
for pred, count in pred_counts.items():
    label = le_target.classes_[pred] if 'le_target' in locals() else str(pred)
    pct = count / len(test_predictions) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")

# Үр дүнг хадгалах
submission_df = pd.DataFrame()
if test_ids is not None:
    submission_df['Loan_ID'] = test_ids
else:
    submission_df['Loan_ID'] = range(1, len(test_predictions) + 1)

submission_df[target_col] = test_predictions
if 'le_target' in locals():
    submission_df[f'{target_col}_Label'] = le_target.inverse_transform(test_predictions)

submission_df.to_csv('loan_predictions.csv', index=False)
print(f"\n✓ Таамаглал хадгалагдлаа: loan_predictions.csv")
print(f"\n📋 Эхний 10 таамаглал:")
print(submission_df.head(10))

# ======================================================================
# 13. ДҮГНЭЛТ
# ======================================================================

print("\n" + "=" * 80)
print("📝 АЛХАМ 12: ДҮГНЭЛТ БА ЗӨВЛӨМЖ")
print("=" * 80)

print(f"""
🎯 ТӨСЛИЙН ҮР ДҮН:

1. ӨГӨГДЛИЙН МЭДЭЭЛЭЛ:
   • Сургалтын өгөгдөл: {len(train_df)} мөр
   • Тестийн өгөгдөл: {len(test_df)} мөр
   • Шинжийн тоо: {len(X.columns)}
   • Зорилтот хувьсагч: {target_col}

2. ХАМГИЙН САЙН ЗАГВАР:
   • Загвар: {best_model_name}
   • Нарийвчлал: {results_df.iloc[0]['accuracy']:.2%}
   • F1-Score: {results_df.iloc[0]['f1_score']:.2%}
   • Cross-Validation: {results_df.iloc[0]['cv_mean']:.2%} (±{results_df.iloc[0]['cv_std']:.2%})
   
3. ХАМГИЙН ЧУХАЛ ХУВЬСАГЧИД (эхний 5):
{chr(10).join([f'   • {row["feature"]}: {row["importance"]:.4f}' 
              for _, row in feature_importance_rf.head(5).iterrows()])}

4. БҮХЭЭГДСЭН ДҮГНЭЛТ:
   • Зээлийн батламжийг машин сургалтын аргаар {results_df.iloc[0]['accuracy']:.1%} 
     нарийвчлалаар таамаглах боломжтой
   • {best_model_name} загвар хамгийн сайн үр дүн үзүүлсэн
   • Бүх загваруудын нарийвчлал {results_df['accuracy'].min():.1%}-{results_df['accuracy'].max():.1%} 
     хооронд байна
   
5. ПРАКТИКТ ХЭРЭГЛЭХ:
   • Банк, санхүүгийн байгууллагуудад зээл батлах/татгалзах шийдвэр 
     гаргахад туслах
   • Эрсдэлийн үнэлгээг автоматжуулах
   • Зээлийн процессыг хурдасгах
   • Хувь хүний хүчин зүйлийг багасгах
   
6. ЦААШДЫН САЙЖРУУЛАЛТ:
   • Илүү олон өгөгдөл цуглуулах
   • Feature engineering - шинэ хувьсагчид үүсгэх
   • Hyperparameter tuning - параметруудыг оновчилох
   • Ensemble методууд туршиж үзэх
   • Deep Learning аргууд ашиглах
   
7. ХЯЗГААРЛАЛТ:
   • Өгөгдлийн чанар, хэмжээнээс үр дүн хамаарна
   • 100% нарийвчлалтай байх боломжгүй
   • Тогтмол сургаж, шинэчлэх шаардлагатай
   • Бусад хүчин зүйлс (эдийн засгийн нөхцөл байдал, 
     геополитик эрсдэл) харгалзах хэрэгтэй
   
⚠️ МЭРГЭЖЛИЙН ЁС ЗҮЙ:
   • Үр дүнг үнэн зөв, шударгаар тайлбарлах
   • Алдаа, дутагдлыг нуун дарагдуулахгүй байх
   • Ашиг сонирхлын зөрчилгөөс ангид байх
   • Хувийн мэдээллийг хамгаалах
   • Загваруудын шийдвэрийг зөвхөн зөвлөмж болгон ашиглах,
     эцсийн шийдвэрийг хүн гаргах
""")

print("\n" + "=" * 80)
print("✅ ТӨСӨЛ АМЖИЛТТАЙ ДУУСЛАА!")
print("=" * 80)

print(f"""
 ХАДГАЛАГДСАН ФАЙЛУУД:
   1. loan_distributions.png - Өгөгдлийн тархалт
   2. correlation_matrix.png - Корреляцийн матриц
   3. feature_relationships.png - Хувьсагчдын хамаарал
   4. model_comparison.png - Загваруудын харьцуулалт
   5. roc_curves.png - ROC муруй (хэрэв binary classification)
   6. confusion_matrices.png - Confusion matrices
   7. feature_importance.png - Хувьсагчдын ач холбогдол
   8. loan_predictions.csv - Тестийн өгөгдлийн таамаглал
   9. loan_train.csv - Анхны сургалтын өгөгдөл (танай файл)
   10. loan_test.csv - Анхны тестийн өгөгдөл (танай файл)
    11. app.py - Төслийн код
""")
# ======================================================================
# 14. PDF ТАЙЛАН ҮҮСГЭХ
# ======================================================================

print("\n" + "=" * 80)
print("📄 АЛХАМ 13: PDF ТАЙЛАН ҮҮСГЭХ")
print("=" * 80)

from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def create_pdf_report():
    """PDF тайлан үүсгэх функц"""
    
    pdf_filename = f'loan_prediction_report_{datetime.now().strftime("%Y%m%d")}.pdf'
    
    with PdfPages(pdf_filename) as pdf:
        
        # ============ ХУУДАС 1: НҮҮР ХУУДАС ============
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7, 'ЗЭЭЛИЙН БАТЛАМЖИЙН ТААМАГЛАЛ', 
                ha='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.65, 'Машин сургалтын төсөл', 
                ha='center', fontsize=16)
        
        fig.text(0.5, 0.55, 'Багийн гишүүд:', 
                ha='center', fontsize=14, fontweight='bold')
        
        team_members = [
            '1. [Нэр 1] - Өгөгдөл боловсруулалт, цэвэрлэгээ',
            '2. [Нэр 2] - Машин сургалтын загвар ажиллуулалт',
            '3. [Нэр 3] - Визуализаци, график үүсгэлт',
            '4. [Нэр 4] - Дүн шинжилгээ, дүгнэлт',
            '5. [Нэр 5] - Тайлан бичилт, танилцуулга'
        ]
        
        y_pos = 0.48
        for member in team_members:
            fig.text(0.5, y_pos, member, ha='center', fontsize=11)
            y_pos -= 0.04
        
        fig.text(0.5, 0.2, f'Огноо: {datetime.now().strftime("%Y-%m-%d")}', 
                ha='center', fontsize=12)
        fig.text(0.5, 0.15, 'Эх сурвалж: Kaggle - Loan Approval Prediction Dataset', 
                ha='center', fontsize=10, style='italic')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ ХУУДАС 2: ХУРААНГУЙ ============
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, 'ХУРААНГУЙ', ha='center', fontsize=18, fontweight='bold')
        
        summary_text = f"""
Энэхүү төсөл нь зээлийн батламжийг машин сургалтын аргаар таамаглах зорилготой.
Kaggle-аас авсан {len(train_df)} мөрөөс бүрдсэн өгөгдлийг ашигласан.

ГҮЙЦЭТГЭСЭН АЖЛУУД:
• Өгөгдлийн цэвэрлэлт, боловсруулалт
• {len(X.columns)} хувьсагчийн шинжилгээ
• 4 төрлийн машин сургалтын загвар ажиллуулсан
• Загваруудын үр дүнг харьцуулсан

ҮР ДҮН:
Хамгийн сайн загвар: {best_model_name}
Нарийвчлал: {results_df.iloc[0]['accuracy']:.2%}
F1-Score: {results_df.iloc[0]['f1_score']:.2%}

ДҮГНЭЛТ:
Зээлийн батламжийг {results_df.iloc[0]['accuracy']:.1%} нарийвчлалаар 
таамаглах боломжтой болсон. Энэ нь банк, санхүүгийн байгууллагуудад
зээл олгох шийдвэр гаргахад тусална.
        """
        
        fig.text(0.1, 0.85, summary_text, fontsize=11, verticalalignment='top',
                wrap=True)
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ ХУУДАС 3: ӨГӨГДЛИЙН ТАРХАЛТ ============
        img = plt.imread('loan_distributions.png')
        fig = plt.figure(figsize=(8.5, 11))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Өгөгдлийн тархалт', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ ХУУДАС 4: КОРРЕЛЯЦИ ============
        img = plt.imread('correlation_matrix.png')
        fig = plt.figure(figsize=(8.5, 11))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Хувьсагчдын хоорондын корреляци', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ ХУУДАС 5: ЗАГВАРУУДЫН ХАРЬЦУУЛАЛТ ============
        img = plt.imread('model_comparison.png')
        fig = plt.figure(figsize=(8.5, 11))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Загваруудын харьцуулалт', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ ХУУДАС 6: CONFUSION MATRICES ============
        img = plt.imread('confusion_matrices.png')
        fig = plt.figure(figsize=(8.5, 11))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Confusion Matrices', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ ХУУДАС 7: ХУВЬСАГЧДЫН АЧ ХОЛБОГДОЛ ============
        img = plt.imread('feature_importance.png')
        fig = plt.figure(figsize=(8.5, 11))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Хувьсагчдын ач холбогдол', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ ХУУДАС 8: ҮР ДҮНГИЙН ХҮСНЭГТ ============
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, 'ЗАГВАРУУДЫН ҮР ДҮНГИЙН ХҮСНЭГТ', 
                ha='center', fontsize=16, fontweight='bold')
        
        # Хүснэгт үүсгэх
        ax = fig.add_subplot(111)
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        table_data.append(['Загвар', 'Accuracy', 'F1-Score', 'AUC', 'CV Score'])
        
        for idx, row in results_df.iterrows():
            table_data.append([
                idx,
                f"{row['accuracy']:.4f}",
                f"{row['f1_score']:.4f}",
                f"{row['auc']:.4f}",
                f"{row['cv_mean']:.4f} (±{row['cv_std']:.4f})"
            ])
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.3, 0.15, 0.15, 0.15, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Header-ийг тодруулах
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ ХУУДАС 9: ДҮГНЭЛТ ============
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, 'ДҮГНЭЛТ БА ЗӨВЛӨМЖ', 
                ha='center', fontsize=18, fontweight='bold')
        
        conclusion_text = f"""
1. ӨГӨГДЛИЙН МЭДЭЭЛЭЛ:
   • Сургалтын өгөгдөл: {len(train_df)} мөр
   • Шинжийн тоо: {len(X.columns)}
   • Зорилтот хувьсагч: {target_col}

2. ХАМГИЙН САЙН ЗАГВАР:
   • Загвар: {best_model_name}
   • Нарийвчлал: {results_df.iloc[0]['accuracy']:.2%}
   • F1-Score: {results_df.iloc[0]['f1_score']:.2%}

3. ХАМГИЙН ЧУХАЛ ХУВЬСАГЧИД:
"""
        for _, row in feature_importance_rf.head(5).iterrows():
            conclusion_text += f"   • {row['feature']}: {row['importance']:.4f}\n"
        
        conclusion_text += f"""

4. ДҮГНЭЛТ:
   Зээлийн батламжийг {results_df.iloc[0]['accuracy']:.1%} нарийвчлалаар 
   таамаглах боломжтой болсон. {best_model_name} загвар хамгийн 
   сайн үр дүн үзүүлсэн.

5. ПРАКТИКТ ХЭРЭГЛЭХ:
   • Банкны зээл батлах/татгалзах шийдвэрт туслах
   • Эрсдэлийн үнэлгээг автоматжуулах
   • Зээлийн процессыг хурдасгах

6. ЦААШДЫН САЙЖРУУЛАЛТ:
   • Илүү олон өгөгдөл цуглуулах
   • Feature engineering хийх
   • Hyperparameter tuning хийх
   • Ensemble методууд туршиж үзэх
"""
        
        fig.text(0.1, 0.85, conclusion_text, fontsize=10, verticalalignment='top',
                family='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ ХУУДАС 10: ЭШ СУРВАЛЖ ============
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, 'АШИГЛАСАН МАТЕРИАЛЫН ЖАГСААЛТ', 
                ha='center', fontsize=18, fontweight='bold')
        
        references = """
ЭШЛЭЛ (APA ФОРМАТ):

[1] Kaggle. (2024). Loan Approval Prediction Dataset. 
    Retrieved from https://www.kaggle.com/datasets/

[2] Scikit-learn Developers. (2024). Scikit-learn: Machine Learning in Python.
    Retrieved from https://scikit-learn.org/

[3] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
    Journal of Machine Learning Research, 12, 2825-2830.

[4] McKinney, W. (2010). Data Structures for Statistical Computing in Python.
    Proceedings of the 9th Python in Science Conference, 56-61.

[5] Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment.
    Computing in Science & Engineering, 9(3), 90-95.

[6] Waskom, M. (2021). seaborn: statistical data visualization.
    Journal of Open Source Software, 6(60), 3021.


АШИГЛАСАН ПРОГРАМ ХАНГАМЖ:

• Python 3.8+
• pandas 1.3.0+
• numpy 1.21.0+
• matplotlib 3.4.0+
• seaborn 0.11.0+
• scikit-learn 1.0.0+


ӨГӨГДЛИЙН ЭШ СУРВАЛЖ:

Төслийн ажилд ашигласан өгөгдөл нь Kaggle платформ дээрх
"Loan Approval Prediction Dataset" юм. Энэ өгөгдөл нь зээл авагчдын
хувийн мэдээлэл, орлого, өр төлбөр болон бусад санхүүгийн 
мэдээллүүдийг агуулдаг.
"""
        
        fig.text(0.1, 0.85, references, fontsize=9, verticalalignment='top',
                family='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # PDF metadata нэмэх
        d = pdf.infodict()
        d['Title'] = 'Зээлийн батламжийн таамаглал - Машин сургалтын төсөл'
        d['Author'] = 'Багийн гишүүд'
        d['Subject'] = 'Машин сургалт, Зээлийн таамаглал'
        d['Keywords'] = 'Machine Learning, Loan Prediction, Credit Risk'
        d['CreationDate'] = datetime.now()
    
    return pdf_filename

# PDF үүсгэх
try:
    pdf_file = create_pdf_report()
    print(f"\nPDF тайлан амжилттай үүсгэгдлээ: {pdf_file}")
except Exception as e:
    print(f"\nАЛДАА: PDF үүсгэхэд алдаа гарлаа: {e}")

print("\n" + "=" * 80)
print("БҮХ АЖИЛ ДУУСЛАА!")
print("=" * 80)
