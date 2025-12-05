# =============================
#   FULL CREDIT RISK PROJECT
#     + PDF REPORT GENERATOR
#     (FULL WORKING VERSION)
# =============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime


# ======================================================
# 1. LOAD CREDIT RISK DATASET
# ======================================================

def load_data():
    # Kaggle dataset (replace with your path if needed)
    df = pd.read_csv(r"C:\Users\tulga\Desktop\Statistic\credit_risk_data.csv")

    # Encode target
    le = LabelEncoder()
    df["loan_status"] = le.fit_transform(df["loan_status"])

    df = df.dropna()
    return df


# ======================================================
# 2. TRAIN MODELS
# ======================================================

def train_models(df):
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB()
    }

    results = []
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # ROC curve
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        roc_data[name] = (fpr, tpr, roc_auc)

        results.append([name, acc, f1])

    results_df = pd.DataFrame(results, columns=["model", "accuracy", "f1_score"]).set_index("model")

    # Feature importance (Random Forest)
    rf = models["Random Forest"]
    importances = rf.feature_importances_
    feature_importance_rf = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    return results_df, feature_importance_rf, roc_data


# ======================================================
# 3. PLOT GRAPHS (SAVED AS PNG)
# ======================================================

def create_plots(results_df, roc_data):
    # Model comparison bar plot
    plt.figure(figsize=(8, 4))
    plt.bar(results_df.index, results_df["accuracy"], color="steelblue")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.savefig("model_comparison.png", bbox_inches="tight")
    plt.close()

    # ROC Curves
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC Curves")
    plt.savefig("roc_curves.png", bbox_inches="tight")
    plt.close()


# ======================================================
# 4. PDF REPORT GENERATOR
# ======================================================

def create_pdf_report(results_df, feature_importance_rf):
    pdf_filename = f"Credit_Risk_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)

    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=1
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2e5090'),
        spaceAfter=12,
        spaceBefore=12
    )

    # TITLE
    story.append(Paragraph("ЗЭЭЛИЙН ЭРСДЛИЙН АНГИЛАЛ", title_style))
    story.append(Paragraph("МАШИН СУРГАЛТЫН ТӨСӨЛ", title_style))
    story.append(Spacer(1, 20))

    story.append(Paragraph(
        f"<b>Огноо:</b> {datetime.now().strftime('%Y-%m-%d')}",
        styles["Normal"]
    ))

    # SUMMARY
    story.append(Paragraph("1. ХУРААНГУЙ", heading_style))

    summary = f"""
    Энэхүү судалгаанд Kaggle өгөгдөл ашиглан 4 төрлийн ML загвар ажиллуулсан.
    Хамгийн сайн үр дүнтэй загвар нь <b>{results_df.index[0]}</b> бөгөөд 
    нарийвчлал нь <b>{results_df.iloc[0]['accuracy']:.2%}</b> болсон.
    """

    story.append(Paragraph(summary, styles["Normal"]))
    story.append(Spacer(1, 20))

    # RESULTS TABLE
    story.append(Paragraph("5. ҮР ДҮН", heading_style))
    table_data = [["Загвар", "Accuracy", "F1"]]

    for model, row in results_df.iterrows():
        table_data.append([model, f"{row['accuracy']:.4f}", f"{row['f1_score']:.4f}"])

    t = Table(table_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('GRID',(0,0),(-1,-1),1,colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    # Feature importance
    story.append(Paragraph("<b>Random Forest — Top Features</b>", styles["Normal"]))

    impt_data = [["Feature", "Importance"]]
    for _, row in feature_importance_rf.head(5).iterrows():
        impt_data.append([row["feature"], f"{row['importance']:.4f}"])

    t2 = Table(impt_data, colWidths=[3*inch, 2*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('GRID',(0,0),(-1,-1),1,colors.black)
    ]))

    story.append(t2)
    story.append(PageBreak())

    # Images
    if os.path.exists("model_comparison.png"):
        story.append(Paragraph("Model Accuracy Comparison", styles["Heading3"]))
        story.append(Image("model_comparison.png", width=6*inch, height=3*inch))

    story.append(PageBreak())

    if os.path.exists("roc_curves.png"):
        story.append(Paragraph("ROC Curves", styles["Heading3"]))
        story.append(Image("roc_curves.png", width=6*inch, height=4*inch))

    doc.build(story)
    print(f"✅ PDF report created: {pdf_filename}")


# ======================================================
# 5. RUN EVERYTHING
# ======================================================

if __name__ == "__main__":
    df = load_data()
    results_df, feature_importance_rf, roc_data = train_models(df)
    create_plots(results_df, roc_data)
    create_pdf_report(results_df, feature_importance_rf)
