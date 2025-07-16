import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, log_loss
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_ind


# Optional: XGBoost if installed
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# Paths to input files
ct_path = "./pet_ct_radiomics_features_CT.csv"
pet_path = "./pet_ct_radiomics_features_PET.csv"
label_path = "./PET_CT_Metadata_with_Metastasis_Labels.csv"

# Load data
ct = pd.read_csv(ct_path)
pet = pd.read_csv(pet_path)
labels = pd.read_csv(label_path)
labels.columns = labels.columns.str.strip()  # clean column names

# Function to prepare data per modality
def prepare_data(radiomics_df, modality_name):
    merged = radiomics_df.merge(labels[['PatientID', 'Metastasis_Label']], on='PatientID', how='inner')
    merged.rename(columns={'Metastasis_Label': 'Metastasis'}, inplace=True)
    X = merged.drop(columns=['PatientID', 'Metastasis']).select_dtypes(include='number')
    y = merged['Metastasis']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Print top contributing features
    feature_names = X.columns
    n_components = X_pca.shape[1]
    components_df = pd.DataFrame(pca.components_, columns=feature_names, index=[f"PC{i+1}" for i in range(n_components)])
    print(f"\nTop features contributing to each PCA component ({modality_name}):")
    for pc in components_df.index:
        top_features = components_df.loc[pc].abs().sort_values(ascending=False).head(5)
        print(f"{pc}: {', '.join(top_features.index)}")
    print(f"\n{modality_name} PCA: Selected {X_pca.shape[1]} components explaining 95% variance.")

    return X_pca, y, components_df

# P-values
def compute_p_values(radiomics_df, label_df, modality):
    merged = radiomics_df.merge(label_df[['PatientID', 'Metastasis_Label']], on='PatientID', how='inner')
    merged.rename(columns={'Metastasis_Label': 'Metastasis'}, inplace=True)

    numeric_features = merged.select_dtypes(include='number').drop(columns=['Metastasis'])
    p_values = {}

    for feature in numeric_features.columns:
        group0 = merged[merged['Metastasis'] == 0][feature]
        group1 = merged[merged['Metastasis'] == 1][feature]
        try:
            _, p = ttest_ind(group0, group1, equal_var=False)
            p_values[feature] = p
        except:
            p_values[feature] = np.nan  

    pvals_df = pd.DataFrame.from_dict(p_values, orient='index', columns=['p-value'])
    pvals_df = pvals_df.sort_values('p-value')
    pvals_df.index.name = 'Feature'
    print(f"\nTop significant features in {modality} (lowest p-values):\n")
    print(pvals_df.head(10))
    
    pvals_df.to_csv(f"p_values_{modality}.csv")
    return pvals_df


# Evaluate models function with all classifiers
def evaluate_models(X, y, modality):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Naive Bayes": GaussianNB()
    }

    if xgb_available:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        loss = log_loss(y_test, y_prob)

        results.append({
            "Modality": modality,
            "Model": name,
            "Accuracy": acc,
            "F1 Score": f1,
            "Log Loss": loss
        })

    return pd.DataFrame(results)

# Helper to extract top 5 features per PCA component into DataFrame
def get_top_features_df(components_df):
    top_features = {}
    for pc in components_df.index:
        top_features[pc] = components_df.loc[pc].abs().sort_values(ascending=False).head(5).index.tolist()
    return pd.DataFrame(top_features)

# Run pipeline
X_ct, y_ct, components_ct = prepare_data(ct, "CT")
X_pet, y_pet, components_pet = prepare_data(pet, "PET")

pvals_ct = compute_p_values(ct, labels, "CT")
pvals_pet = compute_p_values(pet, labels, "PET")


results_ct = evaluate_models(X_ct, y_ct, "CT")
results_pet = evaluate_models(X_pet, y_pet, "PET")

final_results = pd.concat([results_ct, results_pet], ignore_index=True)

print("\nModel Evaluation Results:\n")
print(final_results)

# Export results to CSV
final_results.to_csv("model_evaluation_results.csv", index=False)

# Export PCA top features per component to Excel
ct_top_df = get_top_features_df(components_ct)
pet_top_df = get_top_features_df(components_pet)

with pd.ExcelWriter("pca_top_features_CT_PET.xlsx") as writer:
    ct_top_df.to_excel(writer, sheet_name="CT_Top_Features", index=False)
    pet_top_df.to_excel(writer, sheet_name="PET_Top_Features", index=False)
