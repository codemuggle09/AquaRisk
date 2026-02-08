<h1 align="center" style="font-size:48px">🎯 Groundwater Fluoride Prediction Using Machine Learning &amp; Fuzzy Logic</h1>

<p align="center">
A data-driven, intelligent, and scalable framework to analyze groundwater fluoride contamination across India using Machine Learning, Regression Models, and a Fuzzy Inference System (FIS).  
This system supports early detection of fluoride-vulnerable regions and helps government agencies &amp; water-resource managers make informed decisions.
</p>

<hr>

<h2>🌍 Project At a Glance</h2>

✔ Analyzes <b>16,776+</b> groundwater samples from Indian states & districts  
✔ Predicts fluoride levels using <b>Regression Models</b>  
✔ Classifies water into <b>Safe / Moderate / High-risk</b> categories using ML  
✔ Uses <b>Mamdani Fuzzy Logic</b> for human-interpretable risk scoring  
✔ Generates <b>state-level analysis & heatmaps</b>  
✔ Built for <b>accuracy, interpretability, and large-scale deployment</b>  

<hr>

<h2>🧠 Why This Project?</h2>

Fluoride contamination is a major threat in Indian groundwater. Traditional chemical testing is slow and costly.  
This project solves that by merging:

🔹 <b>Hydrogeochemical science</b>  
🔹 <b>Machine Learning</b>  
🔹 <b>Fuzzy Logic interpretation</b>  

➡ Result: A fast, flexible, and reliable groundwater risk assessment system.

<hr>

<h2>📂 Dataset Overview</h2>

Each record contains:

<table>
<tr><th>Feature Type</th><th>Parameters</th></tr>
<tr><td>Physicochemical</td><td>pH, EC, TDS, Na⁺, Ca²⁺, Mg²⁺, K⁻, Cl⁻, SO₄²⁻, NO₃⁻, HCO₃⁻</td></tr>
<tr><td>Target</td><td>Fluoride concentration (mg/L)</td></tr>
<tr><td>Location</td><td>State + District identifiers</td></tr>
</table>

These features significantly impact fluoride mobility inside aquifers.

<hr>

<h2>⚙️ Data Preprocessing Pipeline</h2>

<h3>🔧 1. Standardization</h3>
• Cleans and normalizes column names (e.g., “EC µS/cm” → “EC”).

<h3>🧹 2. Invalid & Missing Values</h3>
• Converts “NA”, “–”, blanks to NaN  
• Uses <b>Median Imputation</b> for numeric stability  

<h3>🧪 3. Fluoride Risk Label Creation</h3>
Based on WHO drinking water standards:

<table>
<tr><th>Class</th><th>Fluoride Level</th><th>Interpretation</th></tr>
<tr><td>0</td><td>&lt; 1.5 mg/L</td><td>Safe</td></tr>
<tr><td>1</td><td>1.5–2.5 mg/L</td><td>Moderate Risk</td></tr>
<tr><td>2</td><td>&gt; 2.5 mg/L</td><td>High Risk</td></tr>
</table>

<h3>📏 4. Scaling</h3>
• Min–Max scaling to range <b>0–1</b>

<h3>🧩 5. Categorical Encoding</h3>
• One-Hot Encoding for state, district, well-type  

<h3>⚖️ 6. Class Balancing (SMOTE)</h3>
• Balances all 3 risk classes → dataset becomes <b>perfectly balanced</b>.

<hr>

<h2>🤖 Machine Learning Models Implemented</h2>

Seven models were trained:

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Linear | Baseline clarity |
| SVM (RBF) | Kernel | Captures nonlinearity |
| ANN | Neural Network | Learns complex patterns |
| AdaBoost | Ensemble | Focuses on hard samples |
| XGBoost | Gradient Boosting | Fast + accurate |
| LightGBM | Boosting | Efficient, large-scale |
| Random Forest | Ensemble | ⭐ <b>Best classifier</b> |

<h3>🏆 Top Performer: Random Forest Classifier</h3>

🎯 Accuracy: <b>93%</b>  
🎯 Strong precision, recall, and F1 across all classes  

<hr>

<h2>📈 Regression Models for Continuous Prediction</h2>

Three regression models were tested:

<table>
<tr><th>Model</th><th>R² Score</th><th>RMSE</th></tr>
<tr><td>Linear Regression</td><td>0.218</td><td>0.709</td></tr>
<tr><td><b>Random Forest Regressor</b></td><td><b>0.273</b></td><td><b>0.684</b></td></tr>
<tr><td>SVR</td><td>0.174</td><td>0.729</td></tr>
</table>

🏅 <b>Best Model: Random Forest Regressor</b>  

<hr>

<h2>🌡️ Fuzzy Logic Risk Classification</h2>

A Mamdani-type Fuzzy Inference System generates interpretable risk labels.

<h3>🏷 Input Memberships (Fluoride):</h3>

- Very Low  
- Low  
- Normal  
- High  
- Very High  

<h3>🟦 Output Memberships (Risk Score):</h3>

- Low Risk  
- Medium Risk  
- High Risk  

<h3>📜 Example Rules:</h3>

- If Fluoride is <b>Very High</b> → Risk is <b>High</b>  
- If Fluoride is <b>Normal</b> → Risk is <b>Low</b>  
- If Fluoride is <b>Low</b> → Risk is <b>Medium</b>  

<h3>🧮 Final Categories:</h3>

<table>
<tr><th>Risk Score</th><th>Category</th></tr>
<tr><td>&lt; 33</td><td>Low</td></tr>
<tr><td>33–66</td><td>Medium</td></tr>
<tr><td>&gt;= 66</td><td>High</td></tr>
</table>

<hr>

<h2>📊 Key Results</h2>

<h3>✔ ML Performance</h3>
• 93% accuracy  
• Low misclassification  
• Stable precision & recall  

<h3>✔ Fuzzy Interpretation</h3>
• State-wise risk maps  
• Fuzzy score distributions  
• Easy human understanding  

<h3>✔ Combined ML + FIS System</h3>
<b>Accurate + Interpretable + Scalable</b>

<hr>

<h2>⚠️ Limitations</h2>

🔸 Dataset originally imbalanced  
🔸 No seasonal data  
🔸 Missing contaminants (heavy metals, nitrate interactions)  
🔸 No spatial hydrogeology included  

<hr>

<h2>🔮 Future Directions</h2>

✨ GIS heatmaps  
✨ Deep learning for prediction  
✨ Multi-contaminant modeling  
✨ SHAP/LIME for explainability  
✨ Real-time dashboards  

<hr>
<h2>📥 Installation & Usage</h2>

```bash
# Clone the repository
git clone https://github.com/USERNAME/REPOSITORY

# Navigate into project folder
cd REPOSITORY

# Install dependencies
pip install -r requirements.txt

# Run the main pipeline
python main.py
```

<hr>
<h2>👥 Contributors</h2>

👩‍💻 <b>Aishwarya Para (2023BMS-022)</b><br>
👩‍💻 <b>Nihita Kolukula (2023BMS-015)</b>


