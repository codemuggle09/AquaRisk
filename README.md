<h1 align="center" style="font-size:48px">Groundwater Fluoride Prediction Using Machine Learning & Fuzzy Logic</h1>

<p align="center">
A data-driven, intelligent, and scalable framework to analyze groundwater fluoride contamination across India using Machine Learning, Regression Models, and a Fuzzy Inference System (FIS). This system supports early detection of fluoride-vulnerable regions and helps government agencies and water-resource managers make informed decisions.
</p>

<hr>

<h2>Project Objectives</h2>

<ul>
  <li><b>Large-Scale Analysis</b>: Evaluates over 16,776 groundwater samples across various Indian states and districts.</li>
  <li><b>Predictive Modeling</b>: Utilizes advanced regression techniques to estimate precise fluoride concentrations.</li>
  <li><b>Automated Classification</b>: Categorizes water quality into Safe, Moderate, and High-risk zones using optimized classifiers.</li>
  <li><b>Interpretability</b>: Employs Mamdani Fuzzy Logic to convert technical data into human-readable risk scores.</li>
  <li><b>Spatial Visualization</b>: Generates state-level heatmaps and regional analysis for spatial awareness.</li>
</ul>

<hr>

<h2>Dataset Architecture</h2>

<p>The dataset comprises physicochemical parameters that significantly influence fluoride mobility within aquifers.</p>

<table width="100%">
  <tr>
    <th align="left">Feature Category</th>
    <th align="left">Parameters Included</th>
  </tr>
  <tr>
    <td><b>Physicochemical</b></td>
    <td>pH, EC, TDS, Na⁺, Ca²⁺, Mg²⁺, K⁺, Cl⁻, SO₄²⁻, NO₃⁻, HCO₃⁻</td>
  </tr>
  <tr>
    <td><b>Target Variable</b></td>
    <td>Fluoride concentration (mg/L)</td>
  </tr>
  <tr>
    <td><b>Geospatial</b></td>
    <td>State and District identifiers</td>
  </tr>
</table>

<hr>

<h2>Data Preprocessing Pipeline</h2>

<ol>
  <li><b>Standardization</b>: Normalization of column nomenclature (e.g., standardizing “EC µS/cm” to “EC”).</li>
  <li><b>Imputation</b>: Conversion of invalid entries to null values followed by Median Imputation to maintain numeric stability.</li>
  <li><b>Risk Labeling</b>: Implementation of WHO drinking water standards for classification.
    <ul>
      <li>Class 0 (&lt; 1.5 mg/L): Safe</li>
      <li>Class 1 (1.5–2.5 mg/L): Moderate Risk</li>
      <li>Class 2 (&gt; 2.5 mg/L): High Risk</li>
    </ul>
  </li>
  <li><b>Scaling</b>: Application of Min-Max scaling to a standard 0–1 range.</li>
  <li><b>Class Balancing</b>: Utilization of SMOTE (Synthetic Minority Over-sampling Technique) to resolve dataset imbalances and achieve perfect class parity.</li>
</ol>

<hr>

<h2>Machine Learning Performance</h2>

<p>Seven distinct models were evaluated to determine the most effective classifier for fluoride risk.</p>

<table width="100%">
  <tr>
    <th align="left">Model</th>
    <th align="left">Classification Type</th>
    <th align="left">Accuracy</th>
  </tr>
  <tr>
    <td><b>Random Forest</b></td>
    <td><b>Ensemble Learning</b></td>
    <td><b>93% (Top Performer)</b></td>
  </tr>
  <tr>
    <td>XGBoost</td>
    <td>Gradient Boosting</td>
    <td>High Accuracy</td>
  </tr>
  <tr>
    <td>LightGBM</td>
    <td>Boosting</td>
    <td>Efficiency at Scale</td>
  </tr>
  <tr>
    <td>ANN</td>
    <td>Neural Network</td>
    <td>Pattern Recognition</td>
  </tr>
  <tr>
    <td>SVM (RBF)</td>
    <td>Kernel-based</td>
    <td>Nonlinear Mapping</td>
  </tr>
</table>

<h3>Regression Analysis (Continuous Prediction)</h3>

<table width="100%">
  <tr>
    <th align="left">Model</th>
    <th align="left">R² Score</th>
    <th align="left">RMSE</th>
  </tr>
  <tr>
    <td><b>Random Forest Regressor</b></td>
    <td><b>0.273</b></td>
    <td><b>0.684</b></td>
  </tr>
  <tr>
    <td>Linear Regression</td>
    <td>0.218</td>
    <td>0.709</td>
  </tr>
  <tr>
    <td>SVR</td>
    <td>0.174</td>
    <td>0.729</td>
  </tr>
</table>

<hr>

<h2>Fuzzy Logic Inference System (FIS)</h2>

<p>The system utilizes a Mamdani-type FIS to handle environmental uncertainty and provide interpretable results.</p>

<ul>
  <li><b>Input Memberships</b>: Very Low, Low, Normal, High, Very High.</li>
  <li><b>Output Risk Scores</b>: Low Risk (&lt; 33), Medium Risk (33–66), High Risk (&gt;= 66).</li>
</ul>

<hr>

<h2>Limitations & Future Scope</h2>

<p><b>Current Constraints:</b></p>
<ul>
  <li>Absence of seasonal temporal data.</li>
  <li>Limited to fluoride without accounting for heavy metal or nitrate interactions.</li>
  <li>Exclusion of complex spatial hydrogeological layers.</li>
</ul>

<p><b>Future Directions:</b></p>
<ul>
  <li>Implementation of GIS-based real-time heatmaps.</li>
  <li>Integration of Deep Learning for enhanced predictive precision.</li>
  <li>Incorporation of SHAP/LIME for model explainability and transparency.</li>
</ul>

<hr>

<h2>Installation & Usage</h2>

```bash
# Clone the repository
git clone [https://github.com/codemuggle09/AquaRisk](https://github.com/codemuggle09/AquaRisk)

# Navigate into project folder
cd AquaRisk

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
python -m streamlit run webapp.py