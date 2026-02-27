📄 README.md

# ☀️ Solar Energy Transition Predictor (2024-2040)

### 📊 Project Overview
This project uses **Machine Learning (Random Forest)** to predict the future percentage share of solar energy in the global energy mix. Unlike simple models, this approach incorporates **Economic Policy Drivers** and **Time-Lagged Variables** to account for the delay between government funding and actual infrastructure completion.

**Key Achievement:** Improved model accuracy from a baseline $R^2$ of 0.008 to an advanced $R^2$ of **0.8402**.

---

### 🧪 Academic Framework
The model follows a supervised regression approach:
$$Renewable\_Share_t = \beta_0 + \beta_1(Solar\_Cost_t) + \beta_2(Subsidy_t) + \beta_3(Subsidy_{t-1}) + \epsilon$$

* **Solar Panel Cost:** Modeled with a negative correlation (as costs drop, adoption rises).
* **Lagged Subsidies ($t-1$):** Captures "Policy Inertia"—the reality that last year's budget drives this year's construction.
* **Time Projection:** Accounts for the S-Curve of technology adoption up to the year 2040.

---

### 📂 Project Structure
* `data/`: Contains raw energy CSVs and the final merged dataset.
* `scripts/`: 
    * `01_merge.py`: Combines 17 sources into a master file.
    * `04_advanced_model.py`: The Random Forest engine (84% accuracy).
    * `05_economic_integration.py`: Injects Solar Costs and Lagged Subsidies.
* `app.py`: The interactive Streamlit dashboard.
* `results/`: Visualizations of feature importance and error metrics.

---

### 🚀 Running Instructions

#### 1. Setup Environment
Ensure you are in your virtual environment:
```powershell
.\.venv\Scripts\activate

```

#### 2. Install Dependencies

```powershell
pip install streamlit pandas numpy scikit-learn

```

#### 3. Run the Data Pipeline (Optional)

If you want to re-generate the model results:

```powershell
python scripts/05_economic_integration.py
python scripts/06_final_academic_model.py

```

#### 4. Launch the Dashboard

To open the interactive front-end in your browser:

```powershell
streamlit run app.py

```

```

---

### 🛠️ Quick Running Instructions (For your presentation)

1.  **Open Terminal** in VS Code.
2.  **Activate your environment**: Type `.\.venv\Scripts\activate`.
3.  **Start the app**: Type `streamlit run app.py`.
4.  **Interact**: Use the **Year Slider** to show how solar share grows toward 2040, and the **Solar Cost Slider** to show how price drops accelerate the transition.



### 🏁 Final Project Check
You now have:
* ✅ **Clean Data** (Merged from 17 files).
* ✅ **Advanced Model** (Random Forest with 84% $R^2$).
* ✅ **Policy Logic** (Lagged Subsidies included).
* ✅ **Front End** (Interactive Streamlit App).
* ✅ **Documentation** (Professional README).

**You have officially completed the project! Is there anything else you need help with before you present this amazing work?**

```