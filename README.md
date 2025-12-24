[project_readme.md](https://github.com/user-attachments/files/24332726/project_readme.md)
# 👕 Clothing Resale Price Predictor

A machine learning project that predicts the resale value of thrifted clothing items based on attributes like brand, category, condition, material, color, and size.

## 🎯 Project Overview

This project uses machine learning to help thrift store resellers price their inventory more accurately. By analyzing patterns in clothing attributes, the model can predict fair resale prices with 72% accuracy.

## 📊 Dataset

- **Size**: 1,000 clothing items
- **Features**: Brand, Category, Condition, Material, Color, Size
- **Target**: Price (in USD)
- **Price Range**: $2 - $325
- **Brands**: 18 brands ranging from premium (Patagonia, Carhartt) to budget (H&M, Forever 21)

## 🤖 Models Trained

Three regression models were trained and compared:

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| **Gradient Boosting** | **72.49%** | **$15.58** | **$24.70** |
| Random Forest | 60.40% | $19.26 | $29.63 |
| Linear Regression | 33.90% | $26.67 | $38.28 |

**Winner**: Gradient Boosting Regressor

## 🔑 Key Findings

Feature importance analysis revealed:
1. **Category** (38%) - Most important factor (e.g., jackets vs t-shirts)
2. **Brand** (25%) - Premium brands command higher prices
3. **Condition** (15%) - "New with Tags" vs "Fair" significantly impacts price
4. **Color, Material, Size** (22% combined) - Smaller but meaningful impact

## 🚀 Web Application

An interactive Streamlit web app allows users to:
- Select clothing attributes via dropdown menus
- Get instant price predictions
- Receive contextual pricing insights

## 📸 Screenshots

![Model Performance](Screenshots/model_results.png)
![Streamlit App](images/streamlit_app.png)

## 📁 Project Structure

```
clothing-price-predictor/
├── app.py                          # Streamlit web application
├── ML_clothes.ipynb                # Jupyter notebook with full analysis
├── generate_data.py                # Dataset generation script
├── clothing_resale_data.csv        # Generated dataset
├── clothing_price_model.pkl        # Trained model
├── label_encoders.pkl              # Feature encoders
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.x**
- **scikit-learn** - Machine learning models and preprocessing
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib & seaborn** - Data visualization
- **Streamlit** - Web application framework

## 📈 Results

The final model achieves:
- **72% R² score** - Explains 72% of price variance
- **$15.58 average error** - Predictions within ~$16 of actual prices
- Strong performance across all price ranges

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end machine learning workflow
- Data generation with realistic patterns
- Feature engineering and encoding
- Model training, evaluation, and comparison
- Model deployment with interactive UI
- Data visualization and interpretation

## 🔮 Future Improvements

- Add image recognition to extract features from photos
- Incorporate seasonal pricing trends
- Add location-based pricing adjustments
- Expand dataset with real scraped data
- Deploy to cloud (Heroku, Streamlit Cloud)

## 👤 Author

Drew Rosinsky

## 📄 License

This project is open source and available under the MIT License.
