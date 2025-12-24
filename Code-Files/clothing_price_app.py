import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

# Page configuration
st.set_page_config(
    page_title="Clothing Resale Price Predictor",
    page_icon="👕",
    layout="centered"
)

# Title and description
st.title("👕 Clothing Resale Price Predictor")
st.markdown("*Predict the resale value of thrifted clothing items*")
st.markdown("---")

# Load or train model (you'll need to have this ready)
@st.cache_resource
def load_model_and_encoders():
    """
    Load the trained model and label encoders
    If files don't exist, returns None and we'll show a message
    """
    try:
        with open('clothing_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        return None, None

model, label_encoders = load_model_and_encoders()

# Define options for dropdowns
brands = ['Patagonia', 'The North Face', 'Carhartt', "Levi's", 'Nike', 'Adidas',
          'Gap', 'Old Navy', 'American Eagle', 'H&M', 'Zara', 'Uniqlo',
          'Walmart', 'Target', 'Shein', 'Forever 21', 'Primark', 'Fashion Nova']

categories = ['Jacket', 'Coat', 'Dress', 'Boots', 'Jeans', 'Sweater', 
              'Hoodie', 'Sneakers', 'T-shirt', 'Tank Top', 'Shorts', 'Socks']

conditions = ['New with Tags', 'Like New', 'Excellent', 'Good', 'Fair']

materials = ['Cotton', 'Polyester', 'Wool', 'Denim', 'Leather', 'Synthetic', 'Nylon']

colors = ['Black', 'White', 'Blue', 'Red', 'Green', 'Gray', 'Brown', 'Navy', 'Beige']

sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Item Details")
    brand = st.selectbox("Brand", brands, index=0)
    category = st.selectbox("Category", categories, index=0)
    condition = st.selectbox("Condition", conditions, index=0)

with col2:
    st.subheader("Specifications")
    material = st.selectbox("Material", materials, index=0)
    color = st.selectbox("Color", colors, index=0)
    size = st.selectbox("Size", sizes, index=2)  # Default to 'M'

st.markdown("---")

# Predict button
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    if model is None:
        st.error("❌ Model not found! Please train the model first and save it as 'clothing_price_model.pkl'")
    else:
        # Create dataframe with user input
        input_data = pd.DataFrame({
            'Brand': [brand],
            'Category': [category],
            'Condition': [condition],
            'Material': [material],
            'Color': [color],
            'Size': [size]
        })
        
        # Encode the input
        try:
            for col in ['Brand', 'Category', 'Condition', 'Material', 'Color', 'Size']:
                input_data[col] = label_encoders[col].transform(input_data[col])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result with styling
            st.markdown("### 💰 Predicted Resale Price")
            st.markdown(f"## ${prediction:.2f}")
            
            # Add some context
            if prediction < 20:
                st.info("💡 This is a lower-priced item, great for budget thrifters!")
            elif prediction < 50:
                st.info("💡 Mid-range item with good resale potential.")
            elif prediction < 100:
                st.success("💡 Higher-value item! Should sell well.")
            else:
                st.success("💡 Premium item with excellent resale value!")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")

# Add information section at the bottom
st.markdown("---")
with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    This tool predicts the resale price of thrifted clothing items based on:
    - **Brand**: Premium brands (Patagonia, Carhartt) typically command higher prices
    - **Category**: Outerwear and footwear generally have higher resale values
    - **Condition**: Better condition means higher prices
    - **Material**: Quality materials (wool, leather) increase value
    - **Color & Size**: These have minor impacts on pricing
    
    The model was trained using Gradient Boosting Regression and achieves **72% accuracy** 
    in explaining price variations.
    """)

st.markdown("---")
st.markdown("*Built with Python, scikit-learn, and Streamlit*")
