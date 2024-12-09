import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def load_and_preprocess_data(file_path):
    df_inventory = pd.read_csv(file_path, encoding_errors="ignore")
    df_freq = df_inventory.copy()
    df_freq['order date (DateOrders)'] = pd.to_datetime(df_freq['order date (DateOrders)'])
    df_freq['shipping date (DateOrders)'] = pd.to_datetime(df_freq['shipping date (DateOrders)'])

    # Calculate the number of orders per customer
    customer_order_count = df_freq.groupby('Customer Id')['Order Id'].count().reset_index(name='OrderCount')

    # Calculate the time between each order for each customer
    df_freq = df_freq.sort_values(by=['Customer Id', 'order date (DateOrders)'])
    df_freq['PreviousOrderDate'] = df_freq.groupby('Customer Id')['order date (DateOrders)'].shift(1)
    df_freq['TimeBetweenOrders'] = (df_freq['order date (DateOrders)'] - df_freq['PreviousOrderDate']).dt.days

    # Calculate the average time between orders
    avg_time_between_orders = df_freq.groupby('Customer Id')['TimeBetweenOrders'].mean().reset_index(name='AvgTimeBetweenOrders')

    # Calculate the average sales and profit per order
    avg_sales_per_order = df_freq.groupby('Customer Id')['Sales'].mean().reset_index(name='AvgSalesPerOrder')
    avg_profit_per_order = df_freq.groupby('Customer Id')['Order Profit Per Order'].mean().reset_index(name='AvgProfitPerOrder')

    # Merge all the calculated features
    customer_data = customer_order_count.merge(avg_time_between_orders, on='Customer Id', how='left')
    customer_data = customer_data.merge(avg_sales_per_order, on='Customer Id', how='left')
    customer_data = customer_data.merge(avg_profit_per_order, on='Customer Id', how='left')

    # Define segmentation criteria
    def segment_customer(row):
        if row['OrderCount'] == 1 or row['OrderCount'] == 0:
            return 'NewComer'
        elif row['OrderCount'] > 1 and row['AvgTimeBetweenOrders'] <= 30:
            return 'FrequentCustomer'
        elif row['OrderCount'] > 1 and row['AvgTimeBetweenOrders'] > 30 and row['AvgSalesPerOrder'] > 100:
            return 'LoyalCustomer'
        else:
            return 'ImpulseCustomer'

    # Apply segmentation logic
    customer_data['CustomerSegment'] = customer_data.apply(segment_customer, axis=1)

    # Merge the segmentation back to the original DataFrame
    df_freq = df_freq.merge(customer_data[['Customer Id', 'CustomerSegment']], on='Customer Id', how='left')

    sales_df = df_freq.groupby(
        ['order date (DateOrders)','Product Name','Market', 'CustomerSegment']
    ).agg({
        "Category Id": "first",
        "Department Id": "first",
        "Order Item Quantity": 'sum',
        "Sales": "mean",
        "Order Item Discount": "mean"
    }).reset_index()

    sales_df['order date (DateOrders)'] = pd.to_datetime(sales_df['order date (DateOrders)'])
    sales_df = sales_df.sort_values('order date (DateOrders)')

    # Create time-based features
    sales_df["Year"] = sales_df['order date (DateOrders)'].dt.year
    sales_df['Day'] = sales_df['order date (DateOrders)'].dt.day
    sales_df['Weekday'] = sales_df['order date (DateOrders)'].dt.weekday

    # Create a range for holidays based on the data
    cal = calendar()
    date_range = pd.date_range(start=sales_df['order date (DateOrders)'].min(),
                               end=sales_df['order date (DateOrders)'].max())

    # Get US federal holidays in the range
    holidays = cal.holidays(start=date_range.min(), end=date_range.max())
    sales_df['IsHoliday'] = sales_df['order date (DateOrders)'].isin(holidays)

    return sales_df, holidays, df_freq

def predict_discount_rate(order_date, product_name, market, customer_segment, category_id, department_id, order_item_quantity, sales, holidays, df_freq):
    # Load the encoders, scaler, and model for deployment
    encoders_and_scaler = joblib.load('encoders_and_scaler.pkl')
    label_encoders = encoders_and_scaler['label_encoders']
    scaler = encoders_and_scaler['scaler']
    loaded_model = load_model('discount_threshold_model.h5')

    # Test sample as a dictionary
    test_sample = {
        'order_date_(DateOrders)': [order_date],
        'Product_Name': [product_name],
        'Market': [market],
        'CustomerSegment': [customer_segment],
        'Category_Id': [category_id],
        'Department_Id': [department_id],
        'Order_Item_Quantity': [order_item_quantity],
        'Sales': [sales],
    }

    # Convert to DataFrame
    test_df = pd.DataFrame(test_sample)

    # Step 1: Rename columns to match training format
    test_df.rename(columns=lambda x: x.replace(' ', '_').replace('(DateOrders)', ''), inplace=True)

    # Step 2: Preprocess order date (if needed)
    test_df['order_date'] = pd.to_datetime(test_df['order_date_'])

    test_df["Year"] = test_df['order_date'].dt.year
    test_df['Day'] = test_df['order_date'].dt.day
    test_df['Weekday'] = test_df['order_date'].dt.weekday
    test_df['IsHoliday'] = test_df['order_date'].isin(holidays).astype(int)

    # Step 3: Label encode categorical features
    categorical_columns = ['Product_Name', 'Market', 'CustomerSegment']

    for col in categorical_columns:
        test_df[col] = test_df[col].map(
            lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1
        )

    # Step 4: Scale numeric features
    numeric_features = [
        'Order_Item_Quantity', 'Sales', 'Category_Id', 'Department_Id',
        'Year', 'Day', 'Weekday', 'IsHoliday'
    ]
    test_df[numeric_features] = scaler.transform(test_df[numeric_features])

    # Step 5: Prepare inputs for the model
    # Combine categorical and numeric features into a single array
    combined_inputs = np.hstack([test_df[col].values.reshape(-1, 1) for col in categorical_columns + numeric_features])

    # Step 6: Perform inference
    predicted_discount_rate = loaded_model.predict(combined_inputs)
    return predicted_discount_rate[0][0]

def create_dashboard_page():
    import streamlit as st

    # Load and preprocess data
    file_path = "data/Inventory.csv"
    sales_df, holidays, df_freq = load_and_preprocess_data(file_path)
    encoders_and_scaler = joblib.load('encoders_and_scaler.pkl')
    label_encoders = encoders_and_scaler['label_encoders']
    scaler = encoders_and_scaler['scaler']
    # Streamlit app
    st.title("Discount Rate Prediction")

    # Input fields
    order_date = st.date_input("Order Date")
    customer_segment = st.selectbox("Customer Segment", label_encoders['CustomerSegment'].classes_)
    market = st.selectbox("Market", label_encoders['Market'].classes_)
    product_name = st.selectbox("Product Name", label_encoders['Product_Name'].classes_)

    # Filter available categories and departments based on the selected product
    available_categories = df_freq[df_freq['Product Name'] == product_name]['Category Id'].unique()
    available_departments = df_freq[df_freq['Product Name'] == product_name]['Department Id'].unique()

    category_id = st.selectbox("Category Id", available_categories)
    department_id = st.selectbox("Department Id", available_departments)
    order_item_quantity = st.number_input("Order Item Quantity", min_value=1, value=1)
    sales = st.number_input("Sales", min_value=0.0, value=299.98)

    # Predict discount rate
    if st.button("Predict Discount Rate"):
        predicted_rate = predict_discount_rate(order_date, product_name, market, customer_segment, category_id, department_id, order_item_quantity, sales, holidays, df_freq)
        st.write(f"Predicted Discount Rate: {predicted_rate}")
