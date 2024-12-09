import pandas as pd
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

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

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

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, SimpleRNN, Reshape
from tensorflow.keras.losses import MeanSquaredError,MeanAbsoluteError

# Step 1: Encode categorical features as integers
categorical_columns = ['Product_Name','Market','CustomerSegment']

# Use LabelEncoder for each categorical column
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    sales_df[col] = le.fit_transform(sales_df[col])
    label_encoders[col] = le
# Step 2: Scale numeric features
numeric_features = [
     'Order_Item_Quantity', 'Sales', 'Category_Id','Department_Id', 
    'Year','Day', 'Weekday', 'IsHoliday'
]
scaler = MinMaxScaler()
sales_df[numeric_features] = scaler.fit_transform(sales_df[numeric_features])

# Step 3: Split the data into features (X) and target (y)
target = 'Order_Item_Discount'
X_categorical = sales_df[categorical_columns]
X_numeric = sales_df[numeric_features]
y = sales_df[target]

# Train-test split
X_categorical_train, X_categorical_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
    X_categorical, X_numeric, y, test_size=0.2, random_state=42
)

# Combine categorical and numeric features
X_train_combined = pd.concat([X_categorical_train, X_numeric_train], axis=1)
X_test_combined = pd.concat([X_categorical_test, X_numeric_test], axis=1)

# Create the neural network model
def create_sales_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation="relu")(inputs)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="linear")(x)  # Regression output for discount rate
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    return model

# Create and train the model
model = create_sales_model(X_train_combined.shape[1])
history = model.fit(
    X_train_combined, y_train,
    validation_data=(X_test_combined, y_test),
    epochs=100,
    batch_size=32,
    verbose=1
)

# Save the LabelEncoders and MinMaxScaler in a single file
encoders_and_scaler = {'label_encoders': label_encoders, 'scaler': scaler}
joblib.dump(encoders_and_scaler, 'encoders_and_scaler.pkl')
# Save the model
model.save('discount_threshold_model.h5')


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

# Load the encoders, scaler, and model for deployment
encoders_and_scaler = joblib.load('encoders_and_scaler.pkl')
label_encoders = encoders_and_scaler['label_encoders']
scaler = encoders_and_scaler['scaler']
loaded_model = load_model('discount_threshold_model.h5')

# Test sample as a dictionary
test_sample = {
    'order_date_(DateOrders)': ['2015-01-01 00:00:00'],
    'Product_Name': ['Diamondback Women\'s Serene Classic Comfort Bi'],
    'Market': ['LATAM'],
    'CustomerSegment': ['LoyalCustomer'],
    'Category_Id': [43],
    'Department_Id': [7],
    'Order_Item_Quantity': [1],
    'Sales': [299.980011],
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
print(f"Predicted Discount Rate: {predicted_discount_rate[0][0]}")
