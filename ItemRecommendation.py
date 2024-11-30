import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

class ProductRecommender:
    def __init__(self, data):
        # Preprocess and aggregate data
        self.original_data = self._preprocess_dataframe(data)
        self.prepare_data()
    

    def _preprocess_dataframe(self, df):
        # Ensure date columns are datetime
        date_columns = ['order_date', 'shipping_date', 'order date (DateOrders)', 'shipping_date (DateOrders)']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    print(f"Could not convert {col} to datetime: {e}")
        
        return df

    def prepare_data(self):
        df = self.original_data.copy()

        # Group data by Order Id (to treat each order as a single unit)
        df_grouped = df.groupby('Order Id').agg({
            'Customer Id': 'first',
            'Product Name': lambda x: list(x),  # List of products in the order
            'Category Name': 'first',  # Assumes all items in the order are from the same category
            'Order Item Quantity': 'sum',  # Sum of quantities for the order
            'Order Item Total': 'sum',  # Sum of total cost for the order
            'order_date': 'first'  # First date of the order (assuming all products share the same date)
        }).reset_index()

        self.customer_purchase_history = df_grouped.groupby('Customer Id').agg({
            'Product Name': 'sum',  # Concatenate all products from the customer's orders
            'Order Item Total': 'sum',  # Sum of total spend for the customer
            'Order Item Quantity': 'sum',  # Sum of total quantity purchased by the customer
        }).reset_index()

        # Prepare TF-IDF vectorizer for product features
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.product_features = self.tfidf.fit_transform(
            self.original_data['Product Name'].value_counts().index.to_list()
        )

        # Prepare product-level details for recommendations
        self.product_details = df.groupby('Product Name').agg({
            'Product Price': 'mean',
            'Category Name': 'first',
            'Order Item Quantity': 'sum',
            'Order Item Total': 'sum',
            'order_date': ['min', 'max']
        }).reset_index()
        # Flatten the multi-level columns
        self.product_details.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in self.product_details.columns.values]
        self.product_details.columns = [
            'Product Name', 'Product Price', 'Category Name', 
            'Total Quantity', 'Total Sales', 
            'First Sale Date', 'Last Sale Date'
        ]
        return self

    def recommend_products(self, customer_id, category=None, reference_date=None, top_n=5):
        if isinstance(reference_date, str):
            reference_date = pd.to_datetime(reference_date)
        elif reference_date is None:
            reference_date = self.original_data['order_date'].max()
    
        try:
            # Find customer's purchase history
            customer_history = self.customer_purchase_history[
                self.customer_purchase_history['Customer Id'] == customer_id
            ]
            
            # If no customer history, provide general recommendations
            if customer_history.empty:
                # Recommend top-selling products in the last 3 months
                recent_df = self.original_data[
                    (self.original_data['order_date'] > (reference_date - timedelta(days=90)))
                ]
                
                if category:
                    recent_df = recent_df[recent_df['Category Name'] == category]
                
                recent_products = recent_df.groupby('Product Name')['Order Item Quantity'].sum()
                recommendations = recent_products.nlargest(top_n).index.tolist()
                return recommendations
            
            # Get customer's previous product purchases
            previous_products = customer_history['Product Name'].iloc[0]
            
            # Filter out products not in the product details DataFrame
            available_previous_products = [
                product for product in previous_products if product in self.product_details['Product Name'].values
            ]
            
            if not available_previous_products:
                print(f"No available previous products in product details for customer {customer_id}.")
                return []
    
            # Calculate similarity between the previous products and all products
            similarities = []
            for prev_product in available_previous_products:
                prev_product_index = self.product_details[
                    self.product_details['Product Name'] == prev_product
                ].index[0]  # Get the first index of the matching product
                
                prev_product_feature = self.product_features[prev_product_index]  # Indexing sparse matrix
                
                # Calculate cosine similarity with all products
                product_similarities = cosine_similarity(
                    prev_product_feature, 
                    self.product_features
                )[0]  # Flattening the result for cosine similarities
                
                similarities.append(product_similarities)
    
            # Aggregate similarities
            avg_similarities = np.mean(similarities, axis=0)
    
            # Apply temporal and category filters
            temporal_mask = (
                (self.product_details['First Sale Date'] > (reference_date - timedelta(days=180))) &
                (self.product_details['First Sale Date'] <= reference_date)
            )
    
            # Apply the category filter
            if category:
                category_mask = (self.product_details['Category Name'] == category)
                combined_mask = temporal_mask & category_mask
            else:
                combined_mask = temporal_mask
    
            # Ensure that the mask aligns with the length of avg_similarities
            if len(avg_similarities) == len(self.product_details):
                avg_similarities[~combined_mask] = 0  # Zero out similarities for products not meeting criteria
            else:
                print(f"Warning: Length of avg_similarities ({len(avg_similarities)}) does not match product details length ({len(self.product_details)})")
    
            # Ensure the recommendations only come from products that match the available product details
            valid_products = self.product_details[combined_mask]
            valid_similarities = avg_similarities[combined_mask]
    
            if len(valid_similarities) == 0:
                print("No valid products match the criteria.")
                return []
    
            # Get top recommendations based on valid_similarities
            top_indices = valid_similarities.argsort()[::-1][:top_n]
            recommendations = valid_products.iloc[top_indices]['Product Name'].tolist()
            
            return recommendations
        
        except Exception as e:
            print(f"Error in recommendation: {e}")
            return []

    def get_customer_profile(self, customer_id):
        """
        Retrieve detailed customer profile
        """
        customer_info = self.original_data[
            self.original_data['Customer Id'] == customer_id
        ].drop_duplicates('Customer Id')
        
        customer_history = self.customer_purchase_history[
            self.customer_purchase_history['Customer Id'] == customer_id
        ]
        
        return {
            'Customer ID': customer_id,
            'Full Name': customer_info['Customer Full Name'].iloc[0] if not customer_info.empty else 'Unknown',
            'Segment': customer_info['Customer Segment'].iloc[0] if not customer_info.empty else 'Unknown',
            'Country': customer_info['Customer Country'].iloc[0] if not customer_info.empty else 'Unknown',
            # 'First Purchase Date': customer_history['First Purchase Date'].values[0] if not customer_history.empty else None,
            # 'Last Purchase Date': customer_history['Last Purchase Date'].values[0] if not customer_history.empty else None,
            # 'Total Purchases': customer_history['Total Spend'].values[0] if not customer_history.empty else 0,
            # 'Total Quantity': customer_history['Total Quantity'].values[0] if not customer_history.empty else 0
        }
    def recommend_products_by_market(self, market, category=None, reference_date=None, top_n=5):
        """
        Recommend products based on market purchase patterns, using customer-based recommendation logic.
    
        Args:
            market (str): The target market for recommendations.
            category (str, optional): Filter recommendations by category.
            reference_date (datetime or str, optional): The reference date for filtering recent products.
            top_n (int): Number of top products to recommend.
    
        Returns:
            List[str]: List of recommended product names.
        """
        if isinstance(reference_date, str):
            reference_date = pd.to_datetime(reference_date)
        elif reference_date is None:
            reference_date = self.original_data['order_date'].max()
    
        try:
            # Filter data for the specified market
            market_data = self.original_data[self.original_data['Market'] == market]
    
            if market_data.empty:
                print(f"No data available for the market: {market}")
                return []
    
            # Group market-level purchase history
            market_purchase_history = market_data.groupby('Product Name').agg({
                'Order Item Quantity': 'sum',
                'Order Item Total': 'sum'
            }).reset_index()
    
            # Rank products by purchase frequency within the market
            market_purchase_history = market_purchase_history.sort_values(
                by='Order Item Quantity', ascending=False
            )
    
            # If category filter is applied, narrow down the market data
            if category:
                market_data = market_data[market_data['Category Name'] == category]
    
            # Gather features of market-purchased products for similarity calculation
            market_purchased_products = market_purchase_history['Product Name'].tolist()
            available_products = [
                product for product in market_purchased_products if product in self.product_details['Product Name'].values
            ]
    
            if not available_products:
                print(f"No available products in product details for market: {market}.")
                return []
    
            # Calculate similarity between market-purchased products and all products
            similarities = []
            for product in available_products:
                product_index = self.product_details[
                    self.product_details['Product Name'] == product
                ].index[0]
    
                product_feature = self.product_features[product_index]  # Indexing sparse matrix
    
                # Calculate cosine similarity with all products
                product_similarities = cosine_similarity(
                    product_feature,
                    self.product_features
                )[0]  # Flatten the result
    
                similarities.append(product_similarities)
    
            # Aggregate similarities across all products
            avg_similarities = np.mean(similarities, axis=0)
    
            # Apply temporal and category filters
            temporal_mask = (
                (self.product_details['Last Sale Date'] > (reference_date - timedelta(days=30))) &
                (self.product_details['First Sale Date'] <= reference_date)
            )
    
            if category:
                category_mask = (self.product_details['Category Name'] == category)
                combined_mask = temporal_mask & category_mask
            else:
                combined_mask = temporal_mask
    
            # Ensure the similarities align with valid products
            if len(avg_similarities) == len(self.product_details):
                avg_similarities[~combined_mask] = 0
            else:
                print(f"Warning: Similarities length mismatch: {len(avg_similarities)} vs {len(self.product_details)}")
    
            # Get valid products
            valid_products = self.product_details[combined_mask]
            valid_similarities = avg_similarities[combined_mask]
    
            if len(valid_similarities) == 0:
                print("No valid products match the criteria for this market.")
                return []
    
            # Get top recommendations
            top_indices = valid_similarities.argsort()[::-1][:top_n]
            recommendations = valid_products.iloc[top_indices]['Product Name'].tolist()
    
            return recommendations
    
        except Exception as e:
            print(f"Error in market-based recommendation: {e}")
            return []
