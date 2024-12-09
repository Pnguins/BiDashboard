import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import plotly.express as px
import plotly.figure_factory as ff
from discountPrediction import create_dashboard_page, load_and_preprocess_data

st.set_page_config(page_title="Finale De Intelligence D Affaires", page_icon=":bar_chart:", layout="wide")

st.title(" :bar_chart: Finale De Intelligence D Affaires")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Add the directory containing ItemRecommendation.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ItemRecommendation import ProductRecommender

# Load the data
@st.cache_data
def load_data():
    # Replace with the path to your dataset
    return pd.read_csv("data/Inventory.csv", encoding_errors='ignore')

# Initialize the recommender system
@st.cache_resource
def initialize_recommender(data):
    return ProductRecommender(data)

# Load data
data = load_data()

# Initialize recommender
recommender = initialize_recommender(data)

# Sidebar Navigation
st.sidebar.title("Navigation")
dashboard = st.sidebar.radio("Choose Dashboard", ["Customer Recommendations", "Market Recommendations","Dashboard","Discount Prediction"])

if dashboard == "Customer Recommendations":
    st.title("Customer-Based Product Recommendations")

    # Recommendation range slider at the top
    default_top_n = 5
    top_n = st.slider("Number of Product Recommendations", min_value=1, max_value=20, value=default_top_n)

    # Create a searchable table for customers
    customer_table = data[['Customer Id', 'Customer Full Name', 'Customer Segment', 'Customer Country']].drop_duplicates()
    search_customer = st.text_input("Search Customer by Name or ID")
    filtered_customers = customer_table[
        customer_table['Customer Full Name'].str.contains(search_customer, case=False, na=False) |
        customer_table['Customer Id'].astype(str).str.contains(search_customer, case=False, na=False)
    ]

    # Add a 'Selected' column to the DataFrame
    filtered_customers = filtered_customers.copy()
    filtered_customers['Selected'] = False

    # Display customer search results in an interactive table
    st.write("Customer Search Results:")
    edited_customers = st.data_editor(
        filtered_customers,
        disabled=('Customer Id', 'Customer Full Name', 'Customer Segment', 'Customer Country'),
        hide_index=True,
        column_config={
            "Selected": st.column_config.CheckboxColumn(
                default=False,
            )
        }
    )

    # Find the selected customers
    selected_customers = edited_customers[edited_customers['Selected']]['Customer Id'].tolist()

    # Recommendation section
    if selected_customers:
        # If multiple customers are selected, take the first one
        selected_customer_id = selected_customers[0]

        # Get customer profile
        customer_profile = recommender.get_customer_profile(selected_customer_id)

        # Display customer profile
        st.subheader("Customer Profile")
        col1 = st.columns(1)[0]
        with col1:
            st.write(f"**Customer ID:** {customer_profile['Customer ID']}")
            st.write(f"**Name:** {customer_profile['Full Name']}")
            st.write(f"**Segment:** {customer_profile['Segment']}")

        # Get and display recommendations
        st.subheader(f"Top {top_n} Product Recommendations")

        # Category filter (optional)
        category_filter = st.selectbox(
            "Filter Recommendations by Category (Optional)",
            ["All"] + data['Category Name'].unique().tolist()
        )

        # Get recommendations
        if category_filter == "All":
            recommendations = recommender.recommend_products(
                selected_customer_id,
                top_n=top_n
            )
        else:
            recommendations = recommender.recommend_products(
                selected_customer_id,
                category=category_filter,
                top_n=top_n
            )

        # Display recommendations
        if recommendations:
            # Create a DataFrame for recommendations with customer name
            recommendation_details = data[data['Product Name'].isin(recommendations)][
                ['Product Name', 'Product Price', 'Category Name']
            ].drop_duplicates()

            # Append customer first and last name to the DataFrame
            recommendation_details['Customer First Name'] = customer_profile['Full Name'].split()[0]

            # Add inventory metrics
            recommendation_details['DemandRate'] = recommendation_details['Product Name'].apply(
                lambda x: recommender.get_product_demand_rate_and_lead_time(x).get('DemandRate', 'N/A')
            )
            recommendation_details['LeadTime'] = recommendation_details['Product Name'].apply(
                lambda x: recommender.get_product_demand_rate_and_lead_time(x).get('LeadTime', 'N/A')
            )

            st.dataframe(recommendation_details)

            # Optional: Visualize recommendation details
            st.bar_chart(recommendation_details.set_index('Product Name')['Product Price'])
        else:
            st.warning("No recommendations found for this customer.")

elif dashboard == "Market Recommendations":
    st.title("Market-Based Product Recommendations")

    # Market recommendation similar to the previous implementation
    # Create a map visualization for markets
    market_data = data[['Market', 'Latitude', 'Longitude']].drop_duplicates()
    market_data = market_data.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
    st.map(market_data)
    # Convert the order date column to datetime first
    data['order date (DateOrders)'] = pd.to_datetime(data['order date (DateOrders)'])

    # Get the min and max dates from your dataset
    min_date = data['order date (DateOrders)'].min().date()
    max_date = data['order date (DateOrders)'].max().date()
    # Select a market
    selected_market = st.selectbox("Select Market", market_data['Market'].unique())
    category = st.selectbox("Filter by Category (Optional)", ["All"] + data['Category Name'].unique().tolist())
    # Use these dates for min_value and max_value, and set the default to somewhere in the middle
    reference_date = st.date_input(
        "Filter by Date",
        value=min_date,  # Midpoint of the date range
        min_value=min_date,
        max_value=max_date,
    )
    top_n = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)
    # Generate recommendations for the market
    if st.button("Get Market Recommendations"):
        # Convert reference_date to datetime if needed in your recommender method
        reference_datetime = pd.Timestamp(reference_date)

        # Use the market-based recommendation method
        if category == "All":
            market_recommendations = recommender.recommend_products_by_market(
                selected_market,
                reference_date=reference_datetime,  # Pass as datetime
                top_n=top_n
            )
        else:
            market_recommendations = recommender.recommend_products_by_market(
                selected_market,
                category=category,
                reference_date=reference_datetime,  # Pass as datetime
                top_n=top_n
            )

        # Display market recommendations
        if market_recommendations:
            # Create a DataFrame for recommendations
            recommendation_details = data[data['Product Name'].isin(market_recommendations)][
                ['Product Name', 'Product Price', 'Category Name']
            ].drop_duplicates()

            st.dataframe(recommendation_details)

            # Optional: Visualize recommendation details
            st.bar_chart(recommendation_details.set_index('Product Name')['Product Price'])
        else:
            st.warning(f"No recommendations found for market: {selected_market}")
elif dashboard == "Dashboard":
    # File Uploader
    fl = st.file_uploader(":file_folder: Upload a file", type=(["csv","txt","xlsx","xls"]))
    if fl is not None:
        filename = fl.name
        st.write(filename)
        df = pd.read_csv(filename)
    else:
        # Provide a default path if needed
        df = pd.read_csv("data/DataCoSupplyChainDataset.csv",encoding_errors='ignore')

    # Convert order date to datetime
    df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"])

    # Get date range
    startDate = df["order date (DateOrders)"].min()
    endDate = df["order date (DateOrders)"].max()

    col1, col2 = st.columns((2))
    with col1:
        date1 = pd.to_datetime(st.date_input("Start Date", startDate))
    with col2:
        date2 = pd.to_datetime(st.date_input("End Date", endDate))

    # Filter by date
    df = df[(df["order date (DateOrders)"] >= date1) & (df["order date (DateOrders)"] <= date2)].copy()

    # Sidebar Filters
    st.sidebar.header("Choose your filters:")

    # Region Filter
    region = st.sidebar.multiselect("Pick your Region", df["Order Region"].unique())
    if not region:
        df2 = df.copy()
    else:
        df2 = df[df["Order Region"].isin(region)]

    # Country Filter
    country = st.sidebar.multiselect("Pick the Country", df2["Order Country"].unique())
    if not country:
        df3 = df2.copy()
    else:
        df3 = df2[df2["Order Country"].isin(country)]

    # City Filter
    city = st.sidebar.multiselect("Pick the City", df3["Order City"].unique())

    # Product Name Filter
    product = st.sidebar.multiselect("Pick the Product Name", df3["Product Name"].unique())

    # Filtering Logic
    if not region and not country and not city and not product:
        filtered_df = df
    elif product:
        filtered_df = df3[df3["Product Name"].isin(product)]
    elif not country and not city:
        filtered_df = df[df["Order Region"].isin(region)]
    elif not region and not city:
        filtered_df = df[df["Order Country"].isin(country)]
    elif country and city:
        filtered_df = df3[df["Order Country"].isin(country) & df3["Order City"].isin(city)]
    elif region and city:
        filtered_df = df3[df["Order Region"].isin(region) & df3["Order City"].isin(city)]
    elif region and country:
        filtered_df = df3[df["Order Region"].isin(region) & df3["Order Country"].isin(country)]
    elif city:
        filtered_df = df3[df3["Order City"].isin(city)]
    else:
        filtered_df = df3[df3["Order Region"].isin(region) &
                        df3["Order Country"].isin(country) &
                        df3["Order City"].isin(city)]

    # Category Sales Analysis
    category_df = filtered_df.groupby(by=["Category Name"], as_index=False)["Sales"].sum()

    col1, col2 = st.columns((2))
    with col1:
        st.subheader("Category wise Sales")
        fig = px.bar(category_df, x="Category Name", y="Sales",
                    text=['${:,.2f}'.format(x) for x in category_df["Sales"]],
                    template="seaborn")
        st.plotly_chart(fig, use_container_width=True, height=200)

    with col2:
        st.subheader("Delivery Status Distribution")
        fig = px.pie(filtered_df, values="Sales", names="Delivery Status", hole=0.5)
        fig.update_traces(text=filtered_df["Delivery Status"], textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Modify time series analysis section
    filtered_df["month_year"] = filtered_df["order date (DateOrders)"].dt.to_period("M")
    st.subheader('Time Series Analysis')

    # Create a custom sorting dataframe
    linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y : %b"))["Sales"].sum()).reset_index()

    # Convert month_year to datetime for sorting
    linechart["month_year"] = pd.to_datetime(linechart["month_year"], format="%Y : %b")

    # Sort by the actual order date
    linechart = linechart.sort_values(by="month_year")

    fig2 = px.line(linechart, x="month_year", y="Sales", labels={"Sales": "Amount"}, height=500, width=1000, template="gridon")
    st.plotly_chart(fig2, use_container_width=True)

    # Sales vs Profit Scatter Plot
    data1 = px.scatter(filtered_df, x="Sales", y="Order Profit Per Order",
                    size="Order Item Quantity",
                    color="Shipping Mode")
    data1['layout'].update(
        title="Relationship between Sales and Profits",
        titlefont=dict(size=20),
        xaxis=dict(title="Sales", titlefont=dict(size=19)),
        yaxis=dict(title="Profit per Order", titlefont=dict(size=19))
    )
    st.plotly_chart(data1, use_container_width=True)

    # Shipping Analysis
    st.subheader("Shipping Performance")
    shipping_analysis = filtered_df.groupby("Shipping Mode").agg({
        "Days for shipping (real)": "mean",
        "Sales": "sum"
    }).reset_index()
    fig_shipping = px.bar(shipping_analysis, x="Shipping Mode", y="Days for shipping (real)",
                        color="Sales", title="Average Shipping Days by Mode")
    st.plotly_chart(fig_shipping, use_container_width=True)
    # Product Trend Analysis
    st.subheader("Product Sales Trend Analysis")

    # Monthly Product Sales
    monthly_product_sales = filtered_df.groupby([
        filtered_df["order date (DateOrders)"].dt.to_period("M").astype(str),
        "Product Name"
    ])["Sales"].sum().reset_index()

    # Pivot table for monthly product sales
    monthly_pivot = monthly_product_sales.pivot(
        index="order date (DateOrders)",
        columns="Product Name",
        values="Sales"
    ).fillna(0)

    # Weekly Product Sales
    weekly_product_sales = filtered_df.groupby([
        filtered_df["order date (DateOrders)"].dt.to_period("W").astype(str),
        "Product Name"
    ])["Sales"].sum().reset_index()

    # Pivot table for weekly product sales
    weekly_pivot = weekly_product_sales.pivot(
        index="order date (DateOrders)",
        columns="Product Name",
        values="Sales"
    ).fillna(0)

    # Tabs for monthly and weekly views
    tab1, tab2 = st.tabs(["Monthly Product Trend", "Weekly Product Trend"])

    with tab1:
        # Monthly trend line chart
        fig_monthly = px.line(
            monthly_pivot.reset_index(),
            x="order date (DateOrders)",
            y=monthly_pivot.columns.tolist(),
            title="Monthly Sales Trend by Product",
            labels={"value": "Sales", "variable": "Product Name"},
            height=500
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Monthly product sales data
        with st.expander("Monthly Product Sales Data"):
            st.dataframe(monthly_pivot)

    with tab2:
        # Weekly trend line chart
        fig_weekly = px.line(
            weekly_pivot.reset_index(),
            x="order date (DateOrders)",
            y=weekly_pivot.columns.tolist(),
            title="Weekly Sales Trend by Product",
            labels={"value": "Sales", "variable": "Product Name"},
            height=500
        )
        st.plotly_chart(fig_weekly, use_container_width=True)

        # Weekly product sales data
        with st.expander("Weekly Product Sales Data"):
            st.dataframe(weekly_pivot)

    # Top 5 Products Analysis
    st.subheader("Top 5 Products by Total Sales")
    top_products = filtered_df.groupby("Product Name")["Sales"].sum().nlargest(5)
    fig_top_products = px.bar(
        top_products.reset_index(),
        x="Product Name",
        y="Sales",
        title="Top 5 Products by Total Sales"
    )
    st.plotly_chart(fig_top_products, use_container_width=True)
    # Download Options
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Filtered Data',
                    data=csv,
                    file_name="Filtered_Data.csv",
                    mime="text/csv")
elif dashboard == "Discount Prediction":
    create_dashboard_page()