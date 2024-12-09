import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Custom Store Dashboard", page_icon=":bar_chart:", layout="wide")

st.title(" :bar_chart: Custom Store Data Analysis")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

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
