# friendly-succotash
Analyzing Customer Churn
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load and preview data
shopping_data = pd.read_csv("online_shopping_session_data.csv")
shopping_data.head()

## Calculate the online purchase rates during online shopping sessions by customer type for November and December.
# Subset dataframe for November and December data
shopping_Nov_Dec = shopping_data[shopping_data['Month'].isin(['Nov', 'Dec'])]

# Preview to make sure the subset is correct
print(shopping_Nov_Dec.head())

# Make sure we only have November and December data
print(shopping_Nov_Dec['Month'].unique())

# Get session frequency stats by CustomerType and Purchase
count_session = shopping_Nov_Dec.groupby(['CustomerType'])['Purchase'].value_counts()
print(count_session)

# Total number of session by CustomerType
total_new_customer = np.sum(count_session['New_Customer'])
total_returning_customer = np.sum(count_session['Returning_Customer'])

# Total number of purchase by CustomerType
purchase_new_customer = count_session[('New_Customer', 1)]
purchase_returning_customer = count_session[('Returning_Customer', 1)]

# Calculate purchase rates
purchase_rate_new = purchase_new_customer / total_new_customer
purchase_rate_returning = purchase_returning_customer / total_returning_customer

# Therefore, the online purchase for the returning customers is lower than that of the new customers.
purchase_rates = {"Returning_Customer": purchase_rate_returning, "New_Customer": purchase_rate_new}
print(purchase_rates)

## Identify the strongest correlation in total time spent between different types of pages visited by the returning customers during the months of November and December. 
# Calculate correlation with pandas
cor_admin_info = shopping_Nov_Dec['Administrative_Duration'].corr(shopping_Nov_Dec['Informational_Duration'])
cor_admin_product = shopping_Nov_Dec['Administrative_Duration'].corr(shopping_Nov_Dec['ProductRelated_Duration'])
cor_product_info = shopping_Nov_Dec['ProductRelated_Duration'].corr(shopping_Nov_Dec['Informational_Duration'])

print(cor_admin_info)
print(cor_admin_product)
print(cor_product_info)

# Another way to solve this is to use Scipy pearsonr function
# cor_admin_info = stats.pearsonr(shopping_Nov_Dec['Administrative_Duration'], shopping_Nov_Dec['Informational_Duration'])

# Another way to solve this is to use Pandas correlation matrix
# shopping_Nov_Dec[['Administrative_Duration','Informational_Duration','ProductRelated_Duration' ]].corr()

# Store top correlation
top_correlation = {"pair": ('Administrative_Duration', 'ProductRelated_Duration'), "correlation": cor_admin_product}
print(top_correlation)

## A new campaign for the returning customers will boost the purchase rate by 15%. 
## What is the likelihood of achieving at least 100 sales out of 500 online shopping sessions for the returning customers?

# Purchase is a binomial random variable taking the value of either 0 or 1
# We know that the current purchase rate for the returning customers is
print("Current purchase rate for the returning customer:", purchase_rate_returning)

# 15% Increase in this rate would be
increased_purchase_rate_returning = 1.15 * purchase_rate_returning
print("Increased purchase rate for the returning customer:", increased_purchase_rate_returning)

# First, we find the likelihood of having <100 sales of 500 sessions
# We can find this from binomial cdf
prob_sales_100_less = stats.binom.cdf(k=100, n=500, p=increased_purchase_rate_returning)
print("probability of having <100 sales:", prob_sales_100_less)

# Then, to find the probability of having 100 or more sales is 1-prob_sales_100_less
prob_at_least_100_sales = 1 - prob_sales_100_less
print("probability of having at least 100 sales:", prob_at_least_100_sales)

# Plotting the binomial probability distribution
n_sessions = 500
k_values = np.arange(500) + 1
p_binom_values = [stats.binom.pmf(k, n_sessions, increased_purchase_rate_returning) for k in k_values ] 
plt.bar(k_values, p_binom_values) 
plt.vlines(100, 0, 0.08, color='r', linestyle='dashed', label="sales=100")
plt.xlabel("number of sales")
plt.ylabel("probability")
plt.legend()
plt.show()

As a Data Analyst for a startup company with a newly launched online shopping platform, I was tasked with analyzing customer browsing behavior to provide actionable insights for the marketing team. The focus was on understanding shopper behavior during November and December, the busiest months for online sales. Specifically, I was required to identify two key customer groups: those with a low purchase rate and returning customers. The goal was to determine the probability of these groups making a purchase during a new marketing campaign, helping the team gauge the potential success of next year’s sales strategy.

Key Insights and Results: Current Purchase Rates: Returning customers currently have a purchase rate of 19.56%, which is lower than that of new customers. Impact of New Campaign: A targeted campaign for returning customers is projected to increase their purchase rate by 15%, raising it to 22.49%. Sales Probability: There is a 90.12% probability of achieving at least 100 sales during the campaign, indicating strong potential for success.

Impact and Recommendations: Targeted Campaigns: Focus on returning customers with personalized offers such as loyalty rewards, exclusive discounts, or early access to sales to boost engagement and conversions. For low purchase rate customers, implement strategies like time-sensitive promotions or free shipping to encourage purchases. Optimized Customer Experience: Enhance product pages by improving navigation, highlighting reviews, and offering personalized recommendations to increase engagement and conversions. Sales Forecasting: Use the probability model to set realistic sales targets and allocate resources effectively for future campaigns.

Approach: Data Analysis: Examined customer browsing behavior (e.g., session duration, pages viewed) and calculated purchase rates for key customer groups. Identified correlations between browsing patterns and purchase likelihood. Probability Modeling: Used binomial probability to estimate sales targets, including the likelihood of achieving at least 100 sales during the campaign. Key Metrics: Delivered current purchase rates, predicted campaign-driven increases, and probabilities of achieving sales goals.

Limitations: The dataset only covered November and December, limiting insights into year-round customer behavior. External factors such as competitor promotions or economic conditions were not considered in the analysis.

Skills Demonstrated: This project demonstrates my ability to analyze customer behavior, apply statistical modeling, and deliver actionable insights to drive business decisions. By identifying key customer groups and predicting their likelihood of purchase, I provided the marketing team with data-driven strategies to optimize campaigns, enhance customer experience, and achieve sales growth.
