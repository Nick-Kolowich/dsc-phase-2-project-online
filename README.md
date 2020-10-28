# King County Housing Analysis
# King County Housing Analysis

![png](https://github.com/Nick-Kolowich/dsc-phase-2-project-online/blob/master/images/King%20County%20Housing%20Prices.png)

Housing prices in Seattle by price quantile. Red areas indicate more affluent homes, clusters are see in town, most notably Bellevue, and around water. 

### Which features correlate most highly with price? 
<details>
    <summary> Expand </summary>

![png](https://github.com/Nick-Kolowich/dsc-phase-2-project-online/blob/master/images/fig2.png)


```python
# displays 90th percentile correlation values
corr_matrix_90 = correlation.describe(percentiles=[0.9])
corr_matrix_90['Price']
```
    count    12.000000
    mean      0.432650
    std       0.296210
    min       0.036031
    50%       0.424855
    90%       0.698532
    max       1.000000
    Name: Price, dtype: float64

</details>

Grade and Sq. Footage of a home are most closely correlated with price in the correlation matrix. 

###  Create Training & Testing Sets
<details>
    <summary> Expand </summary>
```python
# create target and features
target = data_z['Price']
features = data_z.drop(columns=['Price'], axis=1)

```python
# instantiate a linear regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# predict a linear model for both target and test sets
target_train = linear_reg_model.predict(X_train)
target_test = linear_reg_model.predict(X_test)

```python
# Compute and print r^2 and RMSE
print("r-squared: {}".format(round(linear_reg_model.score(X_train, y_train), 4)))
rmse = np.sqrt(mean_squared_error(y_train, target_train))
print("RMSE: {}".format(round(rmse, 4)))
```

    r-squared: 0.564
    RMSE: 0.6432
    

```python
# Perform 3-fold cross validation
cvscores_3 = cross_val_score(linear_reg_model, features, target, cv=3)
print("3-fold cross validation: {}".format(round(np.mean(cvscores_3),4)))

# perform 5-fold cross validation
cvscores_5 = cross_val_score(linear_reg_model, features, target, cv=5)
print("5-fold cross validation: {}".format(round(np.mean(cvscores_5), 4)))

# perform 10-fold cross validation
cvscores_10 = cross_val_score(linear_reg_model, features, target, cv=10)
print("10-fold cross validation: {}".format(round(np.mean(cvscores_10), 4)))
```

    3-fold cross validation: 0.5621
    5-fold cross validation: 0.5611
    10-fold cross validation: 0.5595

</details>

computed r-squared = 0.564
mean r-squared for 3,5, and 10 fold CVs = 0.5609

The linear model created for the training set should apply fairly well to the test data set.

### Initial OLS Regression Model


```python
# creating an OLS regression model
_target = 'Price'
_features = ['Bedrooms','Bathrooms','sqft_House','sqft_Lot','Floors','Condition','Grade','sqft_Above_Ground','sqft_Basement','sqft_Nearby_Homes','sqft_Nearby_Lots']

predictors = '+'.join(_features)
formula = _target + '~' + predictors
model = ols(formula=formula, data=data_z).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Price</td>      <th>  R-squared:         </th> <td>   0.564</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.564</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2795.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 28 Oct 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>01:55:30</td>     <th>  Log-Likelihood:    </th> <td> -21674.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21596</td>      <th>  AIC:               </th> <td>4.337e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21585</td>      <th>  BIC:               </th> <td>4.346e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td> 8.327e-17</td> <td>    0.004</td> <td> 1.85e-14</td> <td> 1.000</td> <td>   -0.009</td> <td>    0.009</td>
</tr>
<tr>
  <th>Bedrooms</th>          <td>   -0.1239</td> <td>    0.006</td> <td>  -21.309</td> <td> 0.000</td> <td>   -0.135</td> <td>   -0.113</td>
</tr>
<tr>
  <th>Bathrooms</th>         <td>   -0.0364</td> <td>    0.008</td> <td>   -4.712</td> <td> 0.000</td> <td>   -0.051</td> <td>   -0.021</td>
</tr>
<tr>
  <th>sqft_House</th>        <td>    0.2702</td> <td>    0.005</td> <td>   52.486</td> <td> 0.000</td> <td>    0.260</td> <td>    0.280</td>
</tr>
<tr>
  <th>sqft_Lot</th>          <td>    0.0059</td> <td>    0.006</td> <td>    0.905</td> <td> 0.365</td> <td>   -0.007</td> <td>    0.019</td>
</tr>
<tr>
  <th>Floors</th>            <td>    0.0013</td> <td>    0.006</td> <td>    0.207</td> <td> 0.836</td> <td>   -0.011</td> <td>    0.013</td>
</tr>
<tr>
  <th>Condition</th>         <td>    0.1046</td> <td>    0.005</td> <td>   22.169</td> <td> 0.000</td> <td>    0.095</td> <td>    0.114</td>
</tr>
<tr>
  <th>Grade</th>             <td>    0.3357</td> <td>    0.008</td> <td>   42.050</td> <td> 0.000</td> <td>    0.320</td> <td>    0.351</td>
</tr>
<tr>
  <th>sqft_Above_Ground</th> <td>    0.2021</td> <td>    0.005</td> <td>   36.900</td> <td> 0.000</td> <td>    0.191</td> <td>    0.213</td>
</tr>
<tr>
  <th>sqft_Basement</th>     <td>    0.1826</td> <td>    0.005</td> <td>   37.838</td> <td> 0.000</td> <td>    0.173</td> <td>    0.192</td>
</tr>
<tr>
  <th>sqft_Nearby_Homes</th> <td>    0.0463</td> <td>    0.007</td> <td>    6.204</td> <td> 0.000</td> <td>    0.032</td> <td>    0.061</td>
</tr>
<tr>
  <th>sqft_Nearby_Lots</th>  <td>   -0.0554</td> <td>    0.007</td> <td>   -8.488</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.043</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>16831.237</td> <th>  Durbin-Watson:     </th>  <td>   1.986</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>1031494.117</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.251</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>36.227</td>   <th>  Cond. No.          </th>  <td>1.49e+16</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 4.55e-28. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



### Refining the OLS Regression Model


```python
#building an OLS model with the significant features (p<0.05)
significant_features = ['Bedrooms','Bathrooms','sqft_House','Condition','Grade','sqft_Basement','sqft_Nearby_Homes','sqft_Nearby_Lots']
significant_features_orig = ['']
significant_predictors = '+'.join(significant_features)
formula = _target + '~' + significant_predictors
sig_model = ols(formula=formula, data=data_z).fit()
sig_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Price</td>      <th>  R-squared:         </th> <td>   0.564</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.564</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3493.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 28 Oct 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>02:04:44</td>     <th>  Log-Likelihood:    </th> <td> -21674.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21596</td>      <th>  AIC:               </th> <td>4.337e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21587</td>      <th>  BIC:               </th> <td>4.344e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     8</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td> 8.327e-17</td> <td>    0.004</td> <td> 1.85e-14</td> <td> 1.000</td> <td>   -0.009</td> <td>    0.009</td>
</tr>
<tr>
  <th>Bedrooms</th>          <td>   -0.1241</td> <td>    0.006</td> <td>  -21.373</td> <td> 0.000</td> <td>   -0.135</td> <td>   -0.113</td>
</tr>
<tr>
  <th>Bathrooms</th>         <td>   -0.0359</td> <td>    0.007</td> <td>   -4.964</td> <td> 0.000</td> <td>   -0.050</td> <td>   -0.022</td>
</tr>
<tr>
  <th>sqft_House</th>        <td>    0.4953</td> <td>    0.011</td> <td>   45.825</td> <td> 0.000</td> <td>    0.474</td> <td>    0.517</td>
</tr>
<tr>
  <th>Condition</th>         <td>    0.1044</td> <td>    0.005</td> <td>   22.354</td> <td> 0.000</td> <td>    0.095</td> <td>    0.114</td>
</tr>
<tr>
  <th>Grade</th>             <td>    0.3359</td> <td>    0.008</td> <td>   42.637</td> <td> 0.000</td> <td>    0.320</td> <td>    0.351</td>
</tr>
<tr>
  <th>sqft_Basement</th>     <td>    0.0739</td> <td>    0.005</td> <td>   13.776</td> <td> 0.000</td> <td>    0.063</td> <td>    0.084</td>
</tr>
<tr>
  <th>sqft_Nearby_Homes</th> <td>    0.0458</td> <td>    0.007</td> <td>    6.208</td> <td> 0.000</td> <td>    0.031</td> <td>    0.060</td>
</tr>
<tr>
  <th>sqft_Nearby_Lots</th>  <td>   -0.0513</td> <td>    0.005</td> <td>  -11.084</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.042</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>16819.177</td> <th>  Durbin-Watson:     </th>  <td>   1.986</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>1028216.622</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.248</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>36.173</td>   <th>  Cond. No.          </th>  <td>    5.29</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



### Checking for multicollinearity with VIF 


```python
data_columns = data_z[significant_features]
vif = [variance_inflation_factor(data_columns.values, i) for i in range(data_columns.shape[1])]
list(zip(significant_features, vif))
```




    [('Bedrooms', 1.6698022072022862),
     ('Bathrooms', 2.5860486707716666),
     ('sqft_House', 5.787499229746876),
     ('Condition', 1.0806635223846508),
     ('Grade', 3.0744056895501335),
     ('sqft_Basement', 1.4239018276165227),
     ('sqft_Nearby_Homes', 2.6933021146186102),
     ('sqft_Nearby_Lots', 1.0629364075575987)]



### Correlation Matrix of Significant Features 


```python
correlation_significant_features_df = data_z[significant_features]
correlation_significant_features_df.insert(loc=0, column='Price', value=data_z['Price'])
correlation_significant_features = correlation_significant_features_df.corr()

# creates the figure and axis for the subplots

fig1, ax1 = plt.subplots(figsize=(13, 8))

# creates a mask to remove the mirrored half of heatmap

mask = np.triu(np.ones_like(correlation_significant_features, dtype=np.bool))

# adjusts mask and dataframe

mask_sig_adj = mask[1:, :-1]
correlation_sig_adj = correlation_significant_features.iloc[1:,:-1].copy()

# plots heatmap

sns.heatmap(correlation_sig_adj, mask=mask_sig_adj, annot=True, fmt='.2f', cmap='Blues', linewidths=3, vmin=-0.5, vmax=0.9, cbar_kws={"shrink": .8})

# ytick adjustment

plt.xticks(rotation=60)
plt.show()
```


![png](output_22_0.png)


### Lasso Regression


```python
# performs lasso regression to determine most relevant features
features_ = data_z.drop('Price', axis=1).columns
lasso = Lasso(alpha=0.05)
lasso_coef = lasso.fit(features, target).coef_
plt.plot(range(len(features_)), lasso_coef)
plt.xticks(range(len(features_)), features_, rotation=90)
plt.ylabel('Coefficients')
plt.title('Lasso')
plt.show()
```


![png](output_24_0.png)


### Three Variable OLS Regression Model


```python
#building an OLS model with the 3 most explanatory variables
_target_ = 'Price'
_significant_features_ = ['sqft_House','Grade','sqft_Basement']
_significant_predictors_ = '+'.join(_significant_features_)
formula = _target_ + '~' + _significant_predictors_
three_var_model = ols(formula=formula, data=data_).fit()
three_var_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Price</td>      <th>  R-squared:         </th>  <td>   0.541</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.541</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   8494.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 28 Oct 2020</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>01:55:31</td>     <th>  Log-Likelihood:    </th> <td>-2.9896e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21596</td>      <th>  AIC:               </th>  <td>5.979e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21592</td>      <th>  BIC:               </th>  <td>5.980e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>-6.565e+05</td> <td> 1.36e+04</td> <td>  -48.300</td> <td> 0.000</td> <td>-6.83e+05</td> <td> -6.3e+05</td>
</tr>
<tr>
  <th>sqft_House</th>    <td>  156.5049</td> <td>    3.254</td> <td>   48.102</td> <td> 0.000</td> <td>  150.128</td> <td>  162.882</td>
</tr>
<tr>
  <th>Grade</th>         <td> 1.108e+05</td> <td> 2325.619</td> <td>   47.637</td> <td> 0.000</td> <td> 1.06e+05</td> <td> 1.15e+05</td>
</tr>
<tr>
  <th>sqft_Basement</th> <td>   78.0710</td> <td>    4.427</td> <td>   17.636</td> <td> 0.000</td> <td>   69.394</td> <td>   86.748</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>17102.423</td> <th>  Durbin-Watson:     </th>  <td>   1.976</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>1062486.894</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.332</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>36.710</td>   <th>  Cond. No.          </th>  <td>1.87e+04</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.87e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



### Final OLS model 


```python
#building an OLS model with only Grade and SqFt.
_target_ = 'Price'
significant_features_ = ['sqft_House','Grade']
significant_predictors_ = '+'.join(significant_features_)
formula = _target_ + '~' + significant_predictors_
two_var_model = ols(formula=formula, data=data_).fit()
two_var_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Price</td>      <th>  R-squared:         </th>  <td>   0.535</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.535</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>1.241e+04</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 28 Oct 2020</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>01:55:31</td>     <th>  Log-Likelihood:    </th> <td>-2.9912e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21596</td>      <th>  AIC:               </th>  <td>5.982e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21593</td>      <th>  BIC:               </th>  <td>5.983e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td>-6.028e+05</td> <td> 1.33e+04</td> <td>  -45.185</td> <td> 0.000</td> <td>-6.29e+05</td> <td>-5.77e+05</td>
</tr>
<tr>
  <th>sqft_House</th> <td>  184.1237</td> <td>    2.872</td> <td>   64.103</td> <td> 0.000</td> <td>  178.494</td> <td>  189.754</td>
</tr>
<tr>
  <th>Grade</th>      <td> 9.926e+04</td> <td> 2247.789</td> <td>   44.157</td> <td> 0.000</td> <td> 9.48e+04</td> <td> 1.04e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>16960.591</td> <th>  Durbin-Watson:     </th>  <td>   1.976</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>1005805.049</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.304</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>35.774</td>   <th>  Cond. No.          </th>  <td>1.80e+04</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.8e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



###  Scatter Plots of Price vs. Grade & Square Footage

#### Original Data


```python
# 3D plot for the original data
fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection='3d')

x=data_['Grade']
y=data_['sqft_House']
z=data_['Price']

x = np.array(x)
y = np.array(y)
z = np.array(z)

ax.scatter(x, y, z, c='springgreen', marker='o',linewidths=1, edgecolors='blue', alpha=0.8)
fig.set_size_inches(15,10)
plt.title('Price vs. Grade & Sqft. of House')
ax.set_xlabel('Grade')
ax.set_ylabel('sqft. of House')
ax.set_zlabel('Price')
ax.zaxis.set_tick_params(labelsize=8)

ax.ticklabel_format(axis='z', style='plain')

plt.show()
```


![png](output_31_0.png)


#### z-score Data 


```python
# 3D plot for the z-score data
fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection='3d')

x=data_z['Grade']
y=data_z['sqft_House']
z=data_z['Price']

x = np.array(x)
y = np.array(y)
z = np.array(z)

ax.scatter(x, y, z, c='springgreen', marker='o',linewidths=1, edgecolors='blue', alpha=0.8)
fig.set_size_inches(15,10)
plt.title('Z-score Price vs. Grade & Sqft. of House')
ax.set_xlabel('Grade')
ax.set_ylabel('sqft. of House')
ax.set_zlabel('Price')
ax.zaxis.set_tick_params(labelsize=8)

ax.ticklabel_format(axis='z', style='plain')

plt.show()
```


![png](output_33_0.png)


#### Log Transformed Data 


```python
# 3D plot for the log transformed data
fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection='3d')

x=data_log['Grade']
y=data_log['sqft_House']
z=data_log['Price']

x = np.array(x)
y = np.array(y)
z = np.array(z)

ax.scatter(x, y, z, c='springgreen', marker='o',linewidths=1, edgecolors='blue', alpha=0.8)
fig.set_size_inches(13,8)
plt.title('Log Transformed Price vs. Grade & Sqft. of House')
ax.set_xlabel('Grade')
ax.set_ylabel('sqft. of House')
ax.set_zlabel('Price')
ax.zaxis.set_tick_params(labelsize=8)

ax.ticklabel_format(axis='z', style='plain')

plt.show()
```


![png](output_35_0.png)


### Investigating Residuals 


```python
residplot = sm.graphics.qqplot(two_var_model.resid, dist=stats.norm, line='45', fit=True)
```


![png](output_37_0.png)



```python
sns.distplot(two_var_model.resid)
plt.ticklabel_format(style='plain')
plt.title('Distribution of Residuals')
plt.show()
```


![png](output_38_0.png)



```python
for i in range(90, 100):
    q = i / 100
    print('{} percentile: {}'.format(q, data_['Price'].quantile(q=q)))
```

    0.9 percentile: 887000.0
    0.91 percentile: 919994.5
    0.92 percentile: 950000.0
    0.93 percentile: 997967.5
    0.94 percentile: 1060000.0
    0.95 percentile: 1160000.0
    0.96 percentile: 1260000.0
    0.97 percentile: 1390000.0
    0.98 percentile: 1600000.0
    0.99 percentile: 1970000.0
    


```python
# remove all houses above $1,000,000
subset = data_[data_['Price'] <= 1000000]
print('Percent removed:', (round((len(data_) - len(subset))/len(data_), 5)))
outcome = 'Price'
predictors = '+'.join(significant_features_)
formula = outcome + '~' + predictors
model_under_1m = ols(formula=formula, data=subset).fit()
model_under_1m.summary()
```

    Percent removed: 0.06751
    




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Price</td>      <th>  R-squared:         </th>  <td>   0.441</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.441</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   7936.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 28 Oct 2020</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>01:55:33</td>     <th>  Log-Likelihood:    </th> <td>-2.6817e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20138</td>      <th>  AIC:               </th>  <td>5.363e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20135</td>      <th>  BIC:               </th>  <td>5.364e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td>-2.611e+05</td> <td> 8571.388</td> <td>  -30.458</td> <td> 0.000</td> <td>-2.78e+05</td> <td>-2.44e+05</td>
</tr>
<tr>
  <th>sqft_House</th> <td>   88.1682</td> <td>    1.904</td> <td>   46.314</td> <td> 0.000</td> <td>   84.437</td> <td>   91.900</td>
</tr>
<tr>
  <th>Grade</th>      <td> 7.416e+04</td> <td> 1420.151</td> <td>   52.220</td> <td> 0.000</td> <td> 7.14e+04</td> <td> 7.69e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>879.678</td> <th>  Durbin-Watson:     </th> <td>   1.959</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 998.133</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.534</td>  <th>  Prob(JB):          </th> <td>1.81e-217</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.219</td>  <th>  Cond. No.          </th> <td>1.76e+04</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.76e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
fig_drop_outliers = sm.graphics.qqplot(model_under_1m.resid, dist=stats.norm, line='45', fit=True)
```


![png](output_41_0.png)



```python
sns.distplot(model_under_1m.resid)
plt.ticklabel_format(style='plain')
plt.title('Distribution of Residuals')
plt.show()
```


![png](output_42_0.png)



```python
# 3D plot for the original data
fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection='3d')

x=subset['Grade']
y=subset['sqft_House']
z=subset['Price']

x = np.array(x)
y = np.array(y)
z = np.array(z)

ax.scatter(x, y, z, c='springgreen', marker='o',linewidths=1, edgecolors='blue', alpha=0.8)
fig.set_size_inches(15,10)
plt.title('Price vs. Grade & Sqft. of House under $1MM')
ax.set_xlabel('Grade')
ax.set_ylabel('sqft. of House')
ax.set_zlabel('Price')
ax.zaxis.set_tick_params(labelsize=8)

ax.ticklabel_format(axis='z', style='plain')

plt.show()
```


![png](output_43_0.png)


### Conclusion 

The final linear model has an r-squared of 0.535, which means it can only explain about 54% of the response variable variation. The features used to construct the final model has p-values less than 0.05 in the initial model, and were tested for multicollinearity using the litmus test of their variance inflation factors.

Their corresponding VIFs were: 

sqft_House: 5.787
Grade: 3.074
sqft_Basement: 1.424

Removing houses above $1MM makes the distribution of residuals much more normally distributed.
It brings skew down from 3.02 to 0.53 and kurtosis down from 35.77 to 3.2.


```python
jupyter nbconvert --to markdown README.ipynb
```


      File "<ipython-input-28-3c9a2e22f0f1>", line 1
        jupyter nbconvert --to markdown README.ipynb
                ^
    SyntaxError: invalid syntax
    



```python

```
