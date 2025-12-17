# The Relationship between Sustainability and Economic Performance of Countries: Can Sustainability Indicators Predict Economic Outcomes?
**Category:** Data Analysis & Visualization

**By:** Gabriel Pinto Ferreira

Over past decades, environmental challenges, social inequality, and ethical governance have made sustainability a global concern and an increasingly important element in policy decisions and economic development. In this context, it becomes relevant to examine how differences in sustainability commitment are associated with differences in countries' economic outcomes. This project aims to explore these relationships by analysing how sustainability indicators and economic performance vary across countries over time.

This analysis will use publicly available indicators from the World Bank, covering the period 2000-2023. These indicators belong to four categories:
- **Evironmental:** CO2 emissions, fossil-fuel energy consumption, renewable energy consumption, renewable electricity output, methane and nitrous oxide emissions
- **Social:** Gini index, unemployment, and economic and social rights performance score
- **Governance:** Control of corruption, and political stability index
- **Economic Performance:** GDP growth, GDP per capita, inflation, foreign direct investment, and R&D expenditure

A sample of approximately 50 countries will be selected based on data availability, ensuring representation across regions and income levels.

The dataset will be organised as a panel (country - year). That way, I can analyse, through descriptive statistics, how changes in sustainability indicators and economic outcomes evolve within the countries over time, while also accounting for structural differences across countries.

I will estimate a fixed-effects panel regression model where GDP growth is the dependent variable and sustainability indicators serve as explanatory variables:

```math
\text{GDP Growth}_{i,t}
= \beta_0 
+ \beta_1 X^{(\text{env})}_{i,t}
+ \beta_2 X^{(\text{soc})}_{i,t}
+ \beta_3 X^{(\text{gov})}_{i,t}
+ \gamma_i + \delta_t + \varepsilon_{i,t}
```

Where:

```math
  i = \text{country},\quad t = \text{year},
```
```math
  \gamma_i = \text{country fixed effects},\quad \delta_t = \text{year fixed effects},
```
```math
  X^{(\text{env})}_{i,t},\ X^{(\text{soc})}_{i,t},\ X^{(\text{gov})}_{i,t} \text{ are the environmental, social, and governance indicators.}
```
This model will quantify the direction and strength of associations between sustainability indicators and GDP growth.

To complement the regression, I will build predictive models with GDP growth as the target variable, and sustainability indicators are features. Three models will be tested:
- Linear Regression
- Random Forest Regressor
- Gradient Boost Regressor

Models will be evaluated and formally compared using metrics such as mean squared error, the R-squared, and cross-validation scores. This will assess how well sustainability indicators can help predict economic performance and how different models compare.

The analysis is descriptive and predictive only, not causal. The results do not allow any statements about whether sustainability indicators cause higher or lower economic performance.

Expected challenges include incomplete data coverage, uneven time availability for certain indicators, and regional differences in data quality. These issues will be addressed by filtering countries and years with sufficient data and ensuring a diverse sample across regions. Correlated indicators may also affect results; redundant variables will be removed when necessary.

If time permits, I would like to make a dashboard that would bring an easier and more pleasant way to view the results.

To conclude, this project will provide a perspective on how sustainability indicators and economic outcomes co-vary. By combining panel regression with machine-learning models, it will focus on identifying patterns and associations, while strengthening my skills in Python and data analysis.
