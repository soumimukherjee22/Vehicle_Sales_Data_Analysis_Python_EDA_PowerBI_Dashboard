Total Revenue =sum\['vehicle\_sales\_clean'(sellingprice)]



Avg Selling Price = average\['vehicle\_sales\_clean'(sellingprice)]



Avg MMR = AVERAGE(vehicle\_sales\_clean\[mmr])





**Avg vs MMR:** Avg Price vs MMR = \[Avg Selling Price] - \[Avg MMR]



Avg Price vs MMR % =

DIVIDE(

&#x20;   \[Avg Selling Price] - \[Avg MMR],

&#x20;   \[Avg MMR]

)



Price to MMR Ratio =

DIVIDE(\[Avg Selling Price], \[Avg MMR])



Pricing Status =

IF(

&#x20;   \[Avg Selling Price] > \[Avg MMR],

&#x20;   "Above Market",

&#x20;   "Below Market"

)







Units Sold = count(vehicle\_sales\_clean\[vin])



% Above MMR = DIVIDE(

&#x20;   CALCULATE(COUNT(vehicle\_sales\_clean\[vin]), vehicle\_sales\_clean\[above\_mmr] = TRUE()),

&#x20;   COUNT(vehicle\_sales\_clean\[vin])

)



Total Price Difference = SUM(vehicle\_sales\_clean\[price\_vs\_market])



Avg Price Diff % = AVERAGE(vehicle\_sales\_clean\[price\_vs\_market\_pct])



Monthly Sales =

CALCULATE(

&#x20;   \[Total Sales],

&#x20;   ALLEXCEPT(vehicle\_sales\_clean, vehicle\_sales\_clean\[sale\_year], vehicle\_sales\_clean\[sale\_month])

)



YoY Growth % =

DIVIDE(

&#x20;   \[Total Sales] - CALCULATE(\[Total Sales], SAMEPERIODLASTYEAR(vehicle\_sales\_clean\[saledate])),

&#x20;   CALCULATE(\[Total Sales], SAMEPERIODLASTYEAR(vehicle\_sales\_clean\[saledate]))

)



Quarterly Sales =

CALCULATE(

&#x20;   \[Total Sales],

&#x20;   VALUES(vehicle\_sales\_clean\[quarter])

)



Avg Vehicle Age = AVERAGE(vehicle\_sales\_clean\[vehicle\_age])



Avg Mileage = AVERAGE(vehicle\_sales\_clean\[odometer])



Avg Price by Condition =

AVERAGEX(

&#x20;   VALUES(vehicle\_sales\_clean\[cond\_bucket]),

&#x20;   CALCULATE(AVERAGE(vehicle\_sales\_clean\[sellingprice]))

)



Sales Contribution % =

DIVIDE(

&#x20;   \[Total Sales],

&#x20;   CALCULATE(\[Total Sales], ALL(vehicle\_sales\_clean))

)



Advanced / Recruiter-Level Metrics ⭐



Price Efficiency = DIVIDE(\[Avg Selling Price], \[Avg MMR])



Revenue per Age = DIVIDE(\[Total Sales], \[Avg Vehicle Age])



Sales Std Dev = STDEVX.P(vehicle\_sales\_clean, vehicle\_sales\_clean\[sellingprice])

