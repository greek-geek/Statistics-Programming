* Final SAS project for STAT202A;
* Complete the code;

* Import car data;
proc import out= work.data
datafile= "/folders/myfolders/study1/regression_auto.csv"
dbms=csv replace; getnames=yes; datarow=2;
run;

* Compute the correlation between car length and mpg;
proc corr data=data;
title 'Correlation between car length and mpg';
var mpg length;
run; 

* Make a scatterplot of price (x-axis) and mpg (y-axis);
proc sgplot data=data;
title 'Scatter Plot for Price vs MPG';
scatter X=price Y=mpg;
run;

*Make a box plot of mpg for foreign vs domestic cars;
proc sgplot data=data;
title 'MPG for Foreign and Domestic Cars';
vbox mpg / group=foreign;
run;

* Perform simple linear regression, y = mpg, x = price1; 
* Do NOT include the intercept term;
proc reg data=data; 
title 'Linear Regression, y=mpg, x=price1';
model mpg = price1 / noint; 
run;

* Perform linear regression, y = mpg, x1 = length, x2 = length^2; 
* Include the intercept term;
proc glm data=data; 
title 'Linear Regression, y=mpg, x1=length, x2=length^2';
model mpg = length length*length; 
run;
