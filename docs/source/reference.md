# ChiMerge

[https://sci2s.ugr.es/keel/pdf/algorithm/congreso/1992-Kerber-ChimErge-AAAI92.pdf](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/1992-Kerber-ChimErge-AAAI92.pdf)

ChiMerge Algorithm uses Chi-squared statistic to discretize attributes (numeric). In toad, we firstly transform Char/Object attributes to numeric with WOE function. The Algorithm  is clear in paper (i.e. ChiMerge Algorithm Part).

# Stepwise Regression

[https://link.springer.com/article/10.1007%2FBF02576123](https://link.springer.com/article/10.1007%2FBF02576123) [1]

[https://www.sciencedirect.com/science/article/pii/S0950584917305153?via%3Dihub](https://www.sciencedirect.com/science/article/pii/S0950584917305153?via%3Dihub) [2]

[http://www.jstor.org/stable/1434071](http://www.jstor.org/stable/1434071)[3]

Stepwise Regression （Forward/Backward/Stepwise, i.e. [2] 3.6. Stepwise Linear Regression）is uesed to reduce Low Information Gain Attributes and simplify the Final Model.

The Stepwise Regression Process[2]:

```eval_rst
.. image:: images/stepwise.png
   :width: 80%
   :align: center
```

# Scorecard Transformation

John Wiley & Sons, Inc., *Credit Risk Scorecards Developing and Implementing Intelligent Credit Scoring* (Final Scorecard Production Part)



Formula:

Score = Offset + Factor ∗ ln (odds)                    #odds: good:bad

Score + pdo = Offset + Factor ∗ ln (2 ∗ odds)   # pdo: points to double the odds

==>

pdo = Factor ∗ ln (2),  

Factor = pdo / ln (2);

Offset = Score - Factor ∗ ln (odds)

For example, if a scorecard were being scaled where the user wanted

odds of 50:1 at 600 points and wanted the odds to double every 20

points (i.e., pdo = 20), the factor and offset would be:

Factor = 20 / ln (2) = 28.8539

Offset = 600 – 28.8539 * ln (50) = 487.123

==>

Each score corresponding to each set of odds:

Score = 487.123 + 28.8539 * ln (odds)

Scorecard is developed with WOE as input, the formula can be modified as:

```eval_rst
.. image:: images/scorecard.png
   :width: 80%
   :align: center
```

WOE = weight of evidence for each grouped attribute

β = regression coefficient for each characteristic

a = intercept term from logistic regression

n = number of characteristics

k = number of groups (of attributes) in each characteristic
