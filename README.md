This is my second attempt at a Kaggle competition.  I spent 2 afternoons on completing it.  

https://www.kaggle.com/c/cat-in-the-dat/overview

Description:

Is there a cat in your dat?

A common task in machine learning pipelines is encoding categorical variables for a given algorithm in a format that allows as much useful signal as possible to be captured.

Because this is such a common task and important skill to master, we've put together a dataset that contains only categorical features, and includes:

binary features
low- and high-cardinality nominal features
low- and high-cardinality ordinal features
(potentially) cyclical features
This Playground competition will give you the opportunity to try different encoding schemes for different algorithms to compare how they perform. We encourage you to share what you find with the community.

I used embedding layers for all of the features.  My results at the time of submission had me in to top 50% of submissions. The evaluation was done using area under the ROC curve.  My score on the test set was 0.801 compared with the leader 0.808.