# Decision log

DSC-0001: Use Pydantic
Date: 07-11-2024
Decision: use Pydantic 
Motivation: it was used in code we need to incorporate. Keep using for uniformity.
Alternative: no Pydantic 

DSC-0002: Metrics used
Date: 08-11-2024
Decision: use mean absolute error (MAE), R2 score, precision and recall as additional metrics 
Motivation: They are common metrics used in regression and classification and more simple to implement
Alternative: Could have used adjusted R2, mean squared log error (MSLE), ect.

DSC-0003: Models used
Date: 08-11-2024
Decision: For regression we used multiple linear regression, ridge regression and random forest regression. For classification we used logistic regression,  super-vector-machine (SVM) classifier, random forest classifier.
Motivation: They are common models used for regression and classification.
Alternative: Could have used polynomial regression or decision tree classification.

DSC-0004: ID name
Date: 09-11-2024
Decision: instead of id = {encoded_path}:{self.version} we will use id = {encoded_path}_{self.version}
Motivation: Using : in filename caused a JSON Decode Error upon saving a dataset in StreamLit on Windows. 
Alternative: use just {encoded_path} 

DCS-0005: csv datasets
Date: 10-11-2024
Decision: add the option to select a csv dataset from a repo.
Motivation: even though the different structures of new datasets lead to new errors that we could not fix before the deadline, we decided to leave the option to select them. For some datasets, detecting features is not possible, and for others everything except training the dataset works. Default dataset Iris supports full functionality.
Alternative: can leave out the option of selecting a dataset from “dataset repository”, leaving just upload by URL and by local file system.


