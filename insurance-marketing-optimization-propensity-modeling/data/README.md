# Data Description

The dataset contains information about customers and their interactions with a marketing campaign. Below are the features included in the dataset along with their descriptions:

- `custAge`: The age of the customer (in years).
- `profession`: Type of job.
- `marital`: Marital status.
- `schooling`: Education level.
- `default`: Has a previous defaulted account?
- `housing`: Has a housing loan?
- `loan`: Has a personal loan?
- `contact`: Preferred contact type.
- `month`: Last contact month.
- `day_of_week`: Last contact day of the week.
- `campaign`: Number of times the customer was contacted.
- `pdays`: Number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted).
- `previous`: Number of contacts performed before this campaign and for this client.
- `poutcome`: Outcome of the previous marketing campaign.
- `emp.var.rate`: Employment variation rate - quarterly indicator.
- `cons.price.idx`: Consumer price index - monthly indicator.
- `cons.conf.idx`: Consumer confidence index - monthly indicator.
- `euribor3m`: Euribor 3 month rate - daily indicator.
- `nr.employed`: Number of employees - quarterly indicator.
- `pmonths`: Number of months that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted).
- `pastEmail`: Number of previous emails sent to this client.
- `responded`: Did the customer respond to the marketing campaign and purchase a policy? (Target Variable)

The data is split into raw and processed directories. The raw directory contains the following files:
- `test.xlsx`: Test dataset.
- `train.xlsx`: Train dataset.

The processed directory contains the following files:
- `testingCandidate.csv`: Preprocessed test dataset.
- `train_preprocessed.xlsx`: Preprocessed train dataset.
