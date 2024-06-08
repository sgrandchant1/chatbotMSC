import pandas as pd

csv_file_path = '/Users/santiagodegrandchant/Desktop/msc_banco/netflix_userbase.csv'
df = pd.read_csv(csv_file_path)

# The condition should use backticks for column names with spaces and single quotes for string values
# condition = "`User ID` == 1"
# condition = "`Subscription Type` == 'Basic'"
condition = "`Age` == '23'"
result = df.query(condition)

print(result)
