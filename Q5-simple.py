# Load the dataset
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Example dataset: Transactions with items bought by different customers
data = [['Milk', 'Bread', 'Butter'],
        ['Milk', 'Diaper', 'Beer', 'Eggs'],
        ['Milk', 'Diaper', 'Beer', 'Cola'],
        ['Bread', 'Butter', 'Eggs'],
        ['Milk', 'Diaper', 'Eggs'],
        ['Bread', 'Butter', 'Cola']]

# Convert dataset to DataFrame (one-hot encoding)
df = pd.DataFrame([{item: 1 for item in transaction} for transaction in data]).fillna(0)

# Generate frequent itemsets using Apriori
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display results
print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
