# EXPERIMENT 6 : MARKET BASKET ANALYSIS (ASSOCIATION RULE MINING USING APIRIORI ALGORITHM)

## AIM

To perform Market Basket Analysis using the Apriori algorithm to identify frequent itemsets and generate association rules from transaction data for decision-making in retail and business analytics.

## DATASET

A sample retail transaction dataset containing **10 transactions** across **4 grocery items**: `Bread`, `Butter`, `Milk`, and `Eggs`.

| Transaction ID | Items Purchased |
| :---: | :--- |
| **T1** | Milk, Bread, Eggs |
| **T2** | Milk, Bread |
| **T3** | Milk, Eggs |
| **T4** | Bread, Butter |
| **T5** | Milk, Bread, Butter, Eggs |
| **T6** | Bread, Eggs |
| **T7** | Milk, Butter |
| **T8** | Milk, Bread, Butter |
| **T9** | Bread, Butter, Eggs |
| **T10** | Milk, Bread, Butter, Eggs |

*The transaction list is transformed into a One-Hot Encoded binary DataFrame prior to mining.*

## ALGORITHM / STEPS FOLLOWED

1. Import required Python libraries (`pandas`, `mlxtend`).
2. Create the sample grocery transaction dataset.
3. Convert the dataset into a one-hot encoded binary matrix using `TransactionEncoder`.
4. Apply the `apriori()` algorithm (`min_support=0.3`) to identify frequent itemsets.
5. Generate association rules from frequent itemsets using `association_rules()` with `lift >= 1.0`.
6. Filter strong rules based on thresholds (`confidence >= 0.6` and `lift > 1`).
7. Interpret the rules to uncover cross-selling and store layout insights.

## FUNCTIONS AND METHODS USED

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Handling** | `TransactionEncoder()` | Converts a list of transactions into a one-hot encoded binary matrix |
| **Frequent Pattern Mining** | `apriori()` | Finds frequent itemsets meeting the minimum support threshold |
| **Association Rule Generation** | `association_rules()` | Derives association rules using metrics such as support, confidence, and lift |
| **Filtering** | Conditional filtering (`rules[...]`) | Selects rules meeting strong confidence and lift thresholds |
| **Visualization / Display** | `print()` / `DataFrame.head()` | Displays encoded transactions, itemsets, and rules in tabular format |

## PROGRAM & OUTPUT

import pandas as pd  
from mlxtend.preprocessing import TransactionEncoder  
from mlxtend.frequent_patterns import apriori, association_rules  
transactions = [  
    ['Milk', 'Bread', 'Eggs'],  
    ['Milk', 'Bread'],  
    ['Milk', 'Eggs'],  
    ['Bread', 'Butter'],  
    ['Milk', 'Bread', 'Butter', 'Eggs'],  
    ['Bread', 'Eggs'],  
    ['Milk', 'Butter'],  
    ['Milk', 'Bread', 'Butter'],  
    ['Bread', 'Butter', 'Eggs'],  
    ['Milk', 'Bread', 'Butter', 'Eggs']  
]  
te = TransactionEncoder()  
te_ary = te.fit(transactions).transform(transactions)  
df = pd.DataFrame(te_ary, columns=te.columns_)    
print("--- 1. One-Hot Encoded Binary Matrix ---")  
print(df)  
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)  
print("\n--- 2. Frequent Itemsets (Support >= 0.3) ---")  
print(frequent_itemsets)  
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)    
print("\n--- 3. Generated Association Rules ---")    
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])  
strong_rules = rules[(rules['confidence'] >= 0.6) & (rules['lift'] > 1.0)]  
print("\n--- 4. Filtered Strong Association Rules ---")  
print(strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])  

<img width="532" height="732" alt="image" src="https://github.com/user-attachments/assets/f5915f27-5896-49dd-8b01-3385349d8911" />
<img width="473" height="117" alt="image" src="https://github.com/user-attachments/assets/2ec1fcea-cf1b-492e-a22a-50e01fc3c7df" />



## RESULT

Thus to perform Market Basket Analysis using the Apriori algorithm to identify frequent itemsets and generate association rules from transaction data for decision-making in retail and business analytics successfully.
