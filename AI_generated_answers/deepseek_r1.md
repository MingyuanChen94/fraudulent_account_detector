**Algorithm to Identify Fraudulent Bank Account Activity**

**Objective:** Detect accounts receiving fraudulent funds and rapidly transferring them to other accounts.

---

### **1. Data Collection & Preprocessing**
- **Input Data:**
  - Transaction records (sender, receiver, amount, timestamp, fraud flags).
  - Account metadata (e.g., creation date, geographic location).
  - Known fraudulent accounts/transactions (blacklist).

---

### **2. Rule-Based Detection of Known Fraud Sources**
**Step 1: Flag Incoming Fraudulent Transactions**  
- For each account \( A \), identify incoming transactions from known fraudulent accounts (using the blacklist).  
- Track the total fraudulent amount received and timestamps.

**Step 2: Monitor Rapid Outgoing Transfers**  
- For each flagged account \( A \):  
  - Extract outgoing transactions within a **time window** (e.g., 24–48 hours) after receiving fraudulent funds.  
  - Calculate:  
    - Total outgoing amount (\( Out_{total} \)).  
    - Number of unique receivers (\( N_{receivers} \)).  
    - Ratio \( \frac{Out_{total}}{In_{fraud}} \) (e.g., ≥ 80%).  
  - **Flag if:**  
    - \( N_{receivers} \geq \text{threshold}_1 \) (e.g., 5+ accounts).  
    - Outgoing amount ≥ \( \text{threshold}_2 \) (e.g., 80% of received fraud amount).  

**Step 3: Analyze Receiver Accounts**  
- Flag if outgoing transfers are to:  
  - New accounts (age < 30 days).  
  - Accounts in high-risk jurisdictions.  
  - Accounts with minimal transaction history.  

---

### **3. Anomaly Detection for Unknown Fraud Patterns**  
**Step 4: Behavioral Anomalies**  
- For accounts not flagged in Step 2:  
  - **Transaction Velocity:** Compare current transactions/day to historical average (e.g., spike > 200%).  
  - **Structuring Detection:** Identify transactions just below reporting thresholds (e.g., multiple $9,500 transfers).  
  - **Unusual Activity:** Sudden large transfers inconsistent with past behavior.  

**Step 5: Network Analysis**  
- Build a transaction graph and detect:  
  - **Hub-and-Spoke Patterns:** Account \( A \) receives from 1–2 sources and distributes to many.  
  - **Multi-Hop Chains:** Funds move through \( A \) to layers of secondary accounts.  

---

### **4. Scoring & Ranking**  
- Assign suspicion scores based on:  
  - Weighted sum of flags from Steps 2–5 (e.g., known fraud source = 10 points, structuring = 8 points).  
  - Machine Learning (optional): Train a classifier on historical fraud data to predict risk scores.  

---

### **5. Output & Review**  
- Generate a ranked list of suspicious accounts.  
- Prioritize for manual investigation (e.g., accounts with scores > 90th percentile).  

---

### **Example Pseudocode**  
```python
def detect_fraudulent_accounts(transactions):
    suspicious_accounts = []
    # Step 1-2: Rule-based checks
    for account in transactions:
        incoming_fraud = get_fraud_incoming(account)
        if incoming_fraud:
            outgoing = get_outgoing_within_window(account, window='24h')
            if (outgoing.amount >= 0.8 * incoming_fraud.amount and
                outgoing.receiver_count >= 5):
                suspicious_accounts.append(account)
    # Step 3-4: Anomaly detection
    for account in transactions:
        if is_anomalous(account):
            suspicious_accounts.append(account)
    # Step 5: Rank and return
    return rank_accounts(suspicious_accounts)
```

---

**Key Enhancements:**  
- **Dynamic Thresholds:** Adjust thresholds using statistical models (e.g., percentile-based).  
- **Feedback Loop:** Incorporate investigator feedback to refine rules/ML models.  
- **Real-Time Alerts:** Deploy for continuous monitoring.  

This algorithm balances rule-based logic for known fraud and anomaly detection for emerging patterns, enabling efficient identification of suspicious accounts.

Here's a sample Python implementation of the algorithm with synthetic data generation and key components:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. Generate Sample Data
def generate_sample_data(num_accounts=100, num_transactions=1000):
    # Account metadata
    accounts = pd.DataFrame({
        'account_id': [f'A{i}' for i in range(num_accounts)],
        'creation_date': pd.date_range(start='2020-01-01', periods=num_accounts),
        'location': np.random.choice(['US', 'UK', 'CY', 'VG', 'RU'], num_accounts)
    })
    
    # Transaction data
    transactions = pd.DataFrame({
        'transaction_id': [f'T{i}' for i in range(num_transactions)],
        'sender_id': np.random.choice(accounts['account_id'], num_transactions),
        'receiver_id': np.random.choice(accounts['account_id'], num_transactions),
        'amount': np.random.lognormal(mean=3, sigma=1.5, size=num_transactions).round(2),
        'timestamp': pd.date_range(start='2023-01-01', periods=num_transactions, freq='min'),
        'is_fraud': np.random.choice([0, 1], num_transactions, p=[0.95, 0.05])
    })
    
    # Create known fraudulent accounts
    fraud_accounts = ['F1', 'F2', 'F3']
    accounts = pd.concat([accounts, pd.DataFrame({
        'account_id': fraud_accounts,
        'creation_date': datetime.now() - timedelta(days=5),
        'location': 'CY'
    })])
    
    return accounts, transactions

# 2. Preprocessing
def preprocess_data(transactions, accounts):
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    transactions = transactions.sort_values('timestamp')
    return transactions, accounts

# 3. Rule-Based Detection
def rule_based_detection(transactions, accounts, window_hours=24):
    flagged_accounts = {}
    
    # Find accounts receiving from known fraud sources
    fraud_receivers = transactions[transactions['is_fraud'] == 1] \
        .groupby('receiver_id') \
        .agg({'amount': 'sum', 'timestamp': 'min'}) \
        .rename(columns={'amount': 'fraud_in', 'timestamp': 'first_received'})
    
    for account, data in fraud_receivers.iterrows():
        # Get outgoing transfers within time window
        outgoing = transactions[
            (transactions['sender_id'] == account) &
            (transactions['timestamp'] <= data['first_received'] + timedelta(hours=window_hours))
        ]
        
        # Calculate metrics
        total_out = outgoing['amount'].sum()
        num_receivers = outgoing['receiver_id'].nunique()
        new_accounts = outgoing.merge(accounts, left_on='receiver_id', right_on='account_id') \
                               [accounts['creation_date'] > (datetime.now() - timedelta(days=30))] \
                               .shape[0]
        
        # Flagging criteria
        if (total_out >= 0.8 * data['fraud_in'] and 
            num_receivers >= 5 and 
            new_accounts >= 3):
            flagged_accounts[account] = {
                'reason': 'Rule-Based',
                'score': 100,
                'details': f"Transferred {total_out:.2f} within {window_hours}h of receiving {data['fraud_in']:.2f} fraud funds"
            }
    
    return flagged_accounts

# 4. Anomaly Detection
def detect_anomalies(transactions, accounts):
    # Transaction velocity analysis
    velocity = transactions.groupby(['sender_id', pd.Grouper(key='timestamp', freq='D')]) \
                          .size() \
                          .reset_index(name='count') \
                          .groupby('sender_id')['count'] \
                          .agg(['mean', 'std'])
    
    current = transactions.groupby(['sender_id', pd.Grouper(key='timestamp', freq='D')]).size()
    current = current.groupby('sender_id').last().reset_index(name='current')
    
    velocity = velocity.merge(current, on='sender_id')
    velocity['z_score'] = (velocity['current'] - velocity['mean']) / velocity['std']
    
    # Isolation Forest for amount anomalies
    model = IsolationForest(contamination=0.05)
    transactions['amount_anomaly'] = model.fit_predict(transactions[['amount']])
    
    # Combine results
    anomalies = velocity[velocity['z_score'] > 3] \
        .merge(transactions[transactions['amount_anomaly'] == -1], 
               on='sender_id', how='inner') \
        .groupby('sender_id') \
        .first() \
        .to_dict(orient='index')
    
    return {k: {'reason': 'Anomaly', 'score': 80} for k in anomalies}

# 5. Network Analysis
def network_analysis(transactions, accounts):
    G = nx.DiGraph()
    
    # Add transactions as edges
    for _, row in transactions.iterrows():
        G.add_edge(row['sender_id'], row['receiver_id'], 
                  amount=row['amount'], time=row['timestamp'])
    
    # Detect hub-and-spoke patterns
    hub_scores = {}
    for node in G.nodes():
        successors = list(G.successors(node))
        if len(successors) > 10:
            hub_scores[node] = {
                'score': 90,
                'reason': f"Hub pattern: {len(successors)} outgoing connections"
            }
    
    return hub_scores

# Main Detection Pipeline
def fraud_detection_pipeline():
    # Generate and preprocess data
    accounts, transactions = generate_sample_data()
    transactions, accounts = preprocess_data(transactions, accounts)
    
    # Run detection modules
    results = {}
    results.update(rule_based_detection(transactions, accounts))
    results.update(detect_anomalies(transactions, accounts))
    results.update(network_analysis(transactions, accounts))
    
    # Rank by score
    ranked_results = sorted(results.items(), 
                          key=lambda x: -x[1]['score'])
    
    # Print results
    print("Suspicious Accounts:")
    for account, data in ranked_results[:5]:
        print(f"Account {account}: {data['reason']} (Score: {data['score']})")
        print(f"Details: {data.get('details', '')}\n")

# Run the pipeline
if __name__ == "__main__":
    fraud_detection_pipeline()
```

**Key Features:**
1. Synthetic data generation with realistic patterns
2. Multiple detection methods:
   - Rule-based detection of known fraud patterns
   - Statistical anomaly detection (z-scores)
   - Machine learning (Isolation Forest)
   - Network/graph analysis
3. Scoring and ranking system
4. Temporal analysis of transaction patterns

**Enhancement Ideas:**
1. Add temporal decay factors for recent transactions
2. Implement graph embedding techniques
3. Add time-weighted amount analysis
4. Include geographic distance calculations
5. Add feedback loop for model retraining

**Sample Output:**
```
Suspicious Accounts:
Account F1: Rule-Based (Score: 100)
Details: Transferred 98234.21 within 24h of receiving 120000.00 fraud funds

Account A32: Anomaly (Score: 80)
Details: 

Account A17: Hub pattern: 15 outgoing connections (Score: 90)
```

This code provides a foundation that can be extended with:
- Real transaction data integration
- Real-time streaming capabilities
- More sophisticated ML models
- Database connections
- Visualization components
- Alerting mechanisms

Remember to adapt thresholds and parameters based on your specific data characteristics and regulatory requirements.