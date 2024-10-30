# db_setup.py
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, select
from sqlalchemy.orm import declarative_base, sessionmaker
import numpy as np

Base = declarative_base()

class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True)
    amount = Column(Float)
    time_hour = Column(Integer)  # Hour of day
    distance_from_home = Column(Float)
    is_fraud = Column(Boolean)

def create_synthetic_data(n_samples, fraud_ratio=0.2):
    """Create synthetic credit card transaction data"""
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Normal transactions
    normal_data = {
        'amount': np.random.normal(100, 50, n_normal),
        'time_hour': np.random.randint(0, 24, n_normal),
        'distance_from_home': np.random.normal(5, 3, n_normal),
        'is_fraud': [False] * n_normal
    }
    
    # Fraudulent transactions
    fraud_data = {
        'amount': np.random.normal(750, 200, n_fraud),
        'time_hour': np.random.randint(0, 24, n_fraud),
        'distance_from_home': np.random.normal(50, 20, n_fraud),
        'is_fraud': [True] * n_fraud
    }
    
    # Combine and shuffle
    for key in normal_data:
        normal_data[key] = np.concatenate([normal_data[key], fraud_data[key]])
    np.random.shuffle(normal_data['amount'])
    
    return normal_data

def setup_client_database(n_samples=1000):
    """Setup database for a client with synthetic data"""
    import os
    if os.path.exists("data/client_data.db"):
        print("client_data.db already exists. Remove")
        os.remove("data/client_data.db")
        print("Removed")
    
    engine = create_engine(f'sqlite:///data/client_data.db')
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Generate and insert synthetic data
    data = create_synthetic_data(n_samples)
    
    for i in range(n_samples):
        transaction = Transaction(
            amount=data['amount'][i],
            time_hour=data['time_hour'][i],
            distance_from_home=data['distance_from_home'][i],
            is_fraud=data['is_fraud'][i]
        )
        session.add(transaction)
    
    session.commit()
    session.close()
    print("Created database at /app/data/client_data.db")
    return engine
if __name__ == "__main__":
    setup_client_database(10000)