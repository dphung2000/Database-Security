# attacker.py
import requests
import numpy as np
import tensorflow as tf
from client import FLClient

class ModelPoisoningAttacker(FLClient):
    def train_local_model(self, model_weights):
        """Override training with poisoned data"""
        X, y = self.get_local_data()
        # Flip all labels
        y = 1 - y
        
        dataset = self.create_tf_dataset(X, y)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        model.set_weights(model_weights)
        model.fit(dataset, epochs=1, verbose=0)
        
        # Multiply weights by large factor to increase impact
        poisoned_weights = [w * 10 for w in model.get_weights()]
        return poisoned_weights

class MITMAttacker:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
    
    def intercept_and_modify(self, client_id, weights):
        """Intercept and modify weights before sending to server"""
        # Modify weights to cause model degradation
        modified_weights = [w * -1 for w in weights]
        
        # Send modified weights to server
        serialized_weights = FLClient.serialize_weights(modified_weights)
        requests.post(
            f"{self.server_url}/submit_update",
            json={
                'client_id': client_id,
                'weights': serialized_weights
            }
        )

def test_attacks():
    # Test model poisoning
    attacker = ModelPoisoningAttacker(client_id="malicious")
    attacker.participate_in_training()
    
    # Test MITM attack
    mitm = MITMAttacker()
    legitimate_client = FLClient(client_id=0)
    weights = legitimate_client.get_local_data()
    mitm.intercept_and_modify(0, weights)