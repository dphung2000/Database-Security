from flask import Flask, jsonify, request
import tensorflow as tf
import tensorflow_federated as tff
import sys
import json
import requests
import random
app = Flask(__name__)

def model_fn():
    """Create the TFF model"""
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(3,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

def tff_model_fn():
    model = model_fn()  # Get the Keras model
    
    # Define the input_spec as a dictionary for features and a tensor for labels
    input_spec = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )
    
    # Use from_keras_model to wrap the Keras model with the correct input_spec
    return tff.learning.models.from_keras_model(
        keras_model=model,
        input_spec=input_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# Helper function to serialize weights to send to clients
def serialize_weights(weights):
    return json.dumps([w.tolist() for w in weights])

# Helper function to deserialize weights received from clients
def deserialize_weights(weights_json):
    return [tf.convert_to_tensor(w) for w in json.loads(weights_json)]

@app.route('/start', methods=['GET'])
def start_training():
    tff.framework.set_default_context(tff.backends.native.create_sync_local_cpp_execution_context())
    print("Begin...", file=sys.stdout)
    # List of client URLs
    clients = ['http://fl-client-1:5000', 'http://fl-client-2:5000', 'http://fl-client-1:5000']  # Example clients

    # Set the number of rounds
    total_rounds = 3

    # Federated averaging process definition
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=tff_model_fn,
        client_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=0.001)
    )

    state = iterative_process.initialize()

    for round_num in range(total_rounds):
        print(f"Starting Round {round_num + 1}", file=sys.stdout)
        
        # Select a random subset of clients for this round (simulating client availability)
        participating_clients = random.sample(clients, k=random.randint(1, len(clients)))

        print(f"Clients involved in this round: {participating_clients}", file=sys.stdout)
        
        # Number of epochs for each client training session (adjust as needed)
        epochs = random.randint(2, 5)

        model_weights = state.global_model_weights.trainable
        weights_json = serialize_weights(model_weights)

        # Send the model to the selected clients and get updated weights
        client_weights  = []
        for client_url in participating_clients:
            response = requests.post(f'{client_url}/train', json={
                'epochs': epochs,
                'weights': weights_json
            })
            updated_weights = deserialize_weights(response.json()['updated_weights'])
            client_weights.append(updated_weights)

        # Aggregate updated weights (simple averaging)
        def process_client_weights(client_weights_list):
            """Convert client weights into the format expected by TFF"""
            # Create a list of weight measurements
            measurements = []
            for weights in client_weights_list:
                measurement = tff.learning.models.ModelWeights(
                    trainable=tuple(tf.convert_to_tensor(w) for w in weights),
                    non_trainable=()
                )
                measurements.append(measurement)
            return measurements
        with open("log.txt", "w") as file:
            file.write(f"client_weights: {client_weights}")
        state, metrics = iterative_process.next(state, process_client_weights(client_weights))
        
        print(f"Finished Round {round_num + 1}, Metrics: {metrics}", file=sys.stdout)
    
    return jsonify({
            "status": "success",
            "message": "Training completed",
            "rounds_completed": total_rounds,
        })

if __name__ == '__main__':
    tff.framework.set_default_context(tff.backends.native.create_sync_local_cpp_execution_context())

    app.run(host="0.0.0.0", port=5000)