from flask import Flask, jsonify, request
import tensorflow as tf
import sys
import json
import requests
import random
app = Flask(__name__)

global_model =  tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(3,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
global_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])
# Helper function to serialize weights to send to clients
def serialize_weights(weights):
    return json.dumps([w.tolist() for w in weights])

# Helper function to deserialize weights received from clients
def deserialize_weights(weights_json):
    return [tf.convert_to_tensor(w) for w in json.loads(weights_json)]

@app.route('/start', methods=['GET'])
def start_training():
    print("Begin...", file=sys.stderr)
    # List of client URLs
    clients = ['http://fl-client-1:5000', 'http://fl-client-2:5000', 'http://fl-client-3:5000']  # Example clients

    # Set the number of rounds
    total_rounds = 10
    for round_num in range(total_rounds):
        print(f"Starting Round {round_num + 1}", file=sys.stderr)
        
        # Select a random subset of clients for this round (simulating client availability)
        participating_clients = random.sample(clients, k=random.randint(1, len(clients)))

        print(f"Clients involved in this round: {participating_clients}", file=sys.stderr)
        
        # Number of epochs for each client training session (adjust as needed)
        epochs = random.randint(2, 3)

        global_weights  = global_model.get_weights()
        weights_json = serialize_weights(global_weights)

        # print(f"weights_json: {weights_json}", file=sys.stderr)
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

        averaged_weights = []
        for layer_weights in zip(*client_weights):
            layer_avg = sum(layer_weights) / len(layer_weights)
            averaged_weights.append(layer_avg)

        # Update the global model with the aggregated weights
        global_model.set_weights(averaged_weights)
        print(f"averaged_weights: {averaged_weights}\n", file=sys.stderr)
        print(f"Finished Round {round_num + 1}\n", file=sys.stderr)
   
    return jsonify({
            "status": "success",
            "message": "Training completed",
            "rounds_completed": total_rounds,
        })

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=5000)