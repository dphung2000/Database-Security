import tensorflow as tf
from flask import Flask, request, jsonify
import json
import sqlite3
import struct
import sys
# import requests

app = Flask(__name__)

# Function to load data from the SQLite database
def load_data():
    conn = sqlite3.connect('data/client_data.db')  # Connect to the existing database
    cursor = conn.cursor()

    # Fetch all data from the table
    cursor.execute("SELECT amount, time_hour, distance_from_home, is_fraud FROM transactions")
    data = cursor.fetchall()

    conn.close()

    # Split data into training and test sets (e.g., 80/20 split)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    def process_row(row):
        return (
            float(row[0]),
            struct.unpack('q', row[1])[0] / 24.0,
            float(row[2])
        )

    # Prepare training data
    x_train = [process_row(row) for row in train_data]
    y_train = [row[3] for row in train_data]

    # Prepare test data
    x_test = [process_row(row) for row in test_data]
    y_test = [row[3] for row in test_data]

    return (
        (tf.constant(x_train, dtype=tf.float32), tf.constant(y_train, dtype=tf.float32)),
        (tf.constant(x_test, dtype=tf.float32), tf.constant(y_test, dtype=tf.float32))
    )

# Helper function to serialize weights to send to server
def serialize_weights(weights):
    return json.dumps([w.tolist() for w in weights])

# Helper function to deserialize weights received from the server
def deserialize_weights(weights_json):
    return [tf.convert_to_tensor(w) for w in json.loads(weights_json)]

@app.route('/train', methods=['POST'])
def train():
    # Get the model weights and number of epochs from the server
    request_data = request.get_json()
    model_weights = deserialize_weights(request_data['weights'])
    epochs = request_data['epochs']

    # print(f"Initial weights:{model_weights}", file=sys.stderr)
    # Create the model and set its weights
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(3,)),  # Ensure input is not a TensorSpec but an actual shape
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
    model.set_weights(model_weights)
    
    # Load client data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Compile and train the model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs)
    
    # Evaluate the model after training
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Client Evaluation on Test Data - Loss: {loss}, Accuracy: {accuracy}", file=sys.stdout)
    
    # Return updated model weights to the server
    updated_weights = model.get_weights()
    updated_weights_json = serialize_weights(updated_weights)
    
    # print(f"updated weights: {updated_weights_json}", file=sys.stderr)
    return jsonify({'updated_weights': updated_weights_json})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)