import tensorflow as tf
from tensorflow.keras import layers


#################### Modèles ####################
class Encoder(layers.Layer):
    def __init__(self, latent_size=8, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        
        # Couches
        self.conv1 = layers.Conv2D(filters=4, kernel_size=3, strides=1, padding="SAME", 
                                   use_bias=True, data_format='channels_last')
        self.conv2 = layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="SAME", 
                                   use_bias=True, data_format='channels_last')
        
        # Activations
        self.relu = layers.ReLU()
        
        # Couche latente
        self.dense = layers.Dense(units=128, use_bias=True)
        self.latent = layers.Dense(units=latent_size, use_bias=True)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dense(x)
        x = self.relu(x)
        return self.latent(x)


# Notre bloc décodeur
class Decoder(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)
        
        # Couches
        self.de_conv1 = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding="SAME", 
                                               use_bias=True, data_format='channels_last')
        self.de_conv2 = layers.Conv2DTranspose(filters=4, kernel_size=3, strides=1, padding="SAME", 
                                               use_bias=True, data_format='channels_last')
        
        # Activations
        self.relu = layers.ReLU()
        
        # Couche latente
        self.dense = layers.Dense(units=784, use_bias=True)
        self.de_latent = layers.Dense(units=128, use_bias=True)
    
    def call(self, inputs):
        x = self.de_latent(inputs)
        x = self.relu(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.de_conv2(x)
        x = self.relu(x)
        return self.de_conv1(x)


# Classe finale : combinaison encodeur - decodeur
class ConvolutionalAE(tf.keras.Model):
    def __init__(self, latent_size=8, *args, **kwargs):
        super(ConvolutionalAE, self).__init__(*args, **kwargs)
        self.encoder = Encoder(latent_size=latent_size)
        self.decoder = Decoder()

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)


#################### Fonctions ####################
def run_step(model, loss_object, optimizer, loss_operation, inputs, labels=None):
    """
    Fonction qui va rouler une étape, soit le passage d'une batch d'exemples.
    De plus, le graph va se construire durant l'exécution à l'aide de tf.GradientTape().
    Les labels peuvent être de type 'None', auquel cas, la sortie sera comparée avec l'entrée.
    """
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        if labels is None:
            loss = loss_object(inputs, outputs)
        else:
            loss = loss_object(labels, outputs)
    # Le gradient se calcule en dehors du contexte du graph
    gradients = tape.gradient(loss, model.trainable_variables)
    # La backpropagation
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Le suivi de la loss
    return loss_operation(loss)


def train(model, loss_object, optimizer, train_data, loss_operations, epochs=100, use_labels=False):
    """
    Fonction d'entrainement, à chacune des époques, on fait passer l'ensemble du jeu de données.
    """
    epoch_op_loss, step_op_loss = loss_operations
    for ep in range(epochs):
        for (batch, (inputs, labels)) in enumerate(train_data):
            batches_loss = []
            if use_labels:
                loss = run_step(model, loss_object, optimizer, step_op_loss, inputs, labels)
            else:
                loss = run_step(model, loss_object, optimizer, step_op_loss, inputs)
            batches_loss.append(loss)
            print(f'\rBatch : {batch + 1} / {len(train_data)} - Loss : {loss:0.5f}', end='')
        epoch_loss = epoch_op_loss(batches_loss)
        print(f'\rEpoch : {ep+1} / {epochs} - Loss : {epoch_loss:.5f}')