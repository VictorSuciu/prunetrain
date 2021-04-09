# Flattened CNN models

We flatten CNN architectures (each layer module definition) to support easy network architecture reconfiguration and generation into python files. After pruning the CNN model using group lasso regularization, each layer has different channel dimensions. To store layers each with different channel dimensions, it is rather convenient to flatten the network layer structure than building with nested loops.
