#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os

# CIFAR-10 class names for custom image labeling
CIFAR10_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


def build_model1():
    """Fully-connected: Flatten, 3x Dense(128, leaky_relu), Dense(10). Output is logits."""
    model = Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(128, activation=layers.LeakyReLU(negative_slope=0.3)),
        layers.Dense(128, activation=layers.LeakyReLU(negative_slope=0.3)),
        layers.Dense(128, activation=layers.LeakyReLU(negative_slope=0.3)),
        layers.Dense(10),
    ], name='model1')
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def build_model2():
    """CNN: Conv2D 32 stride2 -> BN -> Conv2D 64 stride2 -> BN -> 4x (Conv2D 128 -> BN) -> Flatten -> Dense(10)."""
    model = Sequential([
        layers.Conv2D(32, 3, strides=2, padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(10),
    ], name='model2')
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def build_model3():
    """Depthwise-separable CNN: 6x (SeparableConv2D -> BN), Flatten, Dense(10). First layer is also SeparableConv2D."""
    model = Sequential([
        layers.SeparableConv2D(32, 3, strides=2, padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.SeparableConv2D(64, 3, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(10),
    ], name='model3')
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def build_model50k():
    """Best model with no more than 50,000 parameters. Save after training as best_model.h5."""
    model = Sequential([
        layers.Conv2D(32, 3, strides=2, padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.SeparableConv2D(64, 3, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, 3, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10),
    ], name='best_model')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

    ########################################
    ## Load the CIFAR10 data set
    ########################################
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    train_labels = np.squeeze(y_train)
    test_labels = np.squeeze(y_test)
    train_images = x_train.astype(np.float32) / 255.0
    test_images = x_test.astype(np.float32) / 255.0
    # 20% Validation split 
    n_val = int(0.2 * len(train_images))
    val_images = train_images[-n_val:]
    val_labels = train_labels[-n_val:]
    train_images = train_images[:-n_val]
    train_labels = train_labels[:-n_val]

    ########################################
    ## Build and train model 1
    ########################################
    model1 = build_model1()
    model1.summary()
    hist1 = model1.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=30,
        batch_size=128,
        verbose=1
    )
    print('Model 1 — Train:', model1.evaluate(train_images, train_labels, verbose=0)[1],
          'Val:', model1.evaluate(val_images, val_labels, verbose=0)[1],
          'Test:', model1.evaluate(test_images, test_labels, verbose=0)[1])

    ## Build, compile, and train model 2 (CNN)
    model2 = build_model2()
    model2.summary()
    hist2 = model2.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=30,
        batch_size=128,
        verbose=1
    )
    print('Model 2 — Train:', model2.evaluate(train_images, train_labels, verbose=0)[1],
          'Val:', model2.evaluate(val_images, val_labels, verbose=0)[1],
          'Test:', model2.evaluate(test_images, test_labels, verbose=0)[1])

    # custom image
    for name in CIFAR10_CLASS_NAMES:
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(os.path.dirname(__file__) or '.', f'test_image_{name}{ext}')
            if os.path.isfile(p):
                test_img = np.array(keras.utils.load_img(p, grayscale=False, color_mode='rgb', target_size=(32, 32)), dtype=np.float32) / 255.0
                test_img = np.expand_dims(test_img, axis=0)
                pred_idx = int(np.argmax(model2.predict(test_img, verbose=0)[0]))
                print(f'Custom image: predicted "{CIFAR10_CLASS_NAMES[pred_idx]}" (expected "{name}") — correct: {pred_idx == CIFAR10_CLASS_NAMES.index(name)}')
                break
        else:
            continue
        break

    ### Repeat for model 3
    model3 = build_model3()
    model3.summary()
    hist3 = model3.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=30,
        batch_size=128,
        verbose=1
    )
    print('Model 3 — Train:', model3.evaluate(train_images, train_labels, verbose=0)[1],
          'Val:', model3.evaluate(val_images, val_labels, verbose=0)[1],
          'Test:', model3.evaluate(test_images, test_labels, verbose=0)[1])

    ### best sub-50k params model
    model50k = build_model50k()
    model50k.summary()
    assert model50k.count_params() <= 50000, 'Model must have ≤50k params'
    hist50k = model50k.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=40,
        batch_size=128,
        verbose=1
    )
    print('Model50k — Val:', model50k.evaluate(val_images, val_labels, verbose=0)[1],
          'Test:', model50k.evaluate(test_images, test_labels, verbose=0)[1])
    model50k.save('best_model.h5')
    print('Saved best_model.h5')

    ## plots 
    def plot_history(hist, title, ax):
        ax.plot(hist.history['accuracy'], label='Train acc')
        ax.plot(hist.history['val_accuracy'], label='Val acc')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_history(hist1, 'Model 1 (fully-connected)', axes[0, 0])
    plot_history(hist2, 'Model 2 (CNN)', axes[0, 1])
    plot_history(hist3, 'Model 3 (separable CNN)', axes[1, 0])
    plot_history(hist50k, 'Model 50k (best)', axes[1, 1])
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved training_curves.png')

    # sample CIFAR-10 images, one per class
    n_show = 10
    fig2, axes2 = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(n_show):
        idx = np.where(train_labels == i)[0][0]
        ax = axes2[i // 5, i % 5]
        ax.imshow(train_images[idx])
        ax.set_title(CIFAR10_CLASS_NAMES[i], fontsize=9)
        ax.axis('off')
    plt.suptitle('CIFAR-10 sample (one per class)', fontsize=11)
    plt.tight_layout()
    plt.savefig('cifar10_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved cifar10_samples.png')

    # single comparison plot
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    ax3.plot(hist1.history['val_accuracy'], label='Model 1 (FC)', color='C0')
    ax3.plot(hist2.history['val_accuracy'], label='Model 2 (CNN)', color='C1')
    ax3.plot(hist3.history['val_accuracy'], label='Model 3 (Sep CNN)', color='C2')
    ax3.plot(hist50k.history['val_accuracy'], label='Model 50k (best)', color='C3')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation accuracy')
    ax3.set_title('Validation accuracy comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved model_comparison.png')

    # summary table
    train_a1, test_a1 = model1.evaluate(train_images, train_labels, verbose=0)[1], model1.evaluate(test_images, test_labels, verbose=0)[1]
    train_a2, test_a2 = model2.evaluate(train_images, train_labels, verbose=0)[1], model2.evaluate(test_images, test_labels, verbose=0)[1]
    train_a3, test_a3 = model3.evaluate(train_images, train_labels, verbose=0)[1], model3.evaluate(test_images, test_labels, verbose=0)[1]
    train_a50, test_a50 = model50k.evaluate(train_images, train_labels, verbose=0)[1], model50k.evaluate(test_images, test_labels, verbose=0)[1]
    val_a1 = hist1.history['val_accuracy'][-1]
    val_a2 = hist2.history['val_accuracy'][-1]
    val_a3 = hist3.history['val_accuracy'][-1]
    val_a50 = hist50k.history['val_accuracy'][-1]
    print('\n' + '=' * 70)
    print('SUMMARY TABLE (copy into your report for the comparison question)')
    print('=' * 70)
    print(f'{"Model":<22} {"Params":>10} {"Train acc":>10} {"Val acc":>10} {"Test acc":>10}')
    print('-' * 70)
    print(f'{"Model 1 (FC)":<22} {model1.count_params():>10} {train_a1:>10.2%} {val_a1:>10.2%} {test_a1:>10.2%}')
    print(f'{"Model 2 (CNN)":<22} {model2.count_params():>10} {train_a2:>10.2%} {val_a2:>10.2%} {test_a2:>10.2%}')
    print(f'{"Model 3 (Sep CNN)":<22} {model3.count_params():>10} {train_a3:>10.2%} {val_a3:>10.2%} {test_a3:>10.2%}')
    print(f'{"Model 50k (best)":<22} {model50k.count_params():>10} {train_a50:>10.2%} {val_a50:>10.2%} {test_a50:>10.2%}')
    print('=' * 70)
