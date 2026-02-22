import tensorflow as tf
import numpy as np

def build_pinn(num_layers, num_neurons):
    model = tf.keras.Sequential()
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(units=num_neurons, activation="tanh"))
    model.add(tf.keras.layers.Dense(units=1))
    return model

def get_train_step(system_name, model, optimizer, params, weights):
    """Returns a tf.function specific to the chosen physics model."""
    w_pde, w_bc = weights
    t_train = tf.cast(tf.linspace(0, 10, 100)[:, None], tf.float32)
    t_bc = tf.constant([[0.0]], dtype=tf.float32)

    if system_name == "Newton's Law of Cooling":
        T_init, T_surr, k_cool = params
        
        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                with tf.GradientTape() as t_pde:
                    t_pde.watch(t_train)
                    T_pred = model(t_train)
                dT_dt = t_pde.gradient(T_pred, t_train)
                loss_pde = tf.reduce_mean(tf.square(dT_dt + k_cool * (T_pred - T_surr)))
                loss_bc = tf.reduce_mean(tf.square(model(t_bc) - T_init))
                total_loss = w_pde * loss_pde + w_bc * loss_bc
            
            optimizer.apply_gradients(zip(tape.gradient(total_loss, model.trainable_variables), model.trainable_variables))
            return total_loss
        return train_step

    elif system_name == "Simple Harmonic Oscillator":
        m, k, x0, v0 = params
        omega_sq = k / m
        
        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                with tf.GradientTape() as t1:
                    t1.watch(t_train)
                    with tf.GradientTape() as t2:
                        t2.watch(t_train)
                        x_pred = model(t_train)
                    dx_dt = t2.gradient(x_pred, t_train)
                d2x_dt2 = t1.gradient(dx_dt, t_train)
                
                loss_pde = tf.reduce_mean(tf.square(d2x_dt2 + omega_sq * x_pred))
                
                with tf.GradientTape() as t_bc_v:
                    t_bc_v.watch(t_bc)
                    x_bc_pred = model(t_bc)
                v_bc_pred = t_bc_v.gradient(x_bc_pred, t_bc)
                
                loss_bc = tf.reduce_mean(tf.square(x_bc_pred - x0)) + tf.reduce_mean(tf.square(v_bc_pred - v0))
                total_loss = w_pde * loss_pde + w_bc * loss_bc
                
            optimizer.apply_gradients(zip(tape.gradient(total_loss, model.trainable_variables), model.trainable_variables))
            return total_loss
        return train_step
    

