import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

# --- Параметри ---
N_samples = 1000
batch_size = 100
num_steps = 20000
display_step = 100
learning_rate = 0.01

# --- Дані ---
np.random.seed(42)
X_data = np.random.uniform(0, 1, (N_samples, 1)).astype(np.float32)
y_data = (2 * X_data + 1 + np.random.normal(0, 2, (N_samples, 1))).astype(np.float32)

# --- Плейсхолдери ---
X = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 1))

# --- Змінні та модель ---
with tf.compat.v1.variable_scope('linear-regression'):
    k = tf.Variable(tf.random.normal([1,1], stddev=0.01), name='slope')
    b = tf.Variable(tf.zeros([1,1]), name='bias')
    y_pred = tf.matmul(X, k) + b
    loss = tf.reduce_mean(tf.square(y - y_pred))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# --- Навчання ---
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(1, num_steps+1):
        indices = np.random.choice(N_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]
        _, loss_val, k_val, b_val = sess.run([optimizer, loss, k, b],
                                             feed_dict={X: X_batch, y: y_batch})
        if step % display_step == 0:
            print(f"Епоха {step}: Loss={loss_val:.4f}, k={k_val[0][0]:.4f}, b={b_val[0][0]:.4f}")
    
    # --- Фінальні метрики ---
    y_pred_all = sess.run(tf.matmul(X_data, k) + b)
    final_loss = np.mean((y_data - y_pred_all)**2)
    print("\n--- Фінальні метрики моделі ---")
    print(f"k (нахил) = {k_val[0][0]:.4f}")
    print(f"b (зсув) = {b_val[0][0]:.4f}")
    print(f"MSE (на всіх даних) = {final_loss:.4f}")

    # --- Прогнози ---
    print("\n--- Приклади прогнозів моделі ---")
    for i in range(10):  # показуємо перші 10 прогнозів
        print(f"x={X_data[i,0]:.3f}, y_true={y_data[i,0]:.3f}, y_pred={y_pred_all[i,0]:.3f}")

# --- Графік ---
plt.figure(figsize=(8,6))
plt.scatter(X_data, y_data, alpha=0.5, label='Синтетичні дані')
plt.plot(X_data, y_pred_all, color='red', linewidth=2, label='Навчена пряма')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Лінійна регресія: y = kx + b')
plt.legend()
plt.grid(True)
plt.show()
