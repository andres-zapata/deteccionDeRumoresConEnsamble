import tensorflow as tf


def get_metricas(predictions, truths):
  g = tf.Graph()
  with g.as_default():
    logits = predictions
    labels = truths
    acc, acc_op = tf.compat.v1.metrics.accuracy(logits, labels)
    precision, precision_op = tf.compat.v1.metrics.precision(logits, labels)
    recall, recall_op = tf.compat.v1.metrics.recall(logits, labels)
    global_init = tf.compat.v1.global_variables_initializer()
    local_init = tf.compat.v1.local_variables_initializer()
  sess = tf.compat.v1.Session(graph=g)
  sess.run([global_init, local_init])
  (var_acc, var_acc_op) = sess.run([acc, acc_op])
  (var_precision, var_precision_op) = sess.run([precision, precision_op])
  (var_recall, var_recall_op) = sess.run([recall, recall_op])
  f1_score = 2*(var_precision_op*var_recall_op/var_precision+var_recall_op)

  return var_acc_op, var_precision_op, var_recall_op, f1_score
