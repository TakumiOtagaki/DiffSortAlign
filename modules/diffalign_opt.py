from modules.diffalign import get_alignscore_fn
import jax
from jax import numpy as jnp
import optax
from modules.util import seqprof_to_seq


def alignment(seq1, seq2, epochs, lr, alpha=0.1, a=2.0, b=-1.0, g = -0.0, gap_block_penalty=4.0, regularization_strength=0.1):
  alignscore_fn = get_alignscore_fn(seq1, seq2, alpha, a, b, g, gap_block_penalty, regularization_strength)
  
  theta1 = jax.random.normal(jax.random.PRNGKey(0), (len(seq2),))
  theta2 = jax.random.normal(jax.random.PRNGKey(1), (len(seq1),))

  loss_fn = lambda theta1, theta2:  - alignscore_fn(theta1, theta2)[0]
  grad_fn = jax.grad(loss_fn, argnums=(0, 1))


  schedule = optax.linear_schedule(
      init_value = lr,
      end_value = 0.0,
      transition_steps = epochs
  )
  optimizer = optax.adam(learning_rate=schedule)
  opt_state = optimizer.init((theta1, theta2))

  loss_list = [0 for _ in range(epochs)]

  for epoch in range(epochs):
    loss = loss_fn(theta1, theta2)
    grads = grad_fn(theta1, theta2)
    updates, opt_state = optimizer.update(grads, opt_state)
    theta1, theta2 = optax.apply_updates((theta1, theta2), updates)

    loss_list[epoch] = loss

    if epoch % 20 == 0:
      print(f"epoch: {epoch}, loss: {loss}, ")
      w1, w2 = alignscore_fn(theta1, theta2)[1:]
      seq1_aligned = seqprof_to_seq(w1)
      seq2_aligned = seqprof_to_seq(w2)
      print(f"seq1_aligned: {seq1_aligned}")
      print(f"seq2_aligned: {seq2_aligned}")
  final_score, w1, w2 = alignscore_fn(theta1, theta2)
  return theta1, theta2, final_score, w1, w2, loss_list


    

  
