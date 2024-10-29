import jax
from jax import numpy as jnp
from jax import random
import numpy as onp
import sys
from util import  get_scoring_matrix, one_hot_encode
fast_sort_path = "/content/drive/MyDrive/Colab Notebooks/DiffSortAlign/fast-soft-sort/"
sys.path.insert(0, fast_sort_path)
from fast_soft_sort.jax_ops import soft_rank, soft_sort

# values = jnp.array([[1, 2, 3, 4, 0.1, 1.5, 3]])
# print(soft_sort(values, regularization_strength=0.1))
# print(soft_sort(values, regularization_strength=0.2))
# print(soft_rank(values, regularization_strength=1.0))
# print(soft_rank(values, regularization_strength=2.0))


# soft_sort = jax.jit(soft_sort) # unfortunately, it is unable to jax.jit...
# soft_rank = jax.jit(soft_rank)


# @jax.jit
def get_alignscore_fn(seq1, seq2, alpha, a=2.0, b=-1.0, g = -0.0, gap_block_penalty=1.0, regularization_strength=0.1):
  n, m = len(seq1), len(seq2)
  S = get_scoring_matrix(a, b, g)
  onehot_seq1 = one_hot_encode(seq1)
  onehot_seq2 = one_hot_encode(seq2)

  padded_onehot_seq1 = jnp.pad(onehot_seq1, ((0, m), (0, 0)), constant_values=0.0)
  padded_onehot_seq2 = jnp.pad(onehot_seq2, ((0, n), (0, 0)), constant_values=0.0)

  if gap_block_penalty < 0:
    raise ValueError("gap_block_penalty must be non-negative")

  def extend_with_theta(theta1, theta2):
    # return [1, 2, ..., len_seq2] + [
    # y1 = list(range(1, n+1)) + theta1
    # theta1, 2 は jnp.array とする。だから 先頭に m, n padding してやる
    theta1_padded = jnp.pad(theta1, (n, 0))
    theta2_padded = jnp.pad(theta2, (m, 0))
    y1 = jnp.pad(jnp.arange(1, n+1), (0, m)) + theta1_padded
    y2 = jnp.pad(jnp.arange(1, m+1), (0, n)) + theta2_padded
    return y1, y2
  
  def compute_num_gap(w):
    w_gap = w[:, 4]  # ギャップの確率
    # 次の位置のギャップ確率を取得、最後は0でパディング
    w_gap_shift = jnp.concatenate([w_gap[1:], jnp.zeros(1)])
    # ギャップブロックの開始を計算
    gap_block = w_gap * (1 - w_gap_shift)
    num_gap = jnp.sum(gap_block)
    return num_gap

  def sorted2profile(y1, y2, z1, z2):
    d1 = z1[:, jnp.newaxis] - y1[jnp.newaxis, :]
    d2 = z2[:, jnp.newaxis] - y2[jnp.newaxis, :]


    K1 = jnp.exp(-d1 * alpha)
    K2 = jnp.exp(-d2 * alpha)

    p1 = K1 / jnp.sum(K1, axis=1)
    p2 = K2 / jnp.sum(K2, axis=1)

    w1 = jnp.matmul(p1, padded_onehot_seq1)
    w2 = jnp.matmul(p2, padded_onehot_seq2)

    w1 = jax.nn.softmax(w1, axis=1)
    w2 = jax.nn.softmax(w2, axis=1)

    return w1, w2

  def score(w1, w2):
    mmscore = jnp.einsum("ia,ab,ib->", w1, S, w2)
    num_gap1, num_gap2 = compute_num_gap(w1), compute_num_gap(w2)
    return mmscore - gap_block_penalty * (num_gap1 + num_gap2)

  def alignscore_fn(theta1, theta2):
    y1, y2 = extend_with_theta(theta1, theta2)
    z1 = soft_rank(jnp.array([y1]), regularization_strength=regularization_strength)[0]
    z2 = soft_rank(jnp.array([y2]), regularization_strength=regularization_strength)[0]
    # print("sorted: ", z1, z2)
    w1, w2 = sorted2profile(y1, y2, z1, z2)
    return score(w1, w2), w1, w2

  return alignscore_fn




