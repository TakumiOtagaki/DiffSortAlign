import jax
from jax import numpy as jnp
import numpy as onp
base_to_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '-': 4}

def one_hot_encode(seq):
    indices = jnp.array([base_to_idx.get(base, 4) for base in seq])
    return jax.nn.one_hot(indices, num_classes=5)


def get_scoring_matrix(a=2.0, b=-1.0, g = -0.0):
    """
    スコアリングマトリックスを定義します。

    Parameters:
    -----------
    a : float, optional
        マッチスコア。デフォルトは2.0。
    b : float, optional
        ミスマッチスコア。デフォルトは-1.0。
    
    g : gap score デフォルトは 0.0 なぜなら gap 数最適化を行うから。
    

    Returns:
    --------
    S : jnp.ndarray
        スコアリングマトリックス。形状は (5, 5)。
        S[a, b] は塩基 a と b のスコア。
    """
    # 初期化
    S = a * jnp.ones((5, 5))
    # マッチの場合にスコアを c に設定
    S = S.at[jnp.arange(4), jnp.arange(4)].set(b)
    # ギャップとの組み合わせは0
    S = S.at[:, 4].set(g)
    S = S.at[4, :].set(g)
    return S
  

def seqprof_to_seq(seqprof):
  """
  seqprof を seqに変換する
  """

  seq = ""
  for i in range(len(seqprof)):
    max_index = jnp.argmax(seqprof[i])
    if max_index == 4:
      break
    seq += list(base_to_idx.keys())[max_index]
  return seq
