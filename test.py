import jax.nn as nn
import jax.random as random
import jax.numpy as numpy
import einops
import optax
from ott.solvers.nn import models, neuraldual
from matplotlib import figure

arr = numpy.array

BATCHSIZE = 128
VALIDSIZE = 10000
BATCHES = 10000
NDIM = 2

neural_f = \
  models.ICNN \
  ( dim_data=NDIM
  , dim_hidden=[16, 16, 16, 16]
  )

neural_g = \
  models.MLP \
  ( dim_hidden=[16, 16, 16, 16]
  , is_potential=False
  )

lr_schedule = \
  optax.cosine_decay_schedule \
  ( init_value=1e-4
  , decay_steps=BATCHES
  , alpha=1e-2
  )

optimizer_f = optax.adam(learning_rate=lr_schedule, b1=0.5, b2=0.5)
optimizer_g = optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999)

neural_dual_solver = \
  neuraldual.W2NeuralDual \
  ( NDIM
  , neural_f
  , neural_g
  , optimizer_f
  , optimizer_g
  , num_train_iters=BATCHES
  )


def iternormal(knext, batchsize):
  while 1:
    k , knext = random.split(knext)
    yield random.normal(k, (batchsize, NDIM))

  return


def itertarget(knext, batchsize, mu, chol):
  while 1:
    k , knext = random.split(knext)
    k1 , knext = random.split(knext)
    proc = random.bernoulli(k1, shape=(batchsize,))
    samps = random.multivariate_normal(k, mu, chol, (batchsize,))
    yield samps + einops.repeat(proc, "h -> h 2")*arr([5, 6])
    # yield random.multivariate_normal(k, mu, chol, (batchsize,))

  return


ksource , ktarg , kvalidsrc , kvalidtarg = random.split(random.PRNGKey(0), 4)

mutarget = arr([-1, -2])
choltarget = arr([[2, 3], [0, 4]])

iters = \
  ( iternormal(ksource, BATCHSIZE)
  , itertarget(ktarg, BATCHSIZE, mutarget, choltarget)
  , iternormal(kvalidsrc, VALIDSIZE)
  , itertarget(kvalidtarg, VALIDSIZE, mutarget, choltarget)
  )

print("training")
learned_potentials = neural_dual_solver(*iters)
print("done training")

validsrc = next(iters[2])
validtarg = next(iters[3])
validtrans = learned_potentials.transport(validsrc)

fig , _ = learned_potentials.plot_ot_map(validsrc, validtarg, forward=True)
fig.savefig("fwd.png")

fig , _ = learned_potentials.plot_ot_map(validsrc, validtarg, forward=False)
fig.savefig("bkwd.png")


for i in range(NDIM):
  fig = figure.Figure((6, 6))
  ax = fig.add_subplot(111)

  ax.hist \
    ( [validsrc[:,i], validtarg[:,i], validtrans[:,i]]
    , bins=25
    , label=["src", "targ", "trans"]
    )

  fig.legend()

  fig.savefig("comp%02d.png" % i)
