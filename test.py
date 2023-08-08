import jax.random as random
import jax.numpy as numpy
import optax
from ott.solvers.nn import models, neuraldual
from matplotlib import figure

arr = numpy.array

BATCHSIZE = 128
VALIDSIZE = 1000
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


def iternormal(knext, batchsize, mu, chol):
  while 1:
    k , knext = random.split(knext)
    yield random.multivariate_normal(k, mu, chol, (batchsize,))

  return


musource = arr([0, 0])
cholsource = arr([[1, 0], [0, 1]])
mutarget = arr([-1, -2])
choltarget = arr([[2, 3], [0, 4]])

iters = \
  tuple \
  ( map \
    ( iternormal
    , random.split(random.PRNGKey(0), 4)
    , [ BATCHSIZE ] * 2 + [ VALIDSIZE ] * 2
    , [ musource , mutarget ] * 2
    , [ cholsource , choltarget ] * 2
    ,
    )
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
