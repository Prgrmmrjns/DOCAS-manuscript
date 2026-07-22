# The CGM Error Model

During [replay](replay.md), ReplayBG can turn the model's true output
(interstitial glucose) into a **realistic, noisy CGM measurement**. This is what
[callbacks](replay.md#callbacks) observe when you close the loop, so control
policies act on the same imperfect signal a real device would produce.

## The `Sensor` interface

The sensor is deliberately **output-agnostic**: it knows nothing about glucose or
any specific model — it only sees `model.output(k)`. All sensors subclass the
abstract `Sensor` base class (`sensors/sensor.py`).

```python
from sensors.sensor import Sensor
```

| Attribute | Meaning |
|-----------|---------|
| `ts` | Sample time, in **integration steps** (default `5`). The replay loop measures when `k % ts == 0`. |
| `t_offset` | Time offset (used when a sensor is shared across replay runs). |
| `max_lifetime` | Maximum sensor lifetime in integration steps. When exceeded, the loop reconnects a fresh sensor. |
| `connected_at` | The step at which the current sensor was connected. |

| Method | Role |
|--------|------|
| `connect_new(connected_at=0)` | (Re)sample a fresh sensor error realization. |
| `measure(value, past_values, t)` | Produce a measurement from the current output `value`, the past outputs `past_values`, and `t` (days since connection). **Abstract** — implemented by subclasses. |
| `add_offset(to_add)` | Advance the sensor's internal clock (for shared sensors). |

## The default: `Vettoretti19CGM`

The default CGM error model is `Vettoretti19CGM`, the model published in
**Vettoretti et al., *Sensors*, 2019** — a factory-calibrated device with a
10-day lifespan and a 5-minute sample time.

```python
from sensors import Vettoretti19CGM
```

It applies a time-varying **calibration error** (quadratic in time) to the
interstitial glucose, plus **colored noise** from an AR(2) process:

$$
\begin{cases}
IG_S(t) = (a_0 + a_1 t + a_2 t^2) \cdot IG(t) + b_0 \\
CGM(t) = IG_S(t) + v(t)
\end{cases}
$$

with $a_0, a_1, a_2, b_0$ calibration coefficients and $v(t)$ the colored output
noise. On each `connect_new()`, a 7-dimensional parameter vector is sampled from a
fixed mean/covariance (as defined in the paper), rejecting draws until the AR(2)
noise process is stable and the output-noise standard deviation is ≤ 10 mg/dL.

## Using a sensor in a replay

Pass a sensor instance to `replay(sensor=...)`:

```python
from sensors import Vettoretti19CGM

sensor = Vettoretti19CGM()
results = rbg.replay(rbg_data=rbg_data, model=model, sensor=sensor)
```

When a sensor is supplied, `replay()` returns two extra keys:

- `measurement` — the sensor samples at sensor cadence,
- `measurement_time` — the integration-step index of each sample.

and the callbacks see the noisy reading via `ctx.measurement`. When **no** sensor
is supplied, `ctx.measurement` carries the true output, so policies behave as if
perfectly observed.

!!! note "Reproducibility"
    The replay loop reseeds the RNG from `environment.seed` before connecting the
    sensor, so a given `ReplayBG(seed=...)` yields the same noise realization every
    run.

## Writing a custom CGM error model

Subclass `Sensor` and implement `measure`. The example below is a toy model that
just doubles the interstitial glucose:

```python
from sensors.sensor import Sensor

class FakeCGM(Sensor):
    def __init__(self):
        super().__init__()
        self.max_lifetime = 10   # integration steps

    def measure(self, value, past_values, t):
        return 2.0 * value

results = rbg.replay(rbg_data=rbg_data, model=model, sensor=FakeCGM())
```

Because the sensor is output-agnostic, your `measure` only needs the current
output `value`, the history `past_values`, and the time `t` (in days since the
sensor connected).
