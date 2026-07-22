# Choosing Data for Twinning

Data provided to ReplayBG must comply with strict format requirements **and**
should be selected following some best practices. This page covers both.

## Format requirements

ReplayBG consumes a pandas `DataFrame` with **one row every 5 minutes**. The data
classes read the following columns:

| Column        | Type / units          | Notes                                                    |
|---------------|-----------------------|----------------------------------------------------------|
| `t`           | datetime              | Homogeneous 5-minute grid; **must** be `pd.to_datetime`. Drives `tsteps` and the time-of-day. |
| `glucose`     | float, mg/dL          | CGM. `NaN` allowed — missing samples are excluded from the fit. |
| `cho`         | float, g              | Ingested carbs. `0` when no meal. |
| `bolus`       | float, U              | Bolus insulin. `0` when no bolus. |
| `basal`       | float, U/min          | Basal insulin rate (per row). Its mean defines `u2ss`. |
| `bolus_label` | str                   | Label of each bolus event (empty string is fine if unused). |
| `cho_label`   | str                   | Meal type: `B`/`L`/`D`/`S`/`H` (breakfast/lunch/dinner/snack/hypo-treatment). |

Any other columns are ignored.

**Label handling depends on the blueprint:**

- **[Single meal](blueprints/single_meal.md)** — `cho_label` is ignored; all carbs
  are treated as a single meal.
- **[Multi meal](blueprints/multi_meal.md)** — `cho_label` routes each meal into
  its `meal_B`/`meal_L`/`meal_D`/`meal_S`/`meal_H` channel.
- **[Multi meal extended](blueprints/multi_meal_extended.md)** — additionally
  routes the second-occurrence labels `B2`/`L2`/`S2`.

!!! note "Simulation length"
    The number of integration steps is derived automatically from the `t` column
    (and the 5-minute data sampling time). You do not set it explicitly.

!!! warning
    The `cho` and `bolus` columns should contain at least one event each when
    twinning a segment, so the meal-absorption and insulin parameters have
    something to fit against.

## Best practices

The portion of data you select matters as much as its format. Here are the key
considerations.

### Starting point

The twinning procedure does **not** estimate the model's initial conditions.
Instead, ReplayBG assumes all state variables start at **steady state**. That
assumption is valid only when the actions of exogenous insulin and carbohydrate
intake are "exhausted" — i.e. when the starting point is reasonably distant from
the last meal and insulin bolus, e.g. **~4 hours**.

Pick a starting point that is sufficiently far from the last recorded input so
"the data start from steady state" holds.

!!! tip "Intervals are different"
    When twinning **[multi-day intervals](twinning.md#twinning-multi-day-intervals)**,
    the steady-state assumption only applies to the **first** segment. Every
    subsequent segment starts from the previous segment's final state
    (`x0` + `theta_prev`), so it does not need to start from steady state.

### Minimum data length

As a rule of thumb, use portions of data spanning **at least 6 hours**. As shown
in the literature, this yields better parameter estimates and simulation results.

### Data gaps

To keep twinning reliable, **discard** data portions that have:

- significant gaps — **more than 10%** of missing glucose readings, or
- **no** reported meal intake or insulin bolus at all.

Twinning such portions risks producing digital twins that do not represent the
actual underlying physiology.

## During replay

When [replaying](replay.md) rather than twinning, some of the above relaxes:

- `glucose` is ignored (replay predicts it; it is not fitted).
- Meal and insulin inputs can be **generated online** by [callbacks](replay.md#callbacks)
  rather than read from the data — e.g. a bolus calculator, a hypo-treatment
  policy, or a full control algorithm. In that case the corresponding columns are
  starting points that the callbacks may override step by step.
