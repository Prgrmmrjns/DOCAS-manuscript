# Choosing a Blueprint

Choosing the right **blueprint** is one of the key decisions when using ReplayBG.
A blueprint is the pairing of a **model class** with its matching **data class**:

| Blueprint | Model class | Data class |
|-----------|-------------|------------|
| Single meal | `SingleMealT1DModel` | `SingleMealT1DData` |
| Multi meal | `MultiMealT1DModel` | `MultiMealT1DData` |
| Multi meal extended | `MultiMealExtendedT1DModel` | `MultiMealExtendedT1DData` |

The blueprint is associated with a specific physiological model. It *is* the
resulting digital twin, and it defines the twin's **domain of applicability** and
final capabilities.

!!! important "You choose the blueprint explicitly"
    Unlike the old py_replay_bg (`ReplayBG(blueprint='multi-meal')`), the current
    ReplayBG has **no blueprint string**. You select a blueprint simply by
    importing and instantiating the matching model + data classes. ReplayBG will
    **not** choose one for you based on the data — this is up to you.

    ```python
    from data.multi_meal_t1d_data import MultiMealT1DData
    from model.multi_meal_t1d import MultiMealT1DModel

    rbg_data = MultiMealT1DData(data=df, environment=rbg.environment)
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, t_start=t_start)
    ```

## How to decide

The driver behind the choice is the **data** you want to twin.

- **[Single meal](single_meal.md)** — a period of time when the subject had only
  **one** main meal and a corresponding insulin basal-bolus administration.
  Usually this spans at most 6–8 hours, starts near that meal, and ends before the
  next one.
- **[Multi meal](multi_meal.md)** — a period with **more than one** main meal and
  a basal-bolus regimen. Think of a single day, when multiple meals occur. It
  models per-meal absorption (B/L/D/S/H) and a **time-of-day insulin sensitivity**
  (`SI_B`, `SI_L`, `SI_D`).
- **[Multi meal extended](multi_meal_extended.md)** — the multi-meal model
  augmented with **second-occurrence** meal labels (`B2`, `L2`, `S2`) and an extra
  sensitivity `SI_B2`. Useful for twinning that reaches into a second day so the
  last meals of the primary window are estimated with more data (avoiding the
  "tail effect").

!!! tip
    The **multi-meal** blueprint is the most common choice, since one usually
    deals with data that include more than one meal and span more than 6–8 hours.

## What about data spanning more than one day?

Each multi-meal digital twin represents a **single day** (its equations cover the
meals and the insulin-sensitivity profile of one day, and it starts from steady
state). To cover a longer recording, you create **multiple twins** — one per day
— and "glue" them together by carrying the physiological state forward:

1. split the data into single days;
2. twin the first day starting from steady-state conditions;
3. twin each subsequent day starting from the **final state of the previous day**
   (`model.get_final_x0()` → next `x0`, `model.get_theta()` → next `theta_prev`);
4. iterate.

This eliminates the steady-state assumption for every day except the first. See
the multi-day interval example in the [Twinning Procedure](../twinning.md#twinning-multi-day-intervals).

## Data requirements

Whatever blueprint you pick, the input data must comply with strict format
requirements and should be selected following some best practices (starting
point, minimum length, data gaps). See
[Choosing Data for Twinning](../data_requirements.md).
