from datetime import datetime

import numpy as np
import pandas as pd

from environment import Environment


class MultiMealExtendedT1DData:
    """Prepare and store time-series data for the ReplayBG simulation pipeline.

    The class converts a pandas dataframe into the arrays and metadata needed by
    the model and twinning procedure. It extracts time, glucose, insulin, and
    meal information, and organizes them into regularly sampled vectors that can
    be consumed by the simulator. This extended variant adds second-occurrence
    meal channels (``B2``, ``L2``, ``S2``) to support simulations spanning more
    than one day.

    ...
    Attributes
    ----------
    data_to_input : dict
        Mapping from input-channel index to attribute name used to build ``self.u``.
    u2ss : float
        Mean basal insulin value, used as a steady-state basal estimate.
    body_weight : float
        Patient body weight used to normalize inputs.
    yts : int
        Number of simulation integration steps per data sample.
    tsteps : int
        Total number of simulation steps.
    tysteps : int
        Number of data-sampling steps represented in the simulation.
    t_data : numpy.ndarray
        Original time values from the dataframe.
    t_hour : numpy.ndarray
        Hour value expanded to simulation resolution.
    t_min : numpy.ndarray
        Minute value expanded to simulation resolution.
    y : numpy.ndarray
        Glucose values as a float array.
    y_idxs : numpy.ndarray
        Indices of non-missing glucose observations.
    basal : numpy.ndarray
        Basal insulin input expanded to simulation resolution.
    bolus : numpy.ndarray
        Bolus insulin input expanded to simulation resolution.
    bolus_label : numpy.ndarray
        Labels associated with bolus events.
    meal : numpy.ndarray
        Meal carbohydrate input expanded to simulation resolution.
    meal_announcement : numpy.ndarray
        Meal announcement signal.
    meal_type : numpy.ndarray
        Labels associated with meal events.
    forcing_ip : numpy.ndarray
        External forcing added directly to the plasma insulin compartment
        (zero unless overridden after construction).
    forcing_ra : numpy.ndarray
        External forcing added directly to the glucose rate of appearance
        (zero unless overridden after construction).
    u : numpy.ndarray
        Combined input matrix used by the model.

    Methods
    -------
    None
    """

    def __init__(self, data: pd.DataFrame = None, data_to_input=None, body_weight=100., environment: Environment = None, ):
        """Constructs all the necessary attributes for the data object from a dataframe.

        Parameters
        ----------
        data : pandas.DataFrame, optional, default : None
            Input dataframe containing the raw replay data.
        data_to_input : dict, optional, default : None
            Mapping from input indices to attribute names used to assemble the
            model input matrix. If ``None``, a default mapping appropriate to
            this data variant is used.
        body_weight : float, optional, default : 100.
            Patient body weight used to normalize insulin and meal inputs.
        environment : Environment, optional, default : None
            Environment object containing simulation settings such as the
            integration time step.

        Raises
        ------
        AttributeError
            If required dataframe columns are missing.
        TypeError
            If ``data`` is not a pandas dataframe or time values are not in the
            expected format.
        """
        if data_to_input is None:
            self.data_to_input = {0: 'meal_B',
                                  1: 'meal_L',
                                  2: 'meal_D',
                                  3: 'meal_S',
                                  4: 'meal_H',
                                  5: 'meal_B2',
                                  6: 'meal_L2',
                                  7: 'meal_S2',
                                  8: 'bolus',
                                  9: 'basal',
                                  10: 't_hour',
                                  11: 'forcing_ip',
                                  12: 'forcing_ra'}
        else:
            self.data_to_input = data_to_input

        data = data.reset_index(drop=True)

        self.u2ss = np.mean(data.basal.values) * 1000 / body_weight
        self.body_weight = body_weight

        self.yts = 5

        # From the time retain only the hour since is the only thing actually needed during the simulation
        self.__time_setup(data, environment)

        # Set y (glucose) from given data
        self.y = data.glucose.values.astype(float)
        self.y_idxs = np.where(~np.isnan(self.y))[0]

        # Set insulin from given data
        self.__insulin_setup(data)
        self.__meal_setup(data)

        # Set the forcing inputs
        self.forcing_ip = np.zeros([self.tsteps, ])
        self.forcing_ra = np.zeros([self.tsteps, ])

        # Create u
        self.__setup_u()

    def __time_setup(self,
                     data: pd.DataFrame,
                     environment: Environment = None,
                     ) -> None:
        """Unpacks time information and expands it to simulation resolution.

        Parameters
        ----------
        data : pandas.DataFrame
            Input dataframe containing a ``t`` column with timestamps.
        environment : Environment, optional, default : None
            Environment object providing the integration step count.

        Returns
        -------
        None
        """
        # Prefer pandas Timedelta: np.datetime64.astype(datetime) returns an int on modern NumPy.
        t = pd.to_datetime(data.t)
        self.tsteps = int(
            (t.iloc[-1] - t.iloc[0]).total_seconds() / 60 + self.yts
        ) * environment.ts  # number of steps in the simulation with the sampling rate of the integration step
        self.tysteps = int(
            self.tsteps / self.yts)  # number of steps of the simulation, with the sampling rate of the data

        self.t_data = data['t'].to_numpy()  # times in the data

        self.t_hour = np.zeros([self.tsteps, ])  # hours in the data for each integration step
        self.t_min = np.zeros([self.tsteps, ])  # minutes in the data for each integration step
        t_m = np.array(data.t.dt.minute.values).astype(int)
        t_h = np.array(data.t.dt.hour.values).astype(int)

        self.t_start = t_h[0] * 60 + t_m[0]

        for t in range(data.shape[0]):
            self.t_hour[(t * self.yts):((t + 1) * self.yts)] = t_h[t]
            self.t_min[(t * self.yts):((t + 1) * self.yts)] = np.arange(t_m[t], t_m[t] + self.yts) % 60

    def __insulin_setup(self,
                        data: pd.DataFrame,
                        ) -> None:
        """Unpacks insulin measurements into basal and bolus arrays.

        Parameters
        ----------
        data : pandas.DataFrame
            Input dataframe containing insulin-related columns such as
            ``basal``, ``bolus``, and ``bolus_label``.

        Returns
        -------
        None
        """
        self.basal = np.zeros([self.tsteps, ])
        self.bolus = np.zeros([self.tsteps, ])
        self.bolus_label = np.empty([self.tsteps, ], dtype=str)

        self.bolus_data = data.bolus.values

        # Find the boluses
        b_idx = np.where(data.bolus)[0]

        # Set the bolus vector
        for i in range(np.size(b_idx)):
            self.bolus[(b_idx[i] * self.yts): ((b_idx[i] + 1) * self.yts)] = data['bolus'][b_idx[
                i]] * (1000 / self.body_weight)  # mU/(kg*min)
            self.bolus_label[(b_idx[i] * self.yts): ((b_idx[i] + 1) * self.yts)] = data['bolus_label'][b_idx[i]]

        self.basal_data = data.basal.values
        # Set the basal vector
        for time in range(self.tysteps):
            self.basal[(time * self.yts): ((time + 1) * self.yts)] = \
                data['basal'][time] * (1000 / self.body_weight)  # mU/(kg*min)

    def __meal_setup(self,
                     data: pd.DataFrame,
                     ) -> None:
        """Unpacks meal and carbohydrate announcement information.

        Parameters
        ----------
        data : pandas.DataFrame
            Input dataframe containing meal-related columns such as ``cho`` and
            ``cho_label``.

        Returns
        -------
        None
        """
        # Initialize the meal vector
        self.meal = np.zeros([self.tsteps, ])

        # Initialize the mealAnnouncements vector
        self.meal_announcement = np.zeros([self.tsteps, ])

        # Initialize the meal type vector
        self.meal_type = np.empty([self.tsteps, ], dtype=str)

        self.meal_B = np.zeros([self.tsteps, ])
        self.meal_L = np.zeros([self.tsteps, ])
        self.meal_D = np.zeros([self.tsteps, ])
        self.meal_S = np.zeros([self.tsteps, ])
        self.meal_H = np.zeros([self.tsteps, ])

        self.meal_B2 = np.zeros([self.tsteps, ])
        self.meal_L2 = np.zeros([self.tsteps, ])
        self.meal_S2 = np.zeros([self.tsteps, ])

        self.meal_data = data.cho.values

        # Find the meals
        m_idx = np.where(data.cho)[0]

        # Set the main meal vector
        for i in range(np.size(m_idx)):
            self.meal[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = data['cho'][m_idx[
                i]] * (1000 / self.body_weight)  # mg/(kg*min)
            self.meal_announcement[(m_idx[i] * self.yts)] = data['cho'][m_idx[i]] * self.yts  # mg/(kg*min)

            self.meal_type[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = data['cho_label'][m_idx[i]]

            if data['cho_label'][m_idx[i]] == 'B':
                self.meal_B[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]
            if data['cho_label'][m_idx[i]] == 'L':
                self.meal_L[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]
            if data['cho_label'][m_idx[i]] == 'D':
                self.meal_D[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]
            if data['cho_label'][m_idx[i]] == 'S':
                self.meal_S[(m_idx[i] * self.yts):(
                        (m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]
            if data['cho_label'][m_idx[i]] == 'H':
                self.meal_H[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]

            if data['cho_label'][m_idx[i]] == 'B2':
                self.meal_B2[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]
            if data['cho_label'][m_idx[i]] == 'L2':
                self.meal_L2[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]
            if data['cho_label'][m_idx[i]] == 'S2':
                self.meal_S2[(m_idx[i] * self.yts):(
                        (m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]

    def __setup_u(self):
        """Builds the combined model input matrix from configured inputs.

        Returns
        -------
        None
        """
        self.u = np.empty((self.tsteps, len(self.data_to_input.keys())))
        for i in range(len(self.data_to_input.keys())):
            self.u[:, i] = getattr(self, self.data_to_input[i])