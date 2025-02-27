Estimators
==========

The estimators in Moirae determine the `health and transient states <system-models.html>`_
for a storage system provided observations of it operating.

Estimators come in several categories:

- **Online Estimators** which update parameter estimates after each new observation.
- **Offline Estimators** which generate a single parameter estimate using all observations.

.. contents::
   :local:
   :depth: 2

Offline Estimators
------------------

.. :: Add labels for the symbols into the figures

.. image:: ../_static/explain-offline.svg
    :alt: Offline estimators determine the initial state and ASOH which minimize a loss function.
    :align: center
    :width: 75 %

The :class:`~moirae.estimators.online.OfflineEstimator` defines the interface for all offline estimators.
The **Estimator** finds the minimum of a **Loss** function by adjusting inferences
for both the initial transient state of a system and any state-of-health parameters
`marked as updatable <../system-models.html#controlling-which-parameters-are-updatable>`_.

The **Loss** function translates the inputs from the estimator into the
initial state (:math:`h_0`) and ASOH parameters (:math:`\theta`)
then uses those parameters to simulate the evolution of the system according to a **Model**
following the inputs (:math:`u`) provided in operation **Data**.
Loss functions typically compare the voltage observed in the data (:math:`y`) to that
predicted by the model (:math:`y^\prime`)
The objective returns a scalar fitness metric used by the Estimator to find best parameters.

Building an Estimator
+++++++++++++++++++++

First construct an objective function for the optimizer, which requires

1. The :class:`~moirae.models.base.CellModel` defining system physics
2. A starting guess for the transient state
3. A starting guess for the state of health
4. The observation data as a :class:`~batdata.data.BatteryDataset`

.. code-block:: python

    loss = MeanSquaredLoss(
        cell_model=ecm,
        transient_state=state,
        asoh=asoh,
        observations=dataset
    )

Then provide the objective function to an ``OfflineEstimator`` class along with
any options related to how that optimizer functions.

.. code-block:: python

    scipy = ScipyMinimizer(loss, method='Nelder-Mead')

Using an Estimator
++++++++++++++++++

Begin the state estimation by calling the ``estimate`` method of the Estimator,
which optimizes the transient state and ASOH parameters.

.. code-block:: python

    state_0, asoh, result = scipy.estimate()

The ``state_0`` is an estimate for the starting transient state,
``asoh`` is an estimate for the state of health during the entire
extent of the battery data,
and ``result`` is a diagnostic measure specific to the Estimator.

Online Estimators
-----------------

.. :: Perhaps it would be good to provide another version of this figure that would illustrate how to accommodate additional filters. -noah

.. image:: ../_static/explain-filter.svg
    :alt: Online estimators receive data from a battery dataset then pass inputs and outputs to Filters, which rely on Models to estimate state changes and measurements
    :align: center
    :width: 75 %

The :class:`~moirae.estimators.online.OnlineEstimator` defines the interface for all online estimators.
The **Estimator** operates using at least one **Filter**, which each rely on a **Model** to estimate
how parameters evolve with time.

Building an Estimator
+++++++++++++++++++++

The online estimator is composed of one or more filters which estimate the values of different parts
of the battery state in tandem.
The framework in which the filters interact is defined by the choice of
:class:`~moirae.estimators.online.OnlineEstimator`, which includes the
:class:`~moirae.estimators.online.joint.JointEstimator`.

Build an estimator by first constructing a :class:`~moirae.estimators.online.utils.model.BaseCellWrapper` that defines how
to update or estimate the measurements of a system for each subset of variables being estimated.
Each estimator how the they interact with the underlying :class:`~moirae.models.base.CellModel`
and :class:`~moirae.models.base.DegradationModel`.
Moirae provides a library of wrappers with a consistent interface so that
all estimators can work with any Filter or Model.
For example, the :class:`~moirae.estimators.online.joint.JointEstimator` requires
the :class:`~moirae.estimators.online.utils.model.JointCellModelWrapper`.

.. code-block:: python

    cell_function = JointCellModelWrapper(
      cell_model=ecm,
      asoh=rint_asoh,
      transients=rint_transient,
      input_template=rint_inputs,
      asoh_inputs=('r0.base_values',),
    )


Build the filters that will update the estimate of parameters next.
Every type of filter requires the model wrapper and initial estimates for the values of parameters.
The initial estimates for parameters and the inputs to the system are defined as
`probability distributions <source/estimators.html#module-moirae.estimators.online.filters.distributions>`_,
which are created from NumPy arrays of parameters.
The `Unscented Kálmán Filter <https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter>`_
is a common choice:

.. code-block:: python

    ukf = UKF(
      model=cell_function,
      initial_hidden=MultivariateGaussian(
        mean=np.array([0., 0., 0.05]),  # Three parameters: SOC, hysteresis, R0
        covariance=np.diag([0.01, 0.01, 0.01])
      ),
      initial_controls=MultivariateGaussian(
        mean=np.array([0., 1., 25.]),  # Three inputs: Time, Current, Temperature
        covariance=np.diag([0.001, 0.001, 0.5])
      )
    )


Assemble the filters together to form the estimator as the last step.

.. code-block:: python

    ukf_joint = JointEstimator(joint_filter=ukf)

Estimators provide class methods that assemble common patterns of wrapper and filters in a single step.
Read the documentation on each filter type for further details.

.. toctree::
    :maxdepth: 2

    online/joint
    online/dual
    online/conversions


Using an Estimator
++++++++++++++++++

Use the estimator by calling the ``step`` function to update the estimated state
provided a new observation of the outputs of the system.

The ``step`` function returns a probability distribution of the expected state
and expected outputs.

.. code-block:: python

    # Generate inputs and expected outputs
    next_inputs = ECMInput(time=1., current=1.)
    expected_transients = ECMTransientVector.provide_template(has_C0=False, num_RC=0)
    next_outputs = ecm.calculate_terminal_voltage(next_inputs, expected_transients, rint_asoh)

    # Step the estimator
    state_dist, output_dist = ukf_joint.step(
      next_inputs,
      next_outputs
    )

All estimators provide access to the state through the ``estimator.state`` attribute,
which can include elements from the transient and ASOH.

Retrieve the identities of each state variable using ``estimator.state_names``
or access the current estimates for the transient state and ASOH via the
``get_estimated_state`` method.
