Estimators
==========

The estimators in Moirae determine the `health and transient states <system-models.html>`_
for a storage system provided observations of it operating.

Estimators come in several categories:

- **Online Estimators** which update parameter estimates after each new observation.
- **Offline Estimators** which generate a single parameter estimate using all observations.

Online Estimators
-----------------

The :class:`~moirae.estimators.online.OnlineEstimator` defines the interface for all online estimators.

All estimators require...

1. A :class:`~moirae.models.base.CellModel` which describes how the system state is expected to change and
   relate the current state to observable measurements.
2. An initial estimate for the parameters of the system, which we refer to as the Advanced State of Health (ASOH).
3. An initial estimate for the transient states of the system
4. Identification of which parameters to treat as hidden state. Many implementations of estimators are composites
   which rely on different estimators to adjust subsets of states separately.

Different implementations may require other information, such as an initial guess for the
probability distribution for the values of the states (transient or ASOH).

Use the estimator by calling the ``step`` function to update the estimated state
provided a new observation of the outputs of the system.
The ``step`` function returns a probability distribution of the expected state
and expected outputs.

.. code-block:: python

    # Create an online estimator
    ukf_joint = JointUnscentedKalmanFilter(
        model=ecm_model,
        initial_asoh=asoh,
        initial_transients=transient,
        initial_inputs=inputs,
    )

    # Generate inputs and expected outputs
    next_inputs = ECMInput(time=1., current=1.)
    expected_transients = ECMTransientVector.provide_template(has_C0=False, num_RC=0)
    next_outputs = ecm_model.calculate_terminal_voltage(next_inputs, expected_transients, asoh)

    # Step the estimator
    output_dist, state_dist = ukf_joint.step(
        next_inputs,
        next_outputs
    )

All estimators provide access to the state through the ``estimator.state`` attribute,
which can include elements from the transient and ASOH.

Retrieve the identities of each state variable using ``estimator.state_names``
or access the data current estimates for the transient and ASOH via the
``estimator.asoh`` and ``estimator.transient`` attributes.
