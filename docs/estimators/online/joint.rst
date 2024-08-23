Joint Online Estimation
=======================

:class:`~moirae.estimators.online.joint.JointEstimator` treats all variables, state or health, with a single filter.

Assembling Filters
------------------

The joint estimator requires a single filter that acts using a
:class:`~moirae.estimators.online.utils.model.JointCellModelWrapper`.

Create the model wrapper by supplying the :class:`~moirae.models.base.CellModel` and, optionally,
a list of which variables to operate on.

.. code-block:: python

    cell_function = JointCellModelWrapper(
      cell_model=ecm,
      asoh=rint_asoh,
      transients=rint_transient,
      input_template=rint_inputs,
      asoh_inputs=('r0.base_values',),
    )

Use the model to create the Filter then the filter to create the Estimator.

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

    ukf_joint = JointEstimator(joint_filter=ukf)


.. :: Link to examples, discuss strategies for using Joint effectively