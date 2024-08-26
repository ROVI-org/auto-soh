Dual Estimation
===============

The dual estimation works by operating separate filters on the transient state and the state-of-health.

.. :: Make a figure to demonstrate this

Assembling Filters
------------------

The two filters for a dual estimator differ by how the :class:`~moirae.models.base.CellModel` is used.

The *transient filter* relies on a :class:`~moirae.estimators.online.utils.model.CellModelWrapper`
to project how the transient states evolve and affect the measurements.

.. code-block:: python

    cell_wrapper = CellModelWrapper(cell_model=cell_model,
                                    asoh=initial_asoh,
                                    transients=initial_transients,
                                    inputs=initial_inputs)
    transients_hidden = MultivariateGaussian(mean=initial_transients.flatten(),
                                             uncertainty_matrix=covariance_transient)
    trans_filter = UKF(model=cell_wrapper,
                       initial_hidden=transients_hidden,
                       initial_controls=initial_controls)

The *ASOH filter* uses a a :class:`~moirae.estimators.online.utils.model.DegradationModelWrapper`
to evaluate how to tie changes in the state of health to the observed outputs.

.. code-block:: python

    asoh_wrapper = DegradationModelWrapper(cell_model=cell_model,
                                           asoh=initial_asoh,
                                           transients=initial_transients,
                                           inputs=initial_inputs)
    asoh_hidden = MultivariateGaussian(mean=initial_asoh.get_parameters().flatten(),
                                       covariance=covariance_asoh)
    asoh_filter = UKF(model=asoh_wrapper,
                      initial_hidden=asoh_hidden,
                      initial_controls=initial_controls)

Build the estimator with these two filters

.. code-block:: python

    DualEstimator(transient_filter=trans_filter, asoh_filter=asoh_filter)
