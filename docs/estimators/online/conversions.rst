Advanced: Converting Coordinate Systems
=======================================

The ranges of sensible values for inputs to physics models are often problematic for filters.
Some variables may be only valid in strict regions or the scales between variables vary widely enough to expose numerical issues.
The :class:`~moirae.estimators.online.filters.conversions.ConversionOperator` provides a route to bypassing such issues.

Using a Coordinate Converter
----------------------------

The :class:`~moirae.estimators.online.filters.base.ModelWrapper` accepts converters for each of the hidden state,
control system, and outputs.
Coordinate converters modify the :class:`~moirae.estimators.online.filters.base.ModelWrapper` such
that a :class:`~moirae.estimators.online.filters.base.BaseFilter` operates on a different coordinate system without modification.
Conversion to and from the model's coordinate system occurs when the filter invokes
methods from the model.

The :class:`~moirae.estimators.online.OnlineEstimator` uses the conversion operators to
transpose states between different constituent filters (which may use different coordinate systems)
and communicating the estimated state to users.
One can change the coordinate system used by a filter independently from other filters
and without affecting tools which use the estimator.

Achieving the above goals requires functions which convert single points and covariances between coordinate systems.
Change in covariances are estimated using a `first-order Taylor expansion <https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Non-linear_combinations>`_,
which provides an exact result for our most-common operator (linear).
The :meth:`~moirae.estimators.online.filters.distributions.MultivariateRandomDistribution.convert` method of
a :class:`~moirae.estimators.online.filters.distributions.MultivariateRandomDistribution`
employs point and covariance conversions to produce a new distribution in the desired coordinate system.

Selecting a Conversion Operator
-------------------------------

Coordinate systems available for models may be problematic for many reasons, and we provide different filters for each:

- *Disparate Scales* can be normalized with the :class:`~moirae.estimators.online.filters.conversions.LinearConversionOperator`.
  Provide anticipated mean and variance for each variable, which will be used to scale
  and then added to coordinates before passing to the model.

.. code-block:: python

    from moirae.estimators.online.filters.conversions import LinearConversionOperator

    operator = LinearConversionOperator(additive_array=mean, multiplicative_array=std)

- *Strongly Correlated Variables* can be combined to form a lower-dimensional coordinate systems
  with the :class:`~moirae.estimators.online.filters.conversions.LinearOperator`. (Example TBD)
- *Variables with Defined Ranges* can be enforced by employing the (TBD).