System Models
=============

A battery system, regardless of its scale or technology, is represented by...

- A **transient state** representing how the parameters which vary predictably given operating conditions, like the state of charge
- An **advanced state of health (ASOH)** representing the parameters which vary slowly and unpredictably, like the internal resistance
- A set of **input quantities** and **output quantities** that define how a system is used and what signals it produces.
- A **cell model** describing how the transient state evolves over time given the ASOH and inputs.
- A **degradation model** which captures how ASOH parameters are expected to change with time and use.

Transient State, Inputs, and Outputs
------------------------------------

Transient states, inputs, and outputs are represented as a :class:`~moirae.models.base.GeneralContainer` object specific to a certain type of battery.

The :class:`~moirae.models.base.GeneralContainer` stores numeric values corresponding to different states and
can be easily transformed to or from NumPy arrays.

.. code-block:: python

    from moirae.base import GeneralContainer, ScalarParameter
    import numpy as np

    class SimpleTransientState(GeneralContainer):

        soc: ScalarParameter = 0.
        """How much the battery has been charged"""

    state = SimpleTransientState(soc=1.)

    # Parameters are stored as a 2D arrays where the
    #  first dimension is a batch dimension
    assert state.soc == np.array([[1.]])
    assert state.to_numpy() == np.array([[1.]])

All inputs must include the terminal current,
and all outputs must include the time terminal voltage.
Beyond that, implementations can include any and all parameters
necessary to specify a particular storage systems.

.. note::

    The ``InputQuantities`` and ``OutputQuantities`` classes define
    the required names for the time, current, and voltage.

Health Parameters
-----------------

The health parameters for a battery are defined as a subclass of :class:`~moirae.models.base.HealthVariable`.
As with the transient states, health parameters are specific to the type of battery.
Unlike transient states, the way they are defined allows a hierarchical structure
and the ability annotate which parameters are treated as updatable.

Consider the following parameter set as an example

.. code-block:: python

    class Resistance(HealthVariable):
        full: ScalarParameter
        '''Resistance at fully charged'''
        empty: ScalarParameter
        '''Resistance at fully discharged'''

        def get_resistance(self, soc: float):
            return self.empty + soc * (self.full - self.empty)

    class BatteryHealth(HealthVariable):
        capacity: ScalarParameter
        resistance: Resistance

    model = BatteryHealth(capacity=1., resistance={'full': 0.2, 'empty': 0.1})



Accessing Values
++++++++++++++++

All variables are stored as 2D arrays, regardless of whether they are scalar values
(like the theoretical capacity) or vectors (like the open circuit voltage at different charge states).

Access value of a parameter from the Python attributes

.. code-block:: python

    assert np.allclose(model.resistance.full, [[0.2]])  # Attribute is 2D with shape (1, 1)

or indirectly using :meth:`get_parameters`.

.. code-block:: python

    assert np.allclose(model.get_parameters(['resistance.full']), [[0.2]])

The name of a variable within a hierarchical health variable contains the path to its submodel
and the name of the attribute of the submodel separated by periods.
For example, the resistance at full charge is "resistance.full".

Controlling Which Parameters Are Updatable
++++++++++++++++++++++++++++++++++++++++++

No parameters of the ``HealthVariable`` are treated as updatable by default.
As a result, no estimation scheme will alter their values.

Mark a variable as updatable by marking the submodel(s) holding that variable as updatable and
the name of the variable to the :attr:`updatable` of its submodel.
Marking "resistance.empty" is achieved by

.. code-block:: python

    model.updatable.add('resistance')
    model.resistance.updatable.add('empty')

or using the :meth:`mark_updatable` utility method

.. code-block:: python

    model.mark_updatable('resistance.empty')

All submodels along the path to a specific parameter must be updatable for it to be updatable.
For example, "resistance.full" would not be considered updatable if the "resistance" submodel is not updatable

.. code-block:: python

    model.updatable.remove('resistance')
    model.resistance.mark_updatable('full')  # Has no effect yet because 'resistance' is fixed

Setting Values of Parameters
++++++++++++++++++++++++++++

Provide a list of new values and a list of names to the ``update_parameters`` function.

.. code-block:: python

    model.updatable.add('resistance')  # Allows resistance fields to be updated
    model.update_parameters([[0.1]], names=['resistance.full'])

or omit the specific names to set all updatable variables

.. code-block:: python

    assert model.updatable_names == ['resistance.full', 'resistance.empty']
    model.update_parameters([[0.2, 0.1]])  # As a (1, 2) array for 1-sized batch of 2 values

Defining the Cell Physics
-------------------------

All storage systems are represented using a :class:`~moirae.models.base.CellModel`
that provides two functions:

1. updating transient states, and
2. predicting outputs (e.g., terminal voltage)

Cell models hold no state themselves and only implement the physics
that describes how the state of a battery system should evolve with time.
Attributes of a cell model adjust the how the calculations are performed
or are resource-specific configuration,
such as a path to external components.

Changes in the ASOH for a cell are described as :class:`~moirae.models.base.DegradationModel`.
Such models provide a function which updates the current state of health provided
new inputs, transient state, and measurements.

Available Cell Models
+++++++++++++++++++++

Moirae already contains several cell models:

- :class:`~moirae.models.ecm.EquivalentCircuitModel`: A Thevenin circuit model with no additional dependencies beyond those needed for Moirae.
- :class:`~moirae.models.thevenin.TheveninModel`: A Thevenin model which includes a simple thermal model and is built atop a robust ODE solver.
  Consult `the documentation for Thevenin <https://rovi-org.github.io/thevenin/>`_ for installation instructions.

The `"Extending Moirae" documentation <extending.html#adding-a-new-cell-model>`_ explains how to add a new model.
You need not contribute a new Cell Model to Moirae in order for it to work with the estimators
but we would encourage you to.

.. ::

    We still need to....

    1. Describe where any parameters for the degredation model come from
    2. Indicate if there are any additional states held by the degradation model
    3. Provide an index of available ~~Cell~~ and Degredation models
