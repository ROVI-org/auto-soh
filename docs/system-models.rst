System Models
=============

A battery system, regardless of its scale or technology, is represented by...

- A **transient state** representing how the parameters which vary predictably given operating conditions, like the state of charge
- An **advanced state of health (ASOH)** representing the parameters which vary slowly and unpredictably, like the internal resistance
- A set of **input quantities** and **output quantities** that define how a system is used and what signals it produces.
- A **model** describing the transient state evolves over time given the ASOH and inputs.

Transient State, Inputs, and Outputs
------------------------------------

Transient states, inputs, and outputs are represented as a ``GeneralContainer`` object specific to a certain type of battery.

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

.. note::

    Moirae's models assume that a positive current for batteries charging,
    which is opposite from the `battery-data-toolkit's choice <https://rovi-org.github.io/battery-data-toolkit/schemas.html>`_,
    but in line with conventions common for battery modeling.


Health Parameters
-----------------

The health parameters for a battery are defined as a subclass of ``HealthVariable``.
As with the transient states, health parameters are specific to the type of battery.
Unlike transient states, the way they are defined allows a hierarchical structure
and the ability annotate which parameters are treated as updatable.

.. note:: TODO, figure out how much documentation to copy over from the class docstring

Models
------

All storage systems are represented using a ``CellModel``.
The model class defines how the transient states change given
an inputs applied to a system defined using certain health parameters.