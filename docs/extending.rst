Extending Moirae
================

Welcome to our in progress page on how to add new features to Moirae.
It will become more organized as the package becomes more mature.
For now, it is disconnected sections written as we build capabilities.

Adding a New Cell Model
-----------------------

Add a new mathematical model for a storage system to Moirae through several steps.

Primer on ``GeneralContainer``
++++++++++++++++++++++++++++++

The inputs, outputs, and transient state of a system are defined using the :class:`~moirae.models.base.GeneralContainer`.
The general container allows addressing batches of variables either by name (helpful when implement mathematical models)
or as a array (helpful for estimation operations agnostic to the meaning of a variable).

Any variable stored in a ``GeneralContainer`` is represented as a 2D numpy array.
The first dimension is the batch dimension and the second indexes values within a vector parameter.

Add new attributes to a subclass of :class:`~moirae.models.base.GeneralContainer` by defining them
as either a :class:`~moirae.models.base.ScalarParameter` or :class:`~moirae.models.base.ListParameter`.

.. code-block:: python

    from moirae.models.base import GeneralContainer, ListParameter, ScalarParameter

    class ExampleContainer(GeneralContainer):
        """A container that holds each type of variable"""

        a: ScalarParameter = 1.
        """A variable which is always one value"""
        b: ListParameter = (2., 3.)
        """A variable which is a vector of any length"""

The ``GeneralContainer`` class is based on the ``BaseModel`` class from `pydantic <https://docs.pydantic.dev/latest/>`_,
which will automatically create an ``__init__`` function for you.
The ``ListParameter`` and ``ScalarParameter`` type decorations provide `validation <https://docs.pydantic.dev/latest/concepts/validators/#annotated-validators>`_
and `serialization <https://docs.pydantic.dev/latest/concepts/serialization/#dictmodel-and-iteration>`_
logic which helps ensure the values of each parameter are 2D numpy arrays.
Set default values using native Python syntax (as above) or
via the `Field class from pydantic <https://docs.pydantic.dev/latest/concepts/fields/>`_.

1. Enumerate the Inputs and Outputs
+++++++++++++++++++++++++++++++++++

The inputs of a system define how the system is being controlled.
We assume, by convention, that the (dis)charge current is an input for all storage system.
Define any others by subclassing :class:`~moirae.models.base.InputQuantities`, which is
a `general container <#primer-on-generalcontainer>`_ with the attribute for
current and the time of the inputs already defined.

The outputs of a system are what is observable about the system state,
which must include the terminal voltage.
Define outputs appropriate for the new system by subclassing
:class:`~moirae.models.base.OutputQuantities`.

2. Specify the Transient State
++++++++++++++++++++++++++++++

The transient state for a system are the independent variables on which the dynamics are defined.
For example, the state of charge of a battery is a state variable.
Define the transient state by creating a new `general container <#primer-on-generalcontainer>`_.

An example for a batter that one state, state of charge, is simple

.. code-block:: python

     class TransientState(GeneralContainer):
        """A container that holds each type of variable"""

        soc: ScalarParameter = 0.
        """How much the battery has been charged. 1 is fully charged, 0 is fully discharged"""

3. Define the Health Parameters
+++++++++++++++++++++++++++++++

The state of health of a system are the parameters included in the dynamic model of a battery.
The coefficients which capture open circuit voltage (OCV) changes with state of charge is a common state variable.
Define the parameters for a new battery system by
subclassing :class:`~moirae.models.base.HealthVariable`.

Attributes which represents a health parameter must be
:class:`~moirae.models.base.ScalarParameter` or :class:`~moirae.models.base.ListParameter` type,
an other ``HealthVariable`` class,
or tuples or dictionaries of ``HealthVariable`` classes.

Consider the example battery with a series resistor and polynomial model for OCV below.

.. code-block:: python

    from typing import Union

    import numpy as np
    from numpy.polynomial.polynomial import polyval
    from pydantic import Field

    from moirae.models.base import HealthVariable, ListParameter, ScalarParameter


    class OpenCircuitVoltage(HealthVariable):
        coeffs: ListParameter = [1, 0.5]
        """Parameters of a power-series polynomial"""

        def get_ocv(self, soc: Union[float, np.ndarray]) -> np.ndarray:
            """Compute the OCV as a function of SOC"""
            return polyval(soc, self.coeffs.T, tensor=False)

    class BatteryHealth(HealthVariable):
        ocv: OpenCircuitVoltage = Field(default_factory=OpenCircuitVoltage)
        r: ScalarParameter = 0.01

Note how the ``OpenCircuitVoltage`` is a Python class and, therefore, can provide methods which
operate on its attributes.
The coefficients of the polynomial are a vector of unlimited length, which we specify
using the ``ListParameter`` type.

The ``BatteryHealth`` class uses the ``OpenCircuitVoltage`` as one of its attributes
and a scalar value for the resistance using ``ScalarParameter``.
The default value for the OCV is set using the "default factory" feature of
pydantic so that each instance of ``BatteryHealth`` receives a separate instance of ``OpenCircuitVoltage``.


4. Build a Cell Model
+++++++++++++++++++++

The last step is to define the relationship
between inputs, transient state, health parameters, and output via the :class:`~moirae.models.base.CellModel`.

A cell model contains two functions: update the transient state, and generate expected outputs.

Consider the example for the series resistor model below

.. code-block:: python

    from moirae.models.base import CellModel

    class RintModel(CellModel):

        def update_transient_state(
                self,
                previous_inputs: InputQuantities,
                new_inputs: InputQuantities,
                transient_state: TransientState,
                asoh: BatteryHealth
        ) -> TransientState:
            new_output = transient_state.model_copy(deep=True)  # Return a new copy
            dt = new_inputs.time - previous_inputs.time
            new_output.soc = transient_state.soc + new_inputs.current * dt / 3600.
            return new_output

        def calculate_terminal_voltage(
                self,
                new_inputs: InputQuantities,
                transient_state: TransientState,
                asoh: BatteryHealth) -> OutputQuantities:
            v = new_inputs.current * asoh.r + asoh.ocv.get_ocv(transient_state.soc)
            return OutputQuantities(terminal_voltage=v)

A few points to note:

- Values of health and transient state are accessible as attributes
- The update function returns a *new* transient state object
- The logic here uses `NumPy's broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ to handle
  batches of inputs. Models that do not use NumPy may require inspecting the ``batch_size`` of the states.
- It is acceptable to change the type annotations to match subclass.
  ``RintModel`` expects the ``asoh`` to be a ``BatteryHealth`` class rather than a generic ``HealthVariable``.
