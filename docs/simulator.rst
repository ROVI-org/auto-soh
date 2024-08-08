Simulator
=========

The :class:`~moirae.simulator.Simulator` facilitates generating synthetic data for a battery system.

Create a Simulator by supplying the :class:`~moirae.models.base.CellModel` which defines the system physics,
and the initial state of health, transient state, and inputs.

Defining and Evolving a System
------------------------------

.. code-block:: python

    simulator = Simulator(
        model=ecm,
        asoh=asoh,
        transient_state=transient,
        initial_input=inputs,
        keep_history=True
    )

The Simulator represents a single instance of the battery system.
Evolve the state by either supplying inputs individually

.. code-block:: python

    for time in np.arange(10.)[1:]:
        simulator.step(ECMInput(time=time, current=1.))

or via a list of inputs

.. code-block:: python

    inputs = [ECMInput(time=time, current=1.) for time in np.arange(10.)[1:]]
    simulator.evolve(measurements)


Accessing Results
-----------------

The easiest route for retrieving results of the simulation
are to let the Simulator track results for you
by setting ``keep_history`` to True when creating the class.

Access the history from

- ``input_history`` for the history of the inputs,
- ``transient_history`` for the history of the hidden states of the system,
- or ``measurement_history`` for the history of the measurable outputs.


The ``to_dataframe`` option returns all three types of history as a single Pandas dataframe.

Track the history yourself by using the measurable outputs
and states returned by the ``step`` and ``evolve`` functions.
The current state of the system is also always available as the
``transient_state`` and the last output as the ``measurement`` attributes.