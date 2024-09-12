Extending Moirae
================

Welcome to our in progress page on how to add new features to Moirae.
It will become more organized as the package becomes more mature.
For now, it is disconnected sections written as we build capabilities.

Creating a Health Variable
--------------------------

Define a new system state by subclassing ``HealthVariable`` then providing
adding attributes which describe the learnable parameters of a system.

Attributes which represents a health parameter must be numpy arrays,
other ``HealthVariable`` classes,
or lists or dictionaries of other ``HealthVariable`` classes.

The numpy arrays used to store parameters are 2D arrays where the first dimension is a batch dimension,
even for parameters which represent scalar values.
Use the :class:`ScalarParameter` type for scalar values and :class:`ListParameter` for list values
to enable automatic conversion from user-supplied to the internal format used by :class:`HealthVariable`.

.. note::

    The ``ListParameter`` and ``ScalarParameter`` classes also supply methods needed for serialization to
    and parsing form JSON.
