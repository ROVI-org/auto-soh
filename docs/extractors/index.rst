Extractors
==========

Extractors create initial guesses for model parameters
through model-specific algorithms.
Many algorithms require data from battery cycling performed
under specific operating conditions, which we refer to
as Reference Performance Tests (RPTs).

For example, the :class:`~moirae.extractors.ecm.OCVExtractor` estimates
the open circuit voltage from a cycle which completely charges
and discharges a battery under minimal current.

This part of the documentation describes the general interface for extractors
then each extractor available for each model.

Extractor Interface
-------------------

All extractors provide two operations:

1. :attr:`~moirae.extractors.base.BaseExtractor.check_data` evaluates whether a datasets
   contains data needed to perform the estimation.
2. :attr:`~moirae.extractors.base.BaseExtractor.extract` generates part of a
   `health parameter object <../system-models.html#health-parameter>`_ given a dataset.

Instantiate an extractor by providing configuration options for the algorithm
and, in some cases, information gathered from other extractors.

.. code-block:: python

    from moirae.extractors.ecm import MaxCapacityExtractor, OCVExtractor

    cap = MaxCapacityExtractor().extract(dataset)
    ex = OCVExtractor(capacity=cap)
    ocv = ex.extract(dataset)


Available Extractors
--------------------

Extractors are model-specific, and we describe those available for each type of model below.

Equivalent Circuit Models
~~~~~~~~~~~~~~~~~~~~~~~~~

**Model**: :class:`moirae.models.ecm.EquivalentCircuitModel`

**Demonstration**: `ECM Extractors <demonstrate-ecm-extractors.html>`_

.. toctree::
   :hidden:

   demonstrate-ecm-extractors

Extracting parameters for a Equivalent Circuit Models (ECM) require several steps, each
generating parameters for different parts of the model.

.. list-table::
   :header-rows: 1

   - * Extractor
     * Parameters
     * Data Needs
   - * :class:`~moirae.extractors.ecm.MaxCapacityExtractor`
     * Capacity of battery
     * Complete charge and discharge
   - * :class:`~moirae.extractors.ecm.OCVExtractor`
     * Charge with zero current
     * - Capacity estimate
       - Complete charge and discharge

