from asoh.models.ecm import SingleResistor


def test_rint():
    model = SingleResistor(states=[0, ])

    # Test the parameter functions
    assert model.soh_params == ['r_int', 'ocv_0', 'ocv_1']
    assert model.state_params == ['soc']

    # Make sure it steps appropriately
    x = model.initialize()
    dx = model.dx(x, model.InputContainer({'i': 0.}))
    assert all(i == 0 for i in dx.values())

    dx = model.dx(x, model.InputContainer({'i': 1.}))
    assert all(dx[k] == 0 for k in model.soh_params)
    assert dx['soc'] == 1. / 3600
