from asoh.models.ecm import SingleResistor


def test_rint():
    model = SingleResistor(health_states=['r_int'])

    # Test the parameter functions
    assert model.health_states == ('r_int',)

    # Make sure it steps appropriately
    x = model.initialize()
    dx = model.dx(x, model.InputContainer({'i': 0.}))
    assert all(i == 0 for i in dx.values())

    dx = model.dx(x, model.InputContainer({'i': 1.}))
    assert all(dx[k] == 0 for k in model.health_states)
    assert dx['soc'] == 1. / 3600
