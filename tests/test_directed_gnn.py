import json
from pathlib import Path
import numpy as np
import torch
from codl.eeg.directed_pdc import operator, synthetic_direction, make_graph
from codl.eeg.directed_gnn import DirectedCheb, DirectedGNN


def test_known_pdc_direction():
    assert synthetic_direction()['direction_test'] == 'passed'


def test_directed_polynomial_orientation():
    a = np.zeros((3,3)); a[1,0] = 1; a[2,1] = 1
    lap = torch.tensor(operator(a),dtype=torch.float32)
    x = torch.tensor([[1.],[0.],[0.]])
    conv = DirectedCheb(1,1,3)
    with torch.no_grad():
        conv.weight.zero_(); conv.bias.zero_(); conv.weight[1] = 1
    torch.testing.assert_close(conv(x,lap),torch.tensor([[0.],[-1.],[0.]]))
    with torch.no_grad():
        conv.weight.zero_(); conv.weight[2] = 1
    torch.testing.assert_close(conv(x,lap),(2*lap@lap-torch.eye(3))@x)


def test_short_window_finite_graph_and_car_rank():
    x = np.random.default_rng(42).normal(size=(19,200))
    assert np.linalg.matrix_rank(x-x.mean(axis=0)) == 18
    a,diag = make_graph(x)
    assert a.shape == (4,19,19) and np.isfinite(a).all()
    assert diag['spectral_radius'] < 1 and diag['n_effective'] == 195
    assert np.all(a[:,np.arange(19),np.arange(19)] == 0)


def test_model_checkpoint_and_embedding(tmp_path):
    s = json.loads((Path(__file__).parents[1]/'src/codl/eeg/gnn_config.json').read_text())
    model = DirectedGNN(s).eval()
    x = torch.randn(2,19,200); lap = torch.zeros(2,19,19)
    with torch.no_grad():
        z = model.encode(x,lap); p = model(x,lap)
    assert z.shape == (2,256) and p.shape == (2,)
    torch.save(model.state_dict(),tmp_path/'state.pt')
    restored = DirectedGNN(s).eval()
    restored.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(x,lap),p)


def test_finite_monitor_does_not_change_training():
    from codl.eeg.finite_monitor import FiniteMonitor
    s = json.loads((Path(__file__).parents[1]/'src/codl/eeg/gnn_config.json').read_text())
    torch.manual_seed(42)
    x = torch.randn(4,19,200); lap = torch.zeros(4,19,19)
    states = []
    for monitored in [False,True]:
        torch.manual_seed(0)
        model = DirectedGNN(s)
        optimizer = torch.optim.Adam(model.parameters(),lr=s['lr'])
        monitor = FiniteMonitor(model) if monitored else None
        loss = model(x,lap).square().mean()
        loss.backward(); optimizer.step()
        states.append({k:v.clone() for k,v in model.state_dict().items()})
        if monitor:
            assert monitor.report()['nonfinite_count'] == 0
            monitor.close()
    for key in states[0]:
        assert torch.equal(states[0][key],states[1][key])
