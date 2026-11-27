"""Frozen EEG256 + freshly trained team FT tokens and fusion regression head."""
import time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from codl.clinical.ft_clinical import ClinicalPrep, tensors, prep_state, _modules


def fit_concat(e_train,c_train,y_train,e_val,c_val,y_val,columns,settings,seed,architecture="raw_concat"):
    start = time.perf_counter(); torch.manual_seed(seed)
    prep = ClinicalPrep().fit(c_train)
    assert prep.cat.tolist()==[c=='sex' for c in columns]
    scaler = StandardScaler().fit(e_train)
    et = torch.tensor(scaler.transform(e_train),dtype=torch.float32)
    ev = torch.tensor(scaler.transform(e_val),dtype=torch.float32)
    tx,tm = tensors(prep,c_train); vx,vm = tensors(prep,c_val)
    mean,std = float(np.mean(y_train)),max(float(np.std(y_train)),1e-8)
    ty = torch.tensor((y_train-mean)/std,dtype=torch.float32)
    vy = torch.tensor((y_val-mean)/std,dtype=torch.float32)
    _,Net = _modules()
    model = Net(256,prep.cat,settings['nn_token_dim'],settings['nn_hidden'],settings['nn_dropout']) if architecture == 'raw_concat' else build_architecture_a(prep.cat)
    optimizer = torch.optim.AdamW(model.parameters(),lr=settings['nn_lr'],weight_decay=settings['nn_wd'])
    criterion = torch.nn.HuberLoss(delta=1.)
    best,epoch_best,wait,state = float('inf'),0,0,None
    history = []
    probe = None
    for epoch in range(1,settings['nn_epochs']+1):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(et,tx,tm),ty)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if epoch == 1 and architecture == 'architecture_a':
            probe = model.verify_gradients(et,tx,tm)
        optimizer.step(); model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(ev,vx,vm),vy))
        assert np.isfinite(val_loss)
        history.append({'epoch':epoch,'train_huber':float(loss.detach()),'val_huber':val_loss})
        if val_loss<best-settings['min_delta']:
            best,epoch_best,wait = val_loss,epoch,0
            state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
        if wait>=settings['nn_patience']:
            break
    model.load_state_dict(state); model.eval()
    payload = {'state_dict':state,'prep':prep_state(prep),'eeg_mean':scaler.mean_.tolist(),
               'eeg_scale':scaler.scale_.tolist(),'target_mean':mean,'target_std':std,
               'settings':settings,'columns':columns,'seed':seed,'selected_epoch':epoch_best,
               'seconds':time.perf_counter()-start,'clinical_pretraining':False,'architecture':architecture,'forward_gradient_probe':probe}
    return payload,history


def predict_concat(payload,eeg,clinical):
    from codl.clinical.ft_clinical import restore_prep
    prep = restore_prep(payload['prep']); x,m = tensors(prep,clinical)
    e = torch.tensor((eeg-np.array(payload['eeg_mean']))/np.array(payload['eeg_scale']),dtype=torch.float32)
    s = payload['settings']; _,Net = _modules()
    model = Net(256,prep.cat,s['nn_token_dim'],s['nn_hidden'],s['nn_dropout']) if payload.get('architecture','raw_concat') == 'raw_concat' else build_architecture_a(prep.cat)
    model.load_state_dict(payload['state_dict']); model.eval()
    with torch.no_grad():
        result = model(e,x,m).numpy()*payload['target_std']+payload['target_mean']
    assert np.isfinite(result).all()
    return result


def build_architecture_a(cat_mask):
    from codl.fusion.ft_architecture_a import FTArchitectureA
    return FTArchitectureA(cat_mask)
