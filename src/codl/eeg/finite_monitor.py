"""Read-only activation/gradient hooks; no RNG or arithmetic-path changes."""
import torch


class FiniteMonitor:
    def __init__(self, model):
        self.stats = {}
        self.handles = []
        for name,module in model.named_modules():
            if name and not list(module.children()):
                self.handles.append(module.register_forward_hook(self.forward_hook(name)))
        for name,p in model.named_parameters():
            self.handles.append(p.register_hook(self.gradient_hook('parameter.'+name)))

    def record(self,key,value):
        value = value.detach()
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite '+key)
        stat = self.stats.setdefault(key,{'checks':0,'max_abs':0.})
        stat['checks'] += 1
        stat['max_abs'] = max(stat['max_abs'],float(value.abs().max()))

    def gradient_hook(self,name):
        def hook(gradient):
            self.record('gradient.'+name,gradient)
        return hook

    def forward_hook(self,name):
        def hook(module,args,output):
            self.record('activation.'+name,output)
            if output.requires_grad:
                output.register_hook(self.gradient_hook('activation.'+name))
        return hook

    def report(self):
        assert all('activation.convs.'+str(i) in self.stats for i in range(9))
        assert all('gradient.activation.convs.'+str(i) in self.stats for i in range(9))
        return {'status':'passed','statistics':self.stats,'nonfinite_count':0}

    def close(self):
        for handle in self.handles:
            handle.remove()
