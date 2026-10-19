"""Initial SE graph diagnostic program; observations use only TRAIN batches.

These diagnostics are not dev ranking metrics and never select the common winner.
SE may evolve this diagnostic program through its separate native research path.
"""
import json
import torch
from recbole.data.interaction import Interaction


class DiagnosisProbe:
    def __init__(self,model,device): self.model,self.device=model,device

    def run_full_diagnosis(self,train_loader):
        margins,stds,finite,total=[],[],0,0
        self.model.eval()
        with torch.no_grad():
            for index,batch in enumerate(train_loader):
                if index==8: break
                batch=batch[:32].to(self.device)
                positive=self.model.predict(batch).reshape(-1)
                negative=Interaction({key:value.clone() for key,value in batch.interaction.items()})
                negative[self.model.ITEM_ID]=batch[self.model.NEG_ITEM_ID]
                neg_scores=self.model.predict(negative).reshape(-1)
                scores=self.model.full_sort_predict(batch).reshape(len(batch),-1)[:,1:]
                finite+=int(torch.isfinite(scores).sum().item());total+=scores.numel()
                margins.extend((positive-neg_scores).cpu().tolist())
                stds.extend(scores.std(dim=1,unbiased=False).cpu().tolist())
        if not margins: raise ValueError('No train batches for actual graph diagnosis')
        parameters=list(self.model.parameters())
        values=dict(train_pair_margin_mean=sum(margins)/len(margins),
            train_positive_above_uniform_negative=sum(x>0 for x in margins)/len(margins),
            catalog_score_std_mean=sum(stds)/len(stds),finite_score_fraction=finite/total,
            parameter_l2_norm=float(sum(p.detach().square().sum().item() for p in parameters)**.5),
            observed_train_pairs=len(margins))
        return dict(metrics=values,metric_definitions={
            'train_pair_margin_mean':'Train positive dot score minus a uniform train-negative score; not held-out accuracy.',
            'train_positive_above_uniform_negative':'Fraction of observed TRAIN pairs with positive margin.',
            'catalog_score_std_mean':'Per-user full-catalog score spread, excluding padding; near zero can indicate collapse.',
            'finite_score_fraction':'Fraction of finite raw prediction scores on observed TRAIN users.',
            'parameter_l2_norm':'Current selected-checkpoint parameter norm; diagnostic, not a tuned target.',
            'observed_train_pairs':'Actual pairs inspected, at most8 batches of32; no optimizer step.'})


def diagnosis_interpreter_prompt(diagnosis):
    return ('Interpret these measured TRAIN-only graph diagnostics. Distinguish evidence from hypotheses. '
            'Do not describe them as validation or test performance. Explain instability, collapse or '
            'insufficient learned separation only where supported.\n'+json.dumps(diagnosis,allow_nan=False))
