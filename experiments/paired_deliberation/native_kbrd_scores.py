"""NumPy-only, label-free access to exported KBRD native movie logits."""
import json
from pathlib import Path
import numpy as np

class PrecomputedKBRD:
    def __init__(self,root):
        self.root=Path(root);self.ids=json.loads((self.root/'valid.ids.json').read_text());self.items=json.loads((self.root/'native_item_ids.json').read_text())
        self.row_index={c:i for i,c in enumerate(self.ids)};self.item_index={str(m):j for j,m in enumerate(self.items)}
        self.matrix=np.load(self.root/'valid.native_logits.npy',mmap_mode='r');self.meta=json.loads((self.root/'serving_metadata.json').read_text())
        if self.matrix.shape!=(len(self.ids),len(self.items)):raise ValueError('native logit shape mismatch')
    def score_pool(self,case_id,candidate_ids):
        i=self.row_index[case_id];values=np.asarray(self.matrix[i]);rank=np.argsort(-values,kind='stable');inverse=np.empty(len(rank),dtype=np.int32);inverse[rank]=np.arange(1,len(rank)+1)
        mean=float(values.mean());std=max(float(values.std()),1e-8)
        rows=[]
        for candidate in candidate_ids:
            j=self.item_index[str(candidate)];rows.append([str(candidate),float(values[j]),float((values[j]-mean)/std),int(inverse[j])])
        return {'case_id':case_id,'source_id':'model:kbrd:'+self.meta['checkpoint_sha256'][:12]+':'+case_id,
            'columns':['movie_id','logit','native_z_score','native_rank'],'rows':rows,
            'checkpoint_epoch':self.meta['selected_epoch'],'selection_metric':self.meta['selection_metric'],
            'trained_split':'train','current_user_fact_authority':False,'semantic_truth_certified':False}
