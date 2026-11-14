"""Average weights of several checkpoints (SWA). usage: python swa.py OUT CK1 CK2 ..."""
import sys, torch
from transformers import SegformerForSemanticSegmentation
ms = [SegformerForSemanticSegmentation.from_pretrained(c) for c in sys.argv[2:]]
sd = {k: sum(m.state_dict()[k].float() for m in ms) / len(ms) for k in ms[0].state_dict()}
ms[0].load_state_dict(sd); ms[0].save_pretrained(sys.argv[1]); print('averaged', len(ms))
