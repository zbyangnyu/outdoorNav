"""Training curves (loss + val metrics) for each run dir with log.jsonl. usage: python plot_curves.py run_dir... -> curves.png"""
import json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
runs = sys.argv[1:] or ['run_kd', 'run']
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for r in runs:
    L = [json.loads(l) for l in open(r + '/log.jsonl')]
    tr = [x for x in L if 'loss' in x]; va = [x['val'] for x in L if 'val' in x]
    k = 20; sm = [sum(x['loss'] for x in tr[max(0, i - k):i + 1]) / len(tr[max(0, i - k):i + 1]) for i in range(len(tr))]
    ax[0].plot([x['it'] for x in tr], sm, label=r)
    it = [v['it'] for v in va]
    for j, n in ((1, 'paved'), (2, 'grass'), (3, 'stairs')): ax[1].plot(it, [v['iou'][j] for v in va], label=f'{r} {n} IoU')
    ax[2].plot(it, [v['stairs_as_road'] for v in va], label=f'{r} stairs->road'); ax[2].plot(it, [v['grass_as_road'] for v in va], '--', label=f'{r} grass->road')
ax[0].set_title('train loss (smoothed)'); ax[1].set_title('val IoU'); ax[2].set_title('val confusion (lower better)')
for a in ax: a.set_xlabel('iteration'); a.legend(fontsize=7); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig('curves.png', dpi=110)
