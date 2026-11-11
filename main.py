"""Per-item candidate models blended and calibrated out of fold on the
visible rows, inside predict().

Nothing is learned offline. The visible rows (respondents whose PREDICT block
is filled in) are the only real data this code ever touches, so every choice
-- which candidate to trust, how much, and how sharp to be -- is made on
K-fold out-of-fold (OOF) predictions of those rows, scored with the grader's
own transform (renormalise, mix with a flat 1e-3 floor, log).

Per item the candidates are
  marg  smoothed crowd marginal (the baseline that scores 0)
  dir   Dirichlet back-off down a ladder of GIVEN strata (ladder picked by
        mutual information, depth and alpha picked out of fold)
  lr    multinomial logistic regression on one-hot GIVEN (C picked OOF)
  hgb   HistGradientBoostingClassifier on categorical GIVEN codes, only when
        the running clock says the instrument can afford it

and they are mixed with convex weights fitted on the OOF rows, shrunk a little
towards the default (the marginal), then sharpened or flattened by one
temperature per item, also chosen OOF.

Gated items: when the gate's parent is GIVEN and the visible rows obey the
skip rule exactly, the sentinel slot is deterministic. Otherwise two or three
whole-vector candidates are built -- the gate chain (parent's own predicted
distribution summed over `observed_if`) times P(answer | asked), a learned
P(asked | GIVEN) times P(answer | asked), and a direct k+1-class model with
NA_GATED as an ordinary class -- and blended OOF like everything else.

Time: one clock from module import, a per-instrument budget proportional to
its share of the known total work (visible rows x PREDICT items over the
three instruments), degrading candidate by candidate as the deadline nears
and ending at the marginal.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

_T_IMPORT = time.time()
FLOOR = 1e-3
SEED = 0
N_FOLDS = 3
MIN_ROWS = 30                 # below this: smoothed marginal only

# total work = sum over the three instruments of (visible rows x PREDICT items),
# from data/*.json: phase 1 shows TRAIN, phase 2 shows TRAIN+DEV
TOTAL_WORK = {1: 2336867.0, 2: 2526516.0}
TOTAL_BUDGET = {1: 900.0 * 0.6, 2: 3600.0 * 0.6}

DIR_ALPHAS = (1.0, 4.0, 16.0, 64.0)
DIR_DEPTHS = (1, 2, 3)
LR_CS = (0.1, 1.0)
HGB_ITER = 50
TEMPS = np.round(np.arange(0.6, 1.61, 0.1), 2)


# ----------------------------------------------------------------- scoring --
def _mix(p):
    k = p.shape[1]
    s = p.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return FLOOR + (1.0 - k * FLOOR) * (p / s)


def _logloss(p, y):
    q = _mix(np.clip(p, 0.0, None))
    return -float(np.mean(np.log(q[np.arange(len(y)), y])))


def _normalise(p):
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return p / s


def _marginal(y, k):
    counts = np.bincount(y, minlength=k).astype(float) + 0.5
    return counts / counts.sum()


# ----------------------------------------------------------------- blending --
def _blend_weights(oofs, y, default, n_iter=300, eta=0.5):
    """Convex weights minimising the mixed log loss of sum_m w_m P_m, by
    exponentiated gradient, then shrunk towards the default candidate."""
    m = len(oofs)
    if m == 1:
        return np.ones(1)
    k = oofs[0].shape[1]
    n = len(y)
    py = np.stack([_normalise(p)[np.arange(n), y] for p in oofs], axis=1)
    scale = 1.0 - k * FLOOR
    w = np.full(m, 1.0 / m)
    for _ in range(n_iter):
        denom = FLOOR + scale * (py @ w)
        grad = -np.mean(py * scale / denom[:, None], axis=0)
        w = w * np.exp(-eta * (grad - grad.mean()))
        w = w / w.sum()
    shrink = min(0.5, max(0.02, 100.0 / max(n, 1)))
    w = (1.0 - shrink) * w
    w[default] += shrink
    return w


def _temperature(p_oof, y):
    best_t, best = 1.0, np.inf
    base = _normalise(p_oof) + 1e-12
    for t in TEMPS:
        loss = _logloss(base ** t, y)
        if loss < best - 1e-9:
            best, best_t = loss, t
    return 1.0 + 0.75 * (best_t - 1.0)


def _apply_temperature(p, t):
    return _normalise((_normalise(p) + 1e-12) ** t)


# ------------------------------------------------------------- candidates ---
def _mutual_information(x, y, nx, ny):
    both = (x >= 0) & (y >= 0)
    x, y = x[both], y[both]
    if x.size < 8:
        return 0.0
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny) / x.size
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def _backoff(codes_tr, y_tr, codes_te, ladder, cards, alpha, k, base):
    """Dirichlet back-off: start at the marginal, pull towards the counts of
    each finer stratum in turn; `alpha` is what the coarser level is worth in
    people. Unknown GIVEN codes (-1) form their own level."""
    run = np.tile(base, (codes_te.shape[0], 1))
    key_tr = np.zeros(codes_tr.shape[0], dtype=np.int64)
    key_te = np.zeros(codes_te.shape[0], dtype=np.int64)
    size = 1
    for col in ladder:
        c = cards[col] + 1
        key_tr = key_tr * c + (codes_tr[:, col] + 1)
        key_te = key_te * c + (codes_te[:, col] + 1)
        size *= c
        table = np.bincount(key_tr * k + y_tr, minlength=size * k).astype(float)
        counts = table.reshape(size, k)[key_te]
        run = (counts + alpha * run) / (counts.sum(axis=1, keepdims=True) + alpha)
    return run


def _proba_full(model, X, k):
    out = np.zeros((X.shape[0], k))
    out[:, model.classes_] = model.predict_proba(X)
    return out


def _fit_lr(X, y, C):
    model = LogisticRegression(C=C, max_iter=200, tol=1e-3)
    model.fit(X, y)
    return model


def _fit_hgb(Xc, y, k):
    model = HistGradientBoostingClassifier(
        max_iter=HGB_ITER, learning_rate=0.1, max_leaf_nodes=8, max_depth=4,
        min_samples_leaf=30, l2_regularization=1.0,
        categorical_features=[True] * Xc.shape[1],
        early_stopping=False, random_state=SEED)
    model.fit(Xc, y)
    return model


class _Engine:
    """Everything an instrument's items share: design matrices, folds, clock."""

    def __init__(self, Xv1, Xh1, Cv, Ch, cards, folds, deadline, n_items):
        self.Xv1, self.Xh1 = Xv1, Xh1          # one-hot, visible / held
        self.Cv, self.Ch = Cv, Ch              # integer codes, visible / held
        self.cards = cards
        self.folds = folds
        self.deadline = deadline
        self.n_items = n_items
        self.hgb_unit = None                   # seconds per (row x class) fit
        self.hgb_ok = Cv.shape[1] > 0 and (len(cards) == 0 or max(cards) < 250)
        self.item_idx = 0
        self.item_start = time.time()
        self.fair = deadline - self.item_start
        self.hgb_fits = 0

    # -- clock -------------------------------------------------------------
    def begin_item(self, idx):
        self.item_idx = idx
        self.item_start = time.time()
        left = max(self.deadline - self.item_start, 0.0)
        self.fair = left / max(self.n_items - idx, 1)

    def time_left(self):
        return self.deadline - time.time()

    def level(self):
        """3: all candidates, 2: no hgb, 1: marg+dir only, 0: marg only."""
        left = self.time_left()
        if left <= 0:
            return 0
        if left < 0.08 * max(self.fair * (self.n_items - self.item_idx), 1.0):
            return 1
        return 3

    def afford_hgb(self, n, k):
        if not self.hgb_ok or self.level() < 3:
            return False
        if self.hgb_unit is None:
            est = 0.0                           # probe: let the first one run
        else:
            keff = 1 if k == 2 else k
            est = N_FOLDS * self.hgb_unit * n * keff
        spent = time.time() - self.item_start
        return spent + est <= 1.3 * self.fair and est < 0.5 * self.time_left()

    # -- one sub-problem ---------------------------------------------------
    def fit(self, rows, y, k, want_hgb=True):
        """Blend of candidates for rows `rows` (indices into the visible set)
        with labels y in [0, k). Returns (oof (len(rows), k), held (n_held, k),
        info dict)."""
        n = len(rows)
        marg = _marginal(y, k)
        info = {"cands": ["marg"], "w": [1.0], "t": 1.0}
        if n < MIN_ROWS or len(np.unique(y)) < 2 or self.level() == 0:
            return (np.tile(marg, (n, 1)), np.tile(marg, (self.Xh1.shape[0], 1)),
                    info)
        folds = self.folds[rows]
        Cv, Xv = self.Cv[rows], self.Xv1[rows]
        splits = [(folds != f, folds == f) for f in range(N_FOLDS)]
        splits = [(tr, te) for tr, te in splits if te.sum() > 0 and tr.sum() >= 10]
        oofs, helds, names = [], [], []

        # marg
        oof = np.zeros((n, k))
        for tr, te in splits:
            oof[te] = _marginal(y[tr], k)
        oofs.append(oof); helds.append(np.tile(marg, (self.Xh1.shape[0], 1)))
        names.append("marg")

        # dir
        try:
            if Cv.shape[1] > 0:
                best, best_oof, best_cfg = np.inf, None, None
                ladders = []
                for tr, te in splits:
                    mi = [_mutual_information(Cv[tr, j], y[tr], self.cards[j], k)
                          for j in range(Cv.shape[1])]
                    ladders.append(list(np.argsort(mi)[::-1]))
                for depth in DIR_DEPTHS:
                    if depth > Cv.shape[1]:
                        break
                    for alpha in DIR_ALPHAS:
                        oof = np.zeros((n, k))
                        for (tr, te), lad in zip(splits, ladders):
                            oof[te] = _backoff(Cv[tr], y[tr], Cv[te], lad[:depth],
                                               self.cards, alpha, k,
                                               _marginal(y[tr], k))
                        loss = _logloss(oof, y)
                        if loss < best - 1e-9:
                            best, best_oof, best_cfg = loss, oof, (depth, alpha)
                mi = [_mutual_information(Cv[:, j], y, self.cards[j], k)
                      for j in range(Cv.shape[1])]
                lad = list(np.argsort(mi)[::-1])[:best_cfg[0]]
                held = _backoff(Cv, y, self.Ch, lad, self.cards, best_cfg[1], k, marg)
                oofs.append(best_oof); helds.append(held); names.append("dir")
                info["dir"] = best_cfg
        except Exception:
            pass

        # lr
        if self.level() >= 2 and Xv.shape[1] > 0:
            try:
                best, best_oof, best_models = np.inf, None, None
                for C in LR_CS:
                    oof = np.zeros((n, k))
                    models = []
                    for tr, te in splits:
                        if len(np.unique(y[tr])) < 2:
                            oof[te] = _marginal(y[tr], k); models.append(None)
                            continue
                        m = _fit_lr(Xv[tr], y[tr], C)
                        oof[te] = _proba_full(m, Xv[te], k)
                        models.append(m)
                    loss = _logloss(oof, y)
                    if loss < best - 1e-9:
                        best, best_oof, best_models, best_C = loss, oof, models, C
                    if self.time_left() <= 0:
                        break
                preds = [_proba_full(m, self.Xh1, k) for m in best_models if m is not None]
                if preds:
                    oofs.append(best_oof); helds.append(np.mean(preds, axis=0))
                    names.append("lr"); info["lr_C"] = best_C
            except Exception:
                pass

        # hgb
        if want_hgb and self.afford_hgb(n, k):
            try:
                t0 = time.time()
                oof = np.zeros((n, k))
                preds = []
                for tr, te in splits:
                    if len(np.unique(y[tr])) < 2:
                        oof[te] = _marginal(y[tr], k)
                        continue
                    m = _fit_hgb(Cv[tr].astype(np.float64), y[tr], k)
                    oof[te] = _proba_full(m, Cv[te].astype(np.float64), k)
                    preds.append(_proba_full(m, self.Ch.astype(np.float64), k))
                    if self.hgb_unit is None:
                        keff = 1 if k == 2 else k
                        self.hgb_unit = max(time.time() - t0, 1e-3) / (tr.sum() * keff)
                        if not self.afford_hgb(n, k):
                            raise RuntimeError("hgb too slow for this budget")
                if preds:
                    oofs.append(oof); helds.append(np.mean(preds, axis=0))
                    names.append("hgb"); self.hgb_fits += 1
            except Exception:
                pass

        w = _blend_weights(oofs, y, default=0)
        oof = sum(wi * _normalise(p) for wi, p in zip(w, oofs))
        held = sum(wi * _normalise(p) for wi, p in zip(w, helds))
        t = _temperature(oof, y)
        info.update({"cands": names, "w": [round(float(x), 3) for x in w],
                     "t": round(float(t), 3)})
        return _apply_temperature(oof, t), _apply_temperature(held, t), info


# ------------------------------------------------------------------ predict --
def _codes(frame, items, opts):
    cols = []
    for n in items:
        cols.append(pd.Categorical(frame[n], categories=opts[n]).codes.astype(np.int64))
    if not cols:
        return np.zeros((len(frame), 0), dtype=np.int64)
    return np.stack(cols, axis=1)


def _onehot(codes, cards):
    n = codes.shape[0]
    if codes.shape[1] == 0:
        return np.ones((n, 1), dtype=np.float32)
    blocks = []
    for j, c in enumerate(cards):
        block = np.zeros((n, c), dtype=np.float32)
        ok = codes[:, j] >= 0
        block[np.where(ok)[0], codes[ok, j]] = 1.0
        blocks.append(block)
    return np.hstack(blocks)


def _gate_depth(recs, name):
    d, seen = 0, set()
    while recs.get(name, {}).get("gate") and name not in seen:
        seen.add(name)
        name = recs[name]["gate"]["parent"]
        d += 1
    return d


def predict(frame, schema):
    t0 = time.time()
    gated_value = schema["gated_value"]
    recs = schema["items"]
    items = [n for n, r in recs.items() if r["class"] in ("GIVEN", "PREDICT")]
    given = [n for n in items if recs[n]["class"] == "GIVEN"]
    scored = [n for n in items if recs[n]["class"] == "PREDICT"]
    opts = {n: list(recs[n]["values"]) + ([gated_value] if recs[n].get("gate") else [])
            for n in items}

    held = frame[scored].isna().any(axis=1).to_numpy()
    vis = ~held
    n_vis, n_held = int(vis.sum()), int(held.sum())

    # ---- budget --------------------------------------------------------
    split = schema.get("split") or {}
    phase = 2 if n_vis > split.get("n_train", n_vis) else 1
    share = min(1.0, n_vis * len(scored) / TOTAL_WORK[phase])
    planned = share * TOTAL_BUDGET[phase]
    remaining = TOTAL_BUDGET[phase] - (t0 - _T_IMPORT)
    budget = max(5.0, min(planned, 0.9 * remaining))
    deadline = t0 + budget

    # ---- design --------------------------------------------------------
    cards = [len(opts[g]) for g in given]
    C_all = _codes(frame, given, opts)
    X_all = _onehot(C_all, cards)
    Cv, Ch = C_all[vis], C_all[held]
    Xv, Xh = X_all[vis], X_all[held]
    rng = np.random.default_rng(SEED)
    folds = rng.integers(0, N_FOLDS, size=n_vis)
    eng = _Engine(Xv, Xh, Cv, Ch, cards, folds, deadline, len(scored))
    all_rows = np.arange(n_vis)

    y_vis = {}
    for n in scored:
        y_vis[n] = pd.Categorical(frame[n], categories=opts[n]).codes.astype(np.int64)[vis]
    given_vis = {g: frame[g].to_numpy(dtype=object)[vis] for g in given}
    given_held = {g: frame[g].to_numpy(dtype=object)[held] for g in given}

    order = sorted(scored, key=lambda n: _gate_depth(recs, n))
    full_oof, full_held, diag = {}, {}, {}

    for idx, item in enumerate(order):
        eng.begin_item(idx)
        k = len(opts[item])
        y = y_vis[item]
        marg = _marginal(np.clip(y, 0, k - 1), k)
        oof_item = np.tile(marg, (n_vis, 1))
        held_item = np.tile(marg, (n_held, 1))
        info = {"k": k}
        try:
            gate = recs[item].get("gate")
            if not gate:
                oof_item, held_item, inf = eng.fit(all_rows, y, k)
                info.update(inf)
            else:
                s = k - 1
                asked_v = y != s
                parent = gate["parent"]
                observed_if = list(gate["observed_if"])
                asked_rows = np.where(asked_v)[0]
                # P(answer | asked)
                if len(asked_rows) >= 1:
                    cond_oof_a, cond_held, inf_c = eng.fit(asked_rows, y[asked_rows], s)
                else:
                    cond_oof_a = np.zeros((0, s)); cond_held = np.full((n_held, s), 1.0 / s)
                    inf_c = {}
                cond_marg = (_marginal(y[asked_rows], s) if len(asked_rows)
                             else np.full(s, 1.0 / s))
                cond_oof = np.tile(cond_marg, (n_vis, 1))
                cond_oof[asked_rows] = cond_oof_a
                info["cond"] = inf_c

                if recs.get(parent, {}).get("class") == "GIVEN":
                    ind_v = pd.Series(given_vis[parent]).isin(observed_if).to_numpy()
                    ind_h = pd.Series(given_held[parent]).isin(observed_if).to_numpy()
                    agree = float(np.mean(ind_v == asked_v))
                    info["gate_agree"] = agree
                    if agree == 1.0:
                        pa_oof = ind_v.astype(float); pa_held = ind_h.astype(float)
                        oof_item = np.concatenate([pa_oof[:, None] * cond_oof,
                                                  1.0 - pa_oof[:, None]], axis=1)
                        held_item = np.concatenate([pa_held[:, None] * cond_held,
                                                   1.0 - pa_held[:, None]], axis=1)
                        info["gate"] = "deterministic"
                    else:
                        pa_oof2, pa_held2, inf_a = eng.fit(all_rows, asked_v.astype(np.int64),
                                                           2, want_hgb=False)
                        pa_oof, pa_held = pa_oof2[:, 1], pa_held2[:, 1]
                        oof_item = np.concatenate([pa_oof[:, None] * cond_oof,
                                                  1.0 - pa_oof[:, None]], axis=1)
                        held_item = np.concatenate([pa_held[:, None] * cond_held,
                                                   1.0 - pa_held[:, None]], axis=1)
                        info["gate"] = "learned"; info["asked"] = inf_a
                else:
                    cands_oof, cands_held, names = [], [], []
                    # (A) gate chain from the parent's predicted distribution
                    if parent in full_oof:
                        pvals = opts[parent]
                        opens = [pvals.index(v) for v in observed_if if v in pvals]
                        yp = y_vis[parent]
                        chain_v = np.isin(yp, opens)
                        agree = float(np.mean(chain_v == asked_v))
                        info["gate_agree"] = agree
                        pa = np.clip(full_oof[parent][:, opens].sum(axis=1), 0, 1)
                        ph = np.clip(full_held[parent][:, opens].sum(axis=1), 0, 1)
                        cands_oof.append(np.concatenate([pa[:, None] * cond_oof,
                                                        1.0 - pa[:, None]], axis=1))
                        cands_held.append(np.concatenate([ph[:, None] * cond_held,
                                                         1.0 - ph[:, None]], axis=1))
                        names.append("chain")
                    # (B) learned P(asked | GIVEN)
                    try:
                        pa2, ph2, inf_a = eng.fit(all_rows, asked_v.astype(np.int64),
                                                  2, want_hgb=False)
                        pa, ph = pa2[:, 1], ph2[:, 1]
                        cands_oof.append(np.concatenate([pa[:, None] * cond_oof,
                                                        1.0 - pa[:, None]], axis=1))
                        cands_held.append(np.concatenate([ph[:, None] * cond_held,
                                                         1.0 - ph[:, None]], axis=1))
                        names.append("asked"); info["asked"] = inf_a
                    except Exception:
                        pass
                    # (C) direct, sentinel as a class
                    try:
                        d_oof, d_held, inf_d = eng.fit(all_rows, y, k)
                        cands_oof.append(d_oof); cands_held.append(d_held)
                        names.append("direct"); info["direct"] = inf_d
                    except Exception:
                        pass
                    if cands_oof:
                        default = 0
                        w = _blend_weights(cands_oof, y, default=default)
                        oof_item = sum(wi * _normalise(p) for wi, p in zip(w, cands_oof))
                        held_item = sum(wi * _normalise(p) for wi, p in zip(w, cands_held))
                        t = _temperature(oof_item, y)
                        oof_item = _apply_temperature(oof_item, t)
                        held_item = _apply_temperature(held_item, t)
                        info["gate"] = dict(zip(names, [round(float(x), 3) for x in w]))
                        info["gate_t"] = round(float(t), 3)
        except Exception as exc:          # anything at all: marginal
            info["error"] = repr(exc)[:200]
            oof_item = np.tile(marg, (n_vis, 1))
            held_item = np.tile(marg, (n_held, 1))

        # hygiene before anyone downstream (gate chain) reads it
        held_item = np.asarray(held_item, dtype=float)
        if (held_item.shape != (n_held, k) or not np.isfinite(held_item).all()
                or (held_item.sum(axis=1) <= 0).any()):
            held_item = np.tile(marg, (n_held, 1))
            oof_item = np.tile(marg, (n_vis, 1))
            info["error"] = info.get("error", "") + "|shape"
        oof_item = _normalise(np.nan_to_num(np.asarray(oof_item, dtype=float)))
        held_item = _normalise(held_item)
        full_oof[item] = oof_item
        full_held[item] = held_item
        info["sec"] = round(time.time() - eng.item_start, 3)
        diag[item] = info

    predict.diag = {"items": diag, "budget": round(budget, 1), "phase": phase,
                    "sec": round(time.time() - t0, 1), "hgb_fits": eng.hgb_fits,
                    "n_vis": n_vis, "n_held": n_held}

    # ---- canonical order: rows top to bottom, items in schema key order ---
    held_values = frame.loc[held, items].to_numpy(dtype=object)
    item_index = {n: j for j, n in enumerate(items)}
    vectors = []
    for r in range(held_values.shape[0]):
        for n in items:
            if pd.isna(held_values[r, item_index[n]]):
                v = np.clip(full_held[n][r], 0.0, None)
                if v.shape[0] != len(opts[n]) or not np.isfinite(v).all() or v.sum() <= 0:
                    v = np.ones(len(opts[n]))
                vectors.append([float(x) for x in v])
    return vectors
