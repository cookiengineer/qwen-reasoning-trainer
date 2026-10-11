"""Independent NumPy reference for the Qwen3.8 tiny-model forward pass.

Used by TestReferenceNumpy (guarded by QWEN38_VERIFY_REF=1) to cross-check the
Go implementation. Tensors are stored in GGML/Fortran order: element
(i0,i1,...) is at flat index i0 + i1*d0 + ..., i.e. numpy reshape(order="F").
"""

import json
import sys

import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def T(dump, name):
    t = dump["tensors"][name]
    return np.array(t["data"], dtype=np.float32).reshape(t["dims"], order="F")


def rms(x, w, eps):
    ss = np.mean(x.astype(np.float64) ** 2, axis=0, keepdims=True)
    shape = (-1,) + (1,) * (x.ndim - 1)
    return (x.astype(np.float64) / np.sqrt(ss + eps)).astype(np.float32) * w.reshape(shape)


def mm(w, x):
    return (w.T @ x).astype(np.float32)


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def softplus(x):
    return np.log1p(np.exp(x)).astype(np.float32)


def rope(x, positions, theta, nrot):
    out = x.copy()
    half = nrot // 2
    for i in range(half):
        freq = theta ** (-2.0 * i / nrot)
        c = np.cos(positions * freq).astype(np.float32)
        s = np.sin(positions * freq).astype(np.float32)
        x0 = x[i].copy()
        x1 = x[i + half].copy()
        out[i] = x0 * c - x1 * s
        out[i + half] = x0 * s + x1 * c
    return out


def attention(q, k, v, nhead, nkv, scale, causal):
    hd, _, tq = q.shape
    tk = k.shape[2]
    group = nhead // nkv
    out = np.zeros((hd, nhead, tq), dtype=np.float32)
    for h in range(nhead):
        kv = h // group
        scores = (q[:, h, :].T @ k[:, kv, :]).astype(np.float64) * scale
        if causal and tq == tk:
            scores = np.where(np.triu(np.ones((tq, tk), bool), 1), -np.inf, scores)
        scores -= scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        probs = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        out[:, h, :] = v[:, kv, :] @ probs.T
    return out


def l2norm(x, eps=1e-6):
    ss = np.sum(x.astype(np.float64) ** 2, axis=0, keepdims=True)
    return (x.astype(np.float64) / np.sqrt(ss + eps)).astype(np.float32)


def gdn(q, k, v, gate, beta, state, sv, nh, nt):
    out = np.zeros((sv, nh, nt), dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(sv))
    for h in range(nh):
        M = state[:, :, h].astype(np.float64)
        for t in range(nt):
            M *= np.exp(np.float64(gate[h, t]))
            kk = k[:, h, t].astype(np.float64)
            vv = v[:, h, t].astype(np.float64)
            delta = (vv - M @ kk) * np.float64(beta[h, t])
            M += np.outer(delta, kk)
            out[:, h, t] = (M @ q[:, h, t].astype(np.float64)) * scale
        state[:, :, h] = M
    return out, state


def run(dump):
    cfg = dump["config"]
    tokens = np.array(dump["tokens"], dtype=np.int64)
    eps = np.float32(cfg["RmsEps"])
    hd = cfg["HeadDim"]
    nh = cfg["NHead"]
    nkv = cfg["NHeadKV"]
    nrot = cfg["NRot"]
    theta = cfg["RopeTheta"]
    tt = len(tokens)

    x = T(dump, "token_embd")[:, tokens].astype(np.float32)
    positions = np.arange(tt, dtype=np.float32)
    hs = [x.copy()]

    for il in range(cfg["TrunkLayers"]):
        p = f"l{il}."
        hn = rms(x, T(dump, p + "attn_norm"), eps)
        if cfg["IsRecurrent"][il]:
            qkv = mm(T(dump, p + "attn_qkv"), hn)
            z = mm(T(dump, p + "attn_gate"), hn)
            beta = sigmoid(mm(T(dump, p + "ssm_beta"), hn))
            alpha = mm(T(dump, p + "ssm_alpha"), hn)
            dt = T(dump, p + "ssm_dt")
            sa = T(dump, p + "ssm_a")
            gate = np.zeros((cfg["DtRank"], tt), dtype=np.float32)
            for hh in range(cfg["DtRank"]):
                gate[hh] = softplus(alpha[hh] + dt[hh]) * sa[hh]

            conv_dim = cfg["ConvDim"]
            dconv = cfg["DConv"]
            conv_in = np.zeros((dconv - 1 + tt, conv_dim), dtype=np.float32)
            conv_in[dconv - 1 :, :] = qkv.T
            ker = T(dump, p + "ssm_conv1d")
            conv_out = np.zeros((conv_dim, tt), dtype=np.float32)
            for t in range(tt):
                conv_out[:, t] = np.sum(conv_in[t : t + dconv, :] * ker, axis=0)
            conv_out = silu(conv_out)

            head_k = cfg["DState"]
            head_v = cfg["HeadVDim"]
            nk = cfg["GroupCount"]
            nv = cfg["DtRank"]
            key_dim = cfg["KeyDim"]
            q = conv_out[:key_dim, :].reshape((head_k, nk, tt), order="F")
            k = conv_out[key_dim : 2 * key_dim, :].reshape((head_k, nk, tt), order="F")
            v = conv_out[2 * key_dim :, :].reshape((head_v, nv, tt), order="F")
            q = l2norm(q)
            k = l2norm(k)
            qr = np.concatenate([q] * (nv // nk), axis=1)
            kr = np.concatenate([k] * (nv // nk), axis=1)
            st = np.zeros((head_v, head_v, nv), dtype=np.float32)
            out, _ = gdn(qr, kr, v, gate, beta, st, head_v, nv, tt)
            norm = rms(out, T(dump, p + "ssm_norm"), eps)
            zr = z.reshape((head_v, nv, tt), order="F")
            flat = (norm * silu(zr)).reshape((cfg["ValueDim"], tt), order="F")
            attn_out = mm(T(dump, p + "ssm_out"), flat)
        else:
            qg = mm(T(dump, p + "attn_q"), hn)
            kcur = mm(T(dump, p + "attn_k"), hn)
            vcur = mm(T(dump, p + "attn_v"), hn)
            q4 = qg.reshape((hd, 2, nh, tt), order="F")
            q = rms(q4[:, 0], T(dump, p + "attn_q_norm"), eps)
            gate = q4[:, 1]
            kn = rms(kcur.reshape((hd, nkv, tt), order="F"), T(dump, p + "attn_k_norm"), eps)
            vn = vcur.reshape((hd, nkv, tt), order="F")
            q = rope(q, positions, theta, nrot)
            kn = rope(kn, positions, theta, nrot)
            attn = attention(q, kn, vn, nh, nkv, np.float32(1.0 / np.sqrt(hd)), True)
            attn = attn.reshape((nh * hd, tt), order="F") * sigmoid(
                gate.reshape((nh * hd, tt), order="F")
            )
            attn_out = mm(T(dump, p + "attn_output"), attn)
        x = (x + attn_out).astype(np.float32)

        hn2 = rms(x, T(dump, p + "post_attn_norm"), eps)
        fg = mm(T(dump, p + "ffn_gate"), hn2)
        fu = mm(T(dump, p + "ffn_up"), hn2)
        x = (x + mm(T(dump, p + "ffn_down"), silu(fg) * fu)).astype(np.float32)
        hs.append(x.copy())

    x = rms(x, T(dump, "output_norm"), eps)
    return mm(T(dump, "output"), x), hs


def main():
    dump = load(sys.argv[1])
    logits, hs = run(dump)
    go = np.array(dump["go_logits"], dtype=np.float32).reshape(
        (dump["config"]["NVocab"], len(dump["tokens"])), order="F"
    )
    rel = np.abs(logits - go) / np.maximum(1.0, np.abs(go))
    print(f"logits max_rel_diff={rel.max():.3e}")
    gh = dump.get("go_hidden", [])
    for i, h in enumerate(hs):
        if i >= len(gh):
            break
        a = np.array(gh[i], dtype=np.float32)
        b = h.reshape(-1, order="F")
        d = np.abs(a - b) / np.maximum(1.0, np.abs(b))
        print(f"  hidden[{i}] max_rel_diff={d.max():.3e}")
    ok = rel.max() < 2e-3
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
