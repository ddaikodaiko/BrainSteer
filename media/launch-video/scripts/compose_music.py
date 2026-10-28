"""Pista original para el vídeo de BrainSteer: 120 BPM, La menor, 31 compases (62 s).

Arco de energía pensado para el montaje (1 compás = 2 s = 60 frames a 30 fps):
  0-6   intro (título, problema)          · pad que se abre + hats suaves
  6-10  subida (matemáticas de SKOP)      · bombo a medio tiempo, arpegio, riser
  10-18 DROP (Kᵀ G K, Gemma Scope)        · bombo 4x4, palmas, bajo, arpegio
  18-22 bajón (cabina: escena densa)      · pad + arpegio, sin bombo, riser final
  22-28 DROP final (seguridad, rigor)     · todo
  28-31 outro (CTA)                        · pad que se cierra
"""
import sys
import wave

import numpy as np

SR = 48_000
BPM = 120
BEAT = 60 / BPM
BAR = 4 * BEAT
BARS = 31
N = int(BARS * BAR * SR)
t = np.arange(N) / SR
rng = np.random.default_rng(7)

def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)

# Am - F - C - G (un acorde por compás)
CHORDS = [[57, 60, 64, 69], [53, 57, 60, 65], [55, 60, 64, 67], [55, 59, 62, 67]]
ROOTS = [45, 41, 48, 43]

def bar_range(a, b):
    return int(a * BAR * SR), int(b * BAR * SR)

def env_adsr(n, a, d, s, r, sr=SR):
    a, d, r = int(a * sr), int(d * sr), int(r * sr)
    sus = max(0, n - a - d - r)
    e = np.concatenate([np.linspace(0, 1, a, endpoint=False), np.linspace(1, s, d, endpoint=False),
                        np.full(sus, s), np.linspace(s, 0, r)])
    return e[:n] if len(e) >= n else np.pad(e, (0, n - len(e)))

def one_pole_lp(x, cutoff):
    """Paso bajo de un polo con cutoff (Hz) por muestra."""
    c = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    a = np.exp(-2 * np.pi * c / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a[i]) * x[i] + a[i] * acc
        y[i] = acc
    return y

def lp_block(x, cutoff_fn, block=256):
    """Filtro con cutoff variable por bloques (rápido y suficiente para un pad)."""
    y = np.empty_like(x)
    acc = 0.0
    for s in range(0, len(x), block):
        a = np.exp(-2 * np.pi * cutoff_fn(s / SR) / SR)
        seg = x[s:s + block]
        out = np.empty_like(seg)
        for i, v in enumerate(seg):
            acc = (1 - a) * v + a * acc
            out[i] = acc
        y[s:s + block] = out
    return y

def saw_band(f, tt, harmonics=10):
    out = np.zeros_like(tt)
    for k in range(1, harmonics + 1):
        if f * k > 9000:
            break
        out += np.sin(2 * np.pi * f * k * tt) / k
    return out

mixL = np.zeros(N)
mixR = np.zeros(N)

def add(sig, start, gain=1.0, pan=0.0):
    end = min(N, start + len(sig))
    seg = sig[: end - start] * gain
    mixL[start:end] += seg * np.sqrt(0.5 * (1 - pan))
    mixR[start:end] += seg * np.sqrt(0.5 * (1 + pan))

# ------------------------------------------------------------------ kick env (sidechain)
kick_times = []
for b in range(BARS):
    if 6 <= b < 10:                       # medio tiempo en la subida
        kick_times += [b * BAR, b * BAR + 2 * BEAT]
    if 10 <= b < 18 or 22 <= b < 28:      # 4x4 en los drops
        kick_times += [b * BAR + k * BEAT for k in range(4)]
    if b == 28:
        kick_times += [b * BAR]
side = np.ones(N)
for kt in kick_times:
    s = int(kt * SR); n = int(0.32 * SR)
    side[s:s + n] = np.minimum(side[s:s + n], 1 - 0.65 * np.exp(-np.arange(min(n, N - s)) / (0.09 * SR)))

# ------------------------------------------------------------------ pad
pad = np.zeros(N)
for b in range(BARS):
    chord = CHORDS[b % 4]
    s, e = bar_range(b, b + 1)
    n = e - s + int(0.6 * SR)
    tt = np.arange(n) / SR
    tone = np.zeros(n)
    for m in chord:
        for det in (-0.08, 0.08):        # dos voces desafinadas
            tone += saw_band(hz(m) * 2 ** (det / 12), tt, 8)
    tone *= env_adsr(n, 0.35, 0.4, 0.8, 0.6) / (len(chord) * 2)
    end = min(N, s + n)
    pad[s:end] += tone[: end - s]

def pad_cutoff(sec):
    b = sec / BAR
    if b < 6:  return 350 + 1500 * (b / 6)          # se abre en la intro
    if b < 10: return 1900
    if b < 18: return 2600
    if b < 22: return 1400
    if b < 28: return 2800
    return max(250, 2800 - 900 * (b - 28))         # se cierra en el outro
pad = lp_block(pad, pad_cutoff)
pad_level = np.interp(t, [0, 1.5, 12, 20, 36, 44, 56, 60, 62], [0, .55, .6, .45, .62, .45, .55, .3, 0])
pad *= pad_level * np.where((t >= 20) & (t < 36) | (t >= 44) & (t < 56), side, 1.0)
padL = pad * 1.0
padR = np.roll(pad, int(0.012 * SR))               # ensanche estéreo (Haas suave)
mixL += padL * 0.55; mixR += padR * 0.55

# ------------------------------------------------------------------ kick
def kick():
    n = int(0.45 * SR); tt = np.arange(n) / SR
    f = 45 + 110 * np.exp(-tt * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-tt * 7.5)
    click = rng.standard_normal(n) * np.exp(-tt * 900) * 0.3
    return np.tanh(1.6 * (body + click))
K = kick()
for kt in kick_times:
    add(K, int(kt * SR), 0.9)

# ------------------------------------------------------------------ clap
def clap():
    n = int(0.25 * SR); tt = np.arange(n) / SR
    noise = rng.standard_normal(n)
    noise = noise - one_pole_lp(noise, 900)                       # paso alto
    noise = one_pole_lp(noise, 4200)
    e = np.zeros(n)
    for off in (0, 0.011, 0.022):
        o = int(off * SR); e[o:] += np.exp(-(tt[: n - o]) * 38)
    return noise * e * 0.5
C = clap()
for b in list(range(10, 18)) + list(range(22, 28)):
    for beat in (1, 3):
        add(C, int((b * BAR + beat * BEAT) * SR), 0.55, pan=0.05)

# ------------------------------------------------------------------ hats
def hat(decay):
    n = int(0.12 * SR); tt = np.arange(n) / SR
    noise = rng.standard_normal(n)
    noise = noise - one_pole_lp(noise, 6500)
    return noise * np.exp(-tt * decay)
HC, HO = hat(70), hat(18)
for b in range(3, 28):
    if 18 <= b < 20:
        continue
    for k in range(8):
        tk = b * BAR + k * BEAT / 2
        if k % 2 == 1:
            add(HO if b >= 10 and k % 4 == 3 else HC, int(tk * SR), 0.16 if b < 10 else 0.22, pan=0.35)
        elif b >= 10:
            add(HC, int(tk * SR), 0.09, pan=-0.35)

# ------------------------------------------------------------------ bass
for b in list(range(10, 18)) + list(range(22, 28)):
    root = ROOTS[b % 4]
    for k in range(8):                                             # corcheas
        n = int(BEAT / 2 * SR); tt = np.arange(n) / SR
        f = hz(root)
        note = np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * 2 * f * tt)
        note = np.tanh(1.8 * note) * env_adsr(n, 0.004, 0.08, 0.6, 0.05)
        add(note, int((b * BAR + k * BEAT / 2) * SR), 0.34)

# ------------------------------------------------------------------ arpegio con eco
arp = np.zeros(N)
for b in range(6, 28):
    chord = CHORDS[b % 4]
    notes = [chord[0] + 12, chord[1] + 12, chord[2] + 12, chord[3] + 12, chord[2] + 12, chord[1] + 12]
    step = BEAT / 4 if b >= 10 else BEAT / 2
    count = int(BAR / step)
    for k in range(count):
        n = int(0.22 * SR); tt = np.arange(n) / SR
        f = hz(notes[k % len(notes)])
        pl = (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 3 * f * tt)) * np.exp(-tt * 16)
        s = int((b * BAR + k * step) * SR)
        e = min(N, s + n)
        arp[s:e] += pl[: e - s]
arp_level = np.interp(t, [12, 14, 20, 36, 44, 56], [0, .16, .2, .14, .2, .0])
arp *= arp_level
delay = int(3 * BEAT / 4 * SR)
arpL = arp.copy(); arpR = np.zeros(N)
fb = arp.copy()
for i in range(1, 4):                                              # eco ping-pong
    fb = np.concatenate([np.zeros(delay), fb[:-delay]]) * 0.45
    if i % 2: arpR += fb
    else: arpL += fb
mixL += arpL; mixR += arpR

# ------------------------------------------------------------------ risers e impactos
def riser(bars):
    n = int(bars * BAR * SR); tt = np.arange(n) / SR
    noise = rng.standard_normal(n)
    hp = noise - lp_block(noise, lambda s: 200 + 5000 * (s / (bars * BAR)) ** 2)
    return hp * (tt / tt[-1]) ** 2.2 * 0.45
for start, bars in ((8, 2), (20, 2)):
    r = riser(bars); s = int(start * BAR * SR)
    add(r, s, 0.6, pan=-0.2); add(np.roll(r, 300), s, 0.6, pan=0.2)

def impact():
    n = int(2.8 * SR); tt = np.arange(n) / SR
    boom = np.sin(2 * np.pi * (38 + 40 * np.exp(-tt * 6)) * tt) * np.exp(-tt * 2.4)
    noise = rng.standard_normal(n); noise = noise - one_pole_lp(noise, 3000)
    crash = noise * np.exp(-tt * 1.6) * 0.18
    return boom * 0.6 + crash
I = impact()
for bpos in (10, 22, 28):
    add(I, int(bpos * BAR * SR), 0.7 if bpos != 28 else 0.5)

# ------------------------------------------------------------------ reverb (convolución) y master
ir_n = int(1.6 * SR); irt = np.arange(ir_n) / SR
ir = rng.standard_normal(ir_n) * np.exp(-irt * 3.2); ir[0] = 0; ir /= np.sqrt(np.sum(ir ** 2))
def conv(x):
    L = len(x) + ir_n - 1; nfft = 1 << (L - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[: len(x)]
wetL, wetR = conv(mixL) * 0.16, conv(np.roll(mixR, 211)) * 0.16
outL, outR = mixL + wetL, mixR + wetR
fade = np.interp(t, [0, 0.05, BARS * BAR - 3.5, BARS * BAR], [0, 1, 1, 0])
out = np.stack([outL, outR], axis=1) * fade[:, None]
out = np.tanh(out / np.max(np.abs(out)) * 1.4) * 0.89             # glue suave + margen
pcm = (out * 32767).astype(np.int16)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print(f"{sys.argv[1]}: {N / SR:.2f} s, {BARS} compases a {BPM} BPM")
