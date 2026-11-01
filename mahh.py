
import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heuristicas as L
import solucion_inicial as J   # lectura de instancias, verificador y salida

EPS = L.EPS
H = L.H



# MAHH 
class MAHH:
    def __init__(self, inst, learner_cls, time_limit=30.0, theta=0.035,
                 max_seq_len=25, use_rr=True, seed=0, bonus=3.0, slack=5.0,
                 learner_kw=None, verbose=True):
        self.inst = inst
        self.time_limit = time_limit
        self.theta = theta
        self.max_seq_len = max_seq_len   # largo máximo de una secuencia (0 = sin límite)
        self.use_rr = use_rr
        self.verbose = verbose
        self.name = learner_cls.NAME
        self.rng = random.Random(seed)
        nrng = np.random.default_rng(seed)
        self.llh = L.LowLevelHeuristics(inst, self.rng, bonus, slack)
        self.learner = learner_cls(H, nrng, **(learner_kw or {}))
        self.stats = np.zeros((H, 3), dtype=int)
        self.seq_lens = []    # largo de cada secuencia aplicada
        self.best_seqs = []   # (tiempo, secuencia, objetivo) de cada nuevo mejor

    def _accept(self, c_t, c_c, c_best, tel):
        """Criterio de aceptación, ecuaciones (6) y (7)."""
        if c_t <= c_c + EPS:
            return True
        if not self.use_rr:
            return False
        if c_best >= 1e9:
            rho = 1e-3
        else:
            rho = 1e-5 + self.theta * (1 - tel / self.time_limit)
        return c_t < (1 + rho) * c_best

    def run(self, s_init, cost_init=None):
        """Algoritmo 4. cost_init: costo de s_init si ya se conoce."""
        inst = self.inst
        s_c = s_init.copy()
        cost_c = cost_init if cost_init is not None else L.evaluate(inst, s_c)[0]
        s_best, cost_best = s_c.copy(), cost_c
        history = [(0.0, cost_best)]
        seq, iters = [], 0
        i_last = self.rng.randrange(H)
        t0 = time.time()
        while True:
            tel = time.time() - t0
            if tel >= self.time_limit:
                break
            i_next = self.learner.sample_next(i_last)
            seq.append(i_next)
            u_next = self.learner.sample_apply(i_next)
            al_maximo = self.max_seq_len > 0 and len(seq) >= self.max_seq_len
            if u_next == 1 or al_maximo:
                s_t, cortada = s_c, False
                for h in seq:
                    if time.time() - t0 >= self.time_limit:
                        cortada = True           # sin tiempo: se descarta y se termina
                        break
                    s_t = self.llh.apply(s_t, h)
                if cortada:
                    break
                if L.same_solution(s_t, s_c):   # la secuencia no cambió nada
                    cost_t = cost_c
                else:
                    cost_t = L.evaluate(inst, s_t)[0]
                improved = cost_t < cost_c - 1e-9
                self.learner.update(seq, improved)
                new_best = cost_t < cost_best - 1e-9
                self.seq_lens.append(len(seq))
                for h in seq:
                    self.stats[h, 0] += 1
                for h in set(seq):
                    self.stats[h, 1] += int(improved)
                    self.stats[h, 2] += int(new_best)
                if self._accept(cost_t, cost_c, cost_best, tel):
                    s_c, cost_c = s_t, cost_t
                if new_best:
                    s_best, cost_best = s_t.copy(), cost_t
                    history.append((time.time() - t0, cost_best))
                    self.best_seqs.append((history[-1][0], list(seq), cost_best))
                seq = []
                iters += 1
                i_last = self.rng.randrange(H)   # nueva secuencia desde una LL al azar
            else:
                i_last = i_next
        if self.verbose:
            print("  MAHH-%s: objetivo=%.2f iteraciones=%d" % (self.name, cost_best, iters))
        return s_best, cost_best, history, iters


def run_restarts(inst, s0, learner_cls, time_limit, restarts=1, chain=True, seed=0, **kw):
    """Reparte el tiempo en `restarts` corridas encadenadas, con semillas
    distintas."""
    per = time_limit / max(1, restarts)
    best, best_cost = s0, L.evaluate(inst, s0)[0]
    cost0 = best_cost                        # costo de s0 (ya evaluada)
    info = {"hist": [(0.0, best_cost)], "iters": 0, "stats": np.zeros((H, 3), dtype=int),
            "seq_lens": [], "best_seqs": [], "restarts": [], "best_seed": None}
    off = 0.0
    for r in range(max(1, restarts)):
        seed_r = seed * 1000 + r                 # semilla propia de este reinicio
        hh = MAHH(inst, learner_cls, per, seed=seed_r, **kw)
        s, c, h, it = hh.run(best if chain else s0, best_cost if chain else cost0)
        info["restarts"].append({"reinicio": r + 1, "semilla": seed_r, "objetivo": c,
                                 "mejoro": c < best_cost - 1e-9})
        info["iters"] += it
        info["stats"] += hh.stats
        info["seq_lens"] += hh.seq_lens
        if c < best_cost - 1e-9:
            best, best_cost = s, c
            info["best_seed"] = seed_r
            info["hist"] += [(off + t, x) for t, x in h[1:]]
            info["best_seqs"] += [(off + t, q, x) for t, q, x in hh.best_seqs]
        off += per
    return best, best_cost, info



# Resolver una instancia

def ll_label(h):
    return "LL%d %s" % (h + 1, L.NAMES[h][3:])


def save_heuristics(path, stats):
    """Guarda el uso de cada LL."""
    total = max(1, int(stats[:, 0].sum()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ll", "nombre", "usos", "usos_%", "en_mejoras_actual", "en_nuevo_mejor"])
        for h in sorted(range(H), key=lambda h: -stats[h, 0]):
            w.writerow([h + 1, L.NAMES[h][3:], int(stats[h, 0]),
                        round(100.0 * stats[h, 0] / total, 2), int(stats[h, 1]), int(stats[h, 2])])


def save_best_sequences(path, best_seqs):
    """Cada secuencia que encontró una nueva mejor solución."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tiempo_s", "objetivo", "largo", "secuencia"])
        for t, q, c in best_seqs:
            w.writerow([round(t, 3), round(c, 2), len(q), " - ".join("LL%d" % (h + 1) for h in q)])


def print_heuristics(stats, top=None, indent="  "):
    total = max(1, int(stats[:, 0].sum()))
    order = sorted(range(H), key=lambda h: -stats[h, 0])[:top]
    print("%s%-32s %10s %9s %13s %12s" % (indent, "heurística", "usos", "", "mejoran_act", "nuevo_mejor"))
    for h in order:
        print("%s%-32s %10d (%6.2f%%) %13d %12d" % (indent, ll_label(h), stats[h, 0],
                                                   100.0 * stats[h, 0] / total, stats[h, 1], stats[h, 2]))


def save_evolution(path, hist, time_limit):
    """Guarda la evolución del mejor objetivo en el tiempo."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tiempo_s", "mejor_objetivo"])
        for t, c in hist:
            w.writerow([round(t, 3), round(c, 2)])
        w.writerow([round(max(time_limit, hist[-1][0]), 3), round(hist[-1][1], 2)])


def solve_file(path, learner_cls, time_limit=60.0, seed=0, rounding="ceil",
               theta=0.035, rr="on", max_seq=25, restarts=3, chain=True,
               learner_kw=None, outdir=None, pp_frac=0.10):
    """Resuelve una instancia: solución inicial, MAHH, post-proceso y
    salidas."""
    t0 = time.time()
    inst = J.Instance(path, rounding)
    s0 = L.initial_solution(inst)
    t_ini = time.time() - t0
    obj0 = L.final_outputs(inst, s0.copy())[2]   # misma medición que la solución final

    best, best_cost, info = run_restarts(
        inst, s0, learner_cls, time_limit, restarts, chain, seed, theta=theta,
        max_seq_len=max_seq, use_rr=(rr == "on"), learner_kw=learner_kw, verbose=False)

    best_mahh = best.copy()
    obj_mahh = L.final_outputs(inst, best_mahh.copy())[2]   # resultado de la MAHH sola
    t_pp0 = time.time()
    pp = {"mejoras": 0, "pasadas": [], "iteraciones": 0, "nota": "sin post-proceso"}
    t_pp = max(0.0, pp_frac) * time_limit
    if t_pp > 0:
        import postproceso
        usa_h = inst.P <= 40 and postproceso._politica_importa(inst, best)
        t_h = 0.5 * t_pp if usa_h else 0.0
        best, pp = postproceso.pulir(
            inst, best, time_limit=t_pp - t_h, pp_frac=1.0, seed=seed, cost0=best_cost,
            min_seconds=0.0, max_seconds=t_pp - t_h, h_seconds=t_h, verbose=False)
    pp_t = time.time() - t_pp0

    schs, cr, obj, errs = L.final_outputs(inst, best)
    if obj > obj_mahh + 1e-9:
        out = L.final_outputs(inst, best_mahh)
        if not out[3]:
            best = best_mahh
            schs, cr, obj, errs = out
    note = ""
    if errs:
        schs, cr, obj, errs0 = L.final_outputs(inst, s0)
        note = "MAHH infactible, se devolvió la inicial: " + "; ".join(errs[:2])
        errs = errs0
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0])
        with open(base + "_mahh.json", "w", encoding="utf-8") as f:
            json.dump(J.to_json(inst, schs, cr, obj), f, indent=2, ensure_ascii=False)
        save_evolution(base + "_evolucion.csv", info["hist"], time_limit)
        save_heuristics(base + "_heuristicas.csv", info["stats"])
        save_best_sequences(base + "_secuencias.csv", info["best_seqs"])
    cost_veh = inst.w1 * len(schs) + inst.w3 * sum(sc["vdist"] for sc in schs)
    cost_crew = inst.w2 * cr["ncrews"] + inst.w4 * cr["cdist"]
    res = {"instancia": os.path.basename(path), "metodo": learner_cls.NAME,
           "semilla": seed, "pares": inst.P, "obj_inicial": round(obj0, 2), "obj_mahh": round(obj_mahh, 2),
           "obj_final": round(obj, 2),
           "semillas": " ".join(str(x["semilla"]) for x in info["restarts"]),
           "semilla_mejor": (info["best_seed"] if info["best_seed"] is not None else "inicial"),
           "costo_vehiculos": round(cost_veh, 2), "costo_tripulacion": round(cost_crew, 2),
           "mejora_%": round(100 * (obj0 - obj) / obj0, 2) if obj0 else 0.0,
           "vehiculos": len(schs), "tripulaciones": cr["ncrews"], "iteraciones": info["iters"],
           "mejoras": len(info["hist"]) - 1, "t_mejor": round(info["hist"][-1][0], 3),
           "L_prom": round(float(np.mean(info["seq_lens"])), 3) if info["seq_lens"] else 0.0,
           "L_max": max(info["seq_lens"]) if info["seq_lens"] else 0,
           "factible": "SI" if not errs else "NO", "nota": note,
           "t_inicial_s": round(t_ini, 2), "tiempo_s": round(time.time() - t0, 1),
           "tiempo_limite": time_limit, "restarts": restarts, "theta": theta, "max_seq": max_seq, "redondeo": rounding,
           "stats": info["stats"].tolist(), "pp_mejoras": pp["mejoras"], "pp_pasadas": ";".join("%s:%+.1f" % (n, -d) for n, d, _ in pp["pasadas"]), "pp_tiempo_s": round(pp_t, 2), "pp_iter": pp["iteraciones"], "pp_nota": pp["nota"]}
    # para el total de una carpeta (no va al CSV)
    return res, info



KEYS = ["instancia", "metodo", "semilla", "semillas", "semilla_mejor", "pares", "obj_inicial",
        "obj_mahh", "obj_final",
        "costo_vehiculos", "costo_tripulacion", "mejora_%", "vehiculos",
        "tripulaciones", "iteraciones", "mejoras", "t_mejor", "L_prom", "L_max",
        "factible", "nota", "t_inicial_s", "tiempo_s", "tiempo_limite", "restarts", "theta",
        "max_seq", "redondeo", "pp_mejoras", "pp_pasadas", "pp_tiempo_s", "pp_iter", "pp_nota"]


def update_summary(outdir, rows):
    """Actualiza resumen.csv y heuristicas_total.csv con las filas nuevas."""
    path = os.path.join(outdir, "resumen.csv")
    old = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    merged = {r["instancia"]: r for r in old}
    for r in rows:
        merged[r["instancia"]] = r
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader()
        for k in sorted(merged, key=natural_key):
            w.writerow(merged[k])
    total = np.zeros((H, 3), dtype=int)
    skipped = set()
    for fn in glob.glob(os.path.join(outdir, "*_heuristicas.csv")):
        with open(fn, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                h = int(r["ll"]) - 1   # el CSV numera las LLs desde 1
                # solo se suman filas de esta misma versión de heuristicas.py
                if not (0 <= h < H) or r["nombre"] != L.NAMES[h][3:]:
                    skipped.add(os.path.basename(fn))
                    continue
                total[h] += [int(r["usos"]), int(r["en_mejoras_actual"]), int(r["en_nuevo_mejor"])]
    if skipped:
        print("Aviso: %d archivo(s) _heuristicas.csv de otra versión de las LLs no se "
              "sumaron a heuristicas_total.csv (p. ej. %s)" % (len(skipped), sorted(skipped)[0]))
    save_heuristics(os.path.join(outdir, "heuristicas_total.csv"), total)
    return path, len(merged), total


def _is_instance(path):
    """True si el .txt parece una instancia."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return "VehicleUseCost" in f.read(4096)
    except OSError:
        return False


def natural_key(path):
    """Orden natural de nombres (1_, 2_, ..., 10_)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def _batch_task(args):
    path, learner_cls, kw = args
    try:
        res, _ = solve_file(path, learner_cls, **kw)
        return res
    except Exception as e:
        return {"instancia": os.path.basename(path), "metodo": learner_cls.NAME,
                "factible": "NO", "nota": "%s: %s" % (type(e).__name__, e)}


def base_parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("instancia", help="archivo .txt o carpeta con instancias")
    ap.add_argument("--time", type=float, default=60.0,
                    help="segundos de la MAHH por instancia (el post-proceso va aparte, ver --pp-frac)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounding", choices=["none", "ceil", "round"], default="ceil",
                    help="redondeo de distancias (ceil reproduce el anexo de Lam et al.)")
    ap.add_argument("--theta", type=float, default=0.035,
                    help="tolerancia de aceptación (vartheta en la ec. 7)")
    ap.add_argument("--rr", choices=["on", "off"], default="on",
                    help="criterio de aceptación de las ec. (6)-(7); off = solo no empeorar")
    ap.add_argument("--max-seq", type=int, default=25,
                    help="largo máximo de una secuencia de LLs: al llegar a este largo se "
                         "aplica aunque el learner no lo haya decidido (0 = sin límite, como en el paper)")
    ap.add_argument("--restarts", type=int, default=3,
                    help="reparte el tiempo en k corridas con semillas distintas (1 = como el paper)")
    ap.add_argument("--no-chain", action="store_true",
                    help="cada reinicio parte de la solución inicial (por defecto, de la mejor)")
    ap.add_argument("--pp-frac", type=float, default=0.10,
                    help="tiempo del post-proceso, adicional al de la MAHH, como fracción de "
                         "--time (0.10 = 10 %%; 0 = sin post-proceso)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="procesos en paralelo al resolver una carpeta")
    ap.add_argument("--outdir", default=None, help="carpeta de salida")
    ap.add_argument("--top", type=int, default=10,
                    help="cuántas heurísticas mostrar en la tabla de uso por pantalla")
    return ap


def run_cli(learner_cls, args, learner_kw=None):
    a = args
    outdir = a.outdir or ("resultados_%s_s%d" % (learner_cls.NAME.lower(), a.seed))
    kw = dict(time_limit=a.time, seed=a.seed, rounding=a.rounding, theta=a.theta,
              rr=a.rr, max_seq=a.max_seq, restarts=a.restarts, chain=not a.no_chain,
              learner_kw=learner_kw,
              outdir=outdir, pp_frac=a.pp_frac)

    if os.path.isdir(a.instancia):
        from multiprocessing import Pool
        files = sorted((f for f in glob.glob(os.path.join(a.instancia, "**", "*.txt"), recursive=True)
                        if _is_instance(f)), key=natural_key)
        if not files:
            sys.exit("No hay instancias del JVCRSP (.txt con VehicleUseCost) en " + a.instancia)
        print("MAHH-%s: %d instancias, %d procesos, %.0f s cada una"
              % (learner_cls.NAME, len(files), a.workers, a.time))
        rows = []
        with Pool(a.workers) as pool:
            for k, r in enumerate(pool.imap(_batch_task, [(f, learner_cls, kw) for f in files]), 1):
                rows.append(r)
                print("[%d/%d] %-26s %s -> %s  (veh=%s, trip=%s)  (%s%%)  %s  %s" % (
                    k, len(files), r["instancia"], r.get("obj_inicial", ""), r.get("obj_final", ""),
                    r.get("costo_vehiculos", ""), r.get("costo_tripulacion", ""),
                    r.get("mejora_%", ""), r.get("factible"), r.get("nota", "")))
        os.makedirs(outdir, exist_ok=True)
        path, n_tot, total = update_summary(outdir, rows)
        print("Factibles: %d de %d en esta corrida. Resumen (%d instancias en total) en %s"
              % (sum(r.get("factible") == "SI" for r in rows), len(rows), n_tot, path))

        ok = [r for r in rows if "stats" in r]
        print("\nResultados:")
        print("%-24s %5s %10s %10s %8s %9s %9s %7s %6s" % (
            "instancia", "pares", "inicial", "final", "mejoras", "t_mejor", "tiempo_s", "L_prom", "L_max"))
        print("-" * 94)
        for r in ok:
            print("%-24s %5d %10.1f %10.1f %8d %9.3f %9.1f %7.3f %6d" % (
                r["instancia"][:-4], r["pares"], r["obj_inicial"], r["obj_final"], r["mejoras"],
                r["t_mejor"], r["tiempo_s"], r["L_prom"], r["L_max"]))
        if total.sum():
            print("\nUso de heurísticas (todas las instancias de la carpeta, %d más usadas):"
                  % min(a.top, H))
            print_heuristics(total, a.top)
        return 0

    res, info = solve_file(a.instancia, learner_cls, **kw)
    update_summary(outdir, [res])
    for k in KEYS:
        print("%-18s: %s" % (k, res.get(k, "")))
    print("\nReinicios (semilla propia de cada uno):")
    for x in info["restarts"]:
        print("  reinicio %d: semilla %d -> objetivo %.2f %s" % (
            x["reinicio"], x["semilla"], x["objetivo"], "(nuevo mejor)" if x["mejoro"] else ""))
    print("\nUso de heurísticas (%d más usadas):" % min(a.top, H))
    print_heuristics(info["stats"], a.top)
    if info["best_seqs"]:
        print("\nSecuencias que encontraron una nueva mejor solución (últimas 5):")
        for t, q, c in info["best_seqs"][-5:]:
            print("  t=%8.3f s  objetivo=%10.2f  [%s]" % (t, c, " - ".join("LL%d" % (h + 1) for h in q)))
    print("\nArchivos en %s/" % outdir)
    return 0 if res["factible"] == "SI" else 1