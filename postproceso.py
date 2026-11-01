#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JVCRSP - Post-proceso de la solución final de la MAHH.

Pulidor monótono (nunca empeora el objetivo). Trabaja sobre una copia de la
Sol ganadora, reparte su tiempo por fracciones del presupuesto (no por el
tamaño de la instancia, así que sirve igual para chicas, medianas y grandes) y
al final verifica factibilidad: si algo fallara, devuelve la solución de
entrada sin cambios.

Fases (cada una con tope de tiempo; todas aceptan solo si el objetivo completo
baja, y la búsqueda se repite si la última fase logró mejorar):
  1. Pulido determinista hasta convergencia:
       A. calendario (quitar o reducir esperas)
       B. políticas de asignación de tripulaciones
       C. eliminación de vehículos (reinsertar sus pares sin abrir vehículos)
       D. cambios dentro de una ruta (inversión, or-opt, swap)
       E. relocate (varias rutas destino por par) y swap exhaustivo de pares
       F. fusión de dos rutas
       G. disolver las rutas más caras y reinsertar con regret
  2. Política de tripulaciones (H): descenso sobre cortes forzados y búsqueda
     aleatoria de 2-3 cortes a la vez. Escala con el número de tramos; no
     enumera combinaciones, que en instancias medianas y grandes explotan y
     solo alcanzan los primeros vehículos. Se omite si la asignación exacta
     de tripulaciones aplica (ahí la política no cambia el costo).
  3. Cambios de rutas con horarios ajustados (K): reordena rutas completas y
     mueve pares; a los cambios prometedores les ajusta los horarios (con I)
     antes de compararlos, porque hay rutas que con horarios tempranos parecen
     peores y con las esperas correctas son mejores.
  3b. Horarios (I): prueba retrasos enteros de la salida de cada nodo y pares
     de cambios, en orden aleatorio.
  4. Búsqueda con el tiempo restante (J): secuencias cortas de las LLs de la
     MAHH con aceptación de mejoras o empates. Es lo que aprovecha el tiempo
     cuando la MAHH no alcanzó a converger (típico en instancias grandes) y
     lo que hace converger primero a las fases anteriores en las chicas.

Toda pasada devuelve (sol, costo, mejoró). Usa rng con semilla, pero los
presupuestos dependen del reloj, así que el resultado puede variar entre
corridas.

Uso:
    import postproceso
    best, info_pp = postproceso.pulir(inst, best, time_limit=60.0, pp_frac=0.10, seed=0)
"""
import random
import time

import heuristicas as L

EPS = 1e-9


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def _obj(inst, sol):
    return L.evaluate(inst, sol)[0]


def _try_routes(inst, sol, new_routes, cost):
    """(sol, costo, True) si new_routes mejora el costo; si no, (sol, costo, False)."""
    if new_routes is None:
        return sol, cost, False
    cand = L.Sol(new_routes, sol.pol)
    c = _obj(inst, cand)
    if c < cost - EPS:
        return cand, c, True
    return sol, cost, False


def _clave(R):
    return tuple(R["route"])


def _candidatos_calendario(R):
    """Esperas candidatas para una ruta: ninguna, solo la del depósito, quitar
    cada espera y reducir cada espera intermedia a la mitad (sin repetidos)."""
    md0 = R["min_dep"]
    mds = [{}]
    if md0:
        mds.append({0: md0[0]} if 0 in md0 else {})
        for k, v in md0.items():
            m = dict(md0)
            m.pop(k)
            mds.append(m)
            if k != 0 and v > 1e-6:
                m2 = dict(md0)
                m2[k] = v * 0.5
                mds.append(m2)
    out, vistos = [], {tuple(sorted(md0.items()))}   # sin repetir la actual
    for m in mds:
        key = tuple(sorted(m.items()))
        if key not in vistos:
            vistos.add(key)
            out.append(m)
    return out


# ----------------------------------------------------------------------------
# Fase 1: pulido determinista (A-G)
# ----------------------------------------------------------------------------
def _pasada_A_calendario(inst, sol, cost, lls, t_fin):
    """Para cada ruta prueba todos sus candidatos de calendario; si uno mejora,
    los vuelve a generar desde la ruta nueva."""
    mejoro = False
    i = 0
    while i < len(sol.routes) and time.monotonic() < t_fin:
        R = sol.routes[i]
        cambio = False
        for md in _candidatos_calendario(R):
            if time.monotonic() > t_fin:
                break
            m = L.mk(inst, R["route"], md)
            if m is None:
                continue
            new = list(sol.routes)
            new[i] = m
            sol, cost, ok = _try_routes(inst, sol, new, cost)
            if ok:
                mejoro = cambio = True
                break
        if not cambio:
            i += 1
    return sol, cost, mejoro


def _pasada_B_politica(inst, sol, cost, lls, t_fin):
    """Prueba varias políticas de asignación de tripulaciones."""
    candidatas = [L.default_pol()]
    for fit in ("best", "worst"):
        p = L.default_pol()
        p["fit"] = fit
        candidatas.append(p)
    p = L.default_pol()
    p["prefer_handoff"] = True
    candidatas.append(p)
    for s in (1, 7, 13):
        p = L.default_pol()
        p["shuffle"], p["seed"] = True, s
        candidatas.append(p)
    mejoro = False
    for p in candidatas:
        if time.monotonic() > t_fin:
            break
        cand = L.Sol(sol.routes, p)
        c = _obj(inst, cand)
        if c < cost - EPS:
            sol, cost, mejoro = cand, c, True
    return sol, cost, mejoro


def _pasada_C_quitar_vehiculos(inst, sol, cost, lls, t_fin):
    """Intenta eliminar cada ruta reinsertando sus pares en las demás. En cada
    paso elige, entre las rutas aún no probadas, la de menos pares y mayor rc."""
    mejoro = False
    probadas = set()
    while time.monotonic() < t_fin and len(sol.routes) > 1:
        cands = [i for i, R in enumerate(sol.routes) if _clave(R) not in probadas]
        if not cands:
            break
        i = min(cands, key=lambda i: (len(sol.routes[i]["route"]), -sol.routes[i]["rc"]))
        probadas.add(_clave(sol.routes[i]))
        pares = lls._pairs(sol.routes[i]["route"])
        base = [R for k, R in enumerate(sol.routes) if k != i]
        lls.rng.shuffle(pares)
        if lls._repair_greedy(base, pares, allow_new=False):
            sol, cost, ok = _try_routes(inst, sol, base, cost)
            mejoro = mejoro or ok
    return sol, cost, mejoro


def _variantes_intra(Lr):
    """Inversión de segmentos, or-opt de 1 a 3 nodos y swap de dos nodos."""
    n = len(Lr)
    for a in range(n - 1):
        for b in range(a + 1, n):
            yield Lr[:a] + Lr[a:b + 1][::-1] + Lr[b + 1:]
    for s in (1, 2, 3):
        for st in range(n - s + 1):
            seg, rest = Lr[st:st + s], Lr[:st] + Lr[st + s:]
            for pos in range(len(rest) + 1):
                if pos != st:
                    yield rest[:pos] + seg + rest[pos:]
    for a in range(n - 1):
        for b in range(a + 1, n):
            new = list(Lr)
            new[a], new[b] = new[b], new[a]
            yield new


def _pasada_D_intra_ruta(inst, sol, cost, lls, t_fin, top=6):
    """Para cada ruta ordena sus variantes factibles por rc (estimación) y evalúa
    con el objetivo real las `top` mejores."""
    mejoro = False
    for ridx in range(len(sol.routes)):
        if time.monotonic() > t_fin:
            break
        Lr = sol.routes[ridx]["route"]
        if len(Lr) < 3:
            continue
        cands = {}
        for new in _variantes_intra(Lr):
            if time.monotonic() > t_fin:
                break
            t = tuple(new)
            if new == Lr or t in cands:
                continue
            m = L.mk(inst, new)
            if m is not None:
                cands[t] = m
        for m in sorted(cands.values(), key=lambda m: m["rc"])[:top]:
            if time.monotonic() > t_fin:
                break
            new_routes = list(sol.routes)
            new_routes[ridx] = m
            sol, cost, ok = _try_routes(inst, sol, new_routes, cost)
            if ok:
                mejoro = True
                break
    return sol, cost, mejoro


def _pasada_E_inter_ruta(inst, sol, cost, lls, t_fin, topk=3, nswap=80):
    """Entre rutas, hasta que nada mejore o se acabe el tiempo:
      - relocate: cada par (en orden aleatorio) a sus `topk` mejores rutas
        destino, con evaluación completa;
      - swap: se generan TODOS los intercambios de un par entre dos rutas, se
        ordenan por la estimación rc y se evalúan los `nswap` más prometedores."""
    mejoro, seguir = False, True
    while seguir and time.monotonic() < t_fin:
        seguir = False
        pares = lls._all_pairs(sol.routes)
        lls.rng.shuffle(pares)
        for p, ridx in pares:
            if time.monotonic() > t_fin:
                break
            base = lls._remove(sol.routes, {p})
            if base is None:
                continue
            gone = len(base) < len(sol.routes)
            opts = lls._options(base, p, () if gone else (ridx,), allow_new=not gone)
            for opt in opts[:topk]:
                tmp = list(base)
                if lls._apply_opt(tmp, opt):
                    sol, cost, ok = _try_routes(inst, sol, tmp, cost)
                    if ok:
                        mejoro = seguir = True
                        break
            if seguir:
                break
        if seguir or len(sol.routes) < 2:
            continue
        R = sol.routes
        cands = []
        for r1 in range(len(R) - 1):
            for r2 in range(r1 + 1, len(R)):
                for p1 in lls._pairs(R[r1]["route"]):
                    for p2 in lls._pairs(R[r2]["route"]):
                        if time.monotonic() > t_fin:
                            break
                        a = lls._best_in(lls._without(R[r2]["route"], p2), p1)   # p1 -> r2
                        b = lls._best_in(lls._without(R[r1]["route"], p1), p2)   # p2 -> r1
                        if a is None or b is None:
                            continue
                        cands.append((a[0] + b[0] - R[r1]["rc"] - R[r2]["rc"], r1, r2, a[1], b[1]))
        cands.sort(key=lambda t: t[0])
        for _, r1, r2, l2, l1 in cands[:nswap]:
            if time.monotonic() > t_fin:
                break
            m1, m2 = L.mk(inst, l1), L.mk(inst, l2)
            if m1 is None or m2 is None:
                continue
            new_routes = list(sol.routes)
            new_routes[r1], new_routes[r2] = m1, m2
            sol, cost, ok = _try_routes(inst, sol, new_routes, cost)
            if ok:
                mejoro = seguir = True
                break
    return sol, cost, mejoro


def _pasada_F_fusion(inst, sol, cost, lls, t_fin):
    """Fusiona el par de rutas con mejor rc combinado (si ahorra según rc)."""
    if len(sol.routes) < 2:
        return sol, cost, False
    best = None
    n = len(sol.routes)
    for a in range(n - 1):
        for b in range(a + 1, n):
            if time.monotonic() > t_fin:
                break
            L1, L2 = sol.routes[a]["route"], sol.routes[b]["route"]
            merged = L.mk(inst, L1 + L2) or L.mk(inst, L2 + L1)
            if merged is None:
                base = list(L1)
                for p in lls._pairs(L2):
                    ins = lls._best_in(base, p)
                    if ins is None:
                        base = None
                        break
                    base = ins[1]
                if base is not None:
                    merged = L.mk(inst, base)
            if merged is None:
                continue
            ahorro = sol.routes[a]["rc"] + sol.routes[b]["rc"] + inst.w1 - merged["rc"]
            if ahorro > EPS and (best is None or merged["rc"] < best[0]):
                best = (merged["rc"], a, b, merged)
    if best is None:
        return sol, cost, False
    _, a, b, merged = best
    new_routes = [R for i, R in enumerate(sol.routes) if i not in (a, b)] + [merged]
    return _try_routes(inst, sol, new_routes, cost)


def _pasada_G_regret(inst, sol, cost, lls, t_fin, intentos=2):
    """Disuelve la ruta con mayor rc por par (hasta `intentos` rutas distintas)
    y reinserta sus pares con regret, sin abrir vehículos."""
    mejoro = False
    probadas = set()
    for _ in range(intentos):
        if time.monotonic() > t_fin or len(sol.routes) < 2:
            break
        cands = [i for i, R in enumerate(sol.routes) if _clave(R) not in probadas]
        if not cands:
            break
        i = max(cands, key=lambda i: sol.routes[i]["rc"] / max(1, len(sol.routes[i]["route"])))
        probadas.add(_clave(sol.routes[i]))
        pares = lls._pairs(sol.routes[i]["route"])
        base = [R for k, R in enumerate(sol.routes) if k != i]
        lls.rng.shuffle(pares)
        if lls._repair_regret(base, pares, allow_new=False):
            sol, cost, ok = _try_routes(inst, sol, base, cost)
            mejoro = mejoro or ok
    return sol, cost, mejoro


PASADAS = (("A", _pasada_A_calendario, 0.25), ("B", _pasada_B_politica, 0.05),
           ("C", _pasada_C_quitar_vehiculos, 0.30), ("D", _pasada_D_intra_ruta, 0.10),
           ("E", _pasada_E_inter_ruta, 0.15), ("F", _pasada_F_fusion, 0.10),
           ("G", _pasada_G_regret, 0.05))


def _pulido(inst, sol, cost, lls, t_fin, registrar, ronda):
    """Repite A-G hasta que ninguna pasada mejora (o se acaba el tiempo).
    Devuelve (sol, costo, mejoró)."""
    mejoro = False
    presupuesto = max(0.0, t_fin - time.monotonic())
    while time.monotonic() < t_fin:
        hubo = False
        for nombre, fn, peso in PASADAS:
            if time.monotonic() >= t_fin:
                break
            antes = cost
            sol, cost, ok = fn(inst, sol, cost, lls,
                               min(t_fin, time.monotonic() + max(peso * presupuesto, 0.5)))
            hubo = hubo or ok
            registrar(nombre, antes, cost, ronda)
        mejoro = mejoro or hubo
        if not hubo:
            break
    return sol, cost, mejoro


# ----------------------------------------------------------------------------
# Fase 2: política de tripulaciones (H)
# ----------------------------------------------------------------------------
def _politica_importa(inst, sol):
    """False si la asignación exacta de tripulaciones aplica a estas rutas: en
    ese caso evaluate ya usa el óptimo y la política del voraz no cambia el costo."""
    return L.exact_crews(inst, [R["sch"] for R in sol.routes]) is None


def _con_politica(sol, **kw):
    p = dict(sol.pol)
    p["forced"] = set(sol.pol.get("forced", ()))
    p.update(kw)
    return L.Sol(list(sol.routes), p)


def _pasada_H_politica(inst, sol, cost, lls, t_fin):
    """(1) Las 4 combinaciones best/worst-fit x prefer_handoff. (2) Descenso sobre
    los cortes forzados: agrega o quita un tramo a la vez (primera mejora, orden
    aleatorio) hasta que ninguno mejora. (3) Con el tiempo que quede, cambia 2 o 3
    cortes a la vez al azar. Cada paso es O(1) evaluaciones por tramo, así que
    escala a cualquier tamaño."""
    mejoro = False
    for fit in ("best", "worst"):
        for ph in (False, True):
            if time.monotonic() > t_fin:
                return sol, cost, mejoro
            if fit == sol.pol.get("fit", "best") and ph == sol.pol.get("prefer_handoff", False):
                continue               # es la política actual: mismo costo, no se evalúa
            cand = _con_politica(sol, fit=fit, prefer_handoff=ph)
            c = _obj(inst, cand)
            if c < cost - EPS:
                sol, cost, mejoro = cand, c, True
    rng = lls.rng
    seguir = True
    while seguir and time.monotonic() < t_fin:
        seguir = False
        tramos = [(v, m) for v, R in enumerate(sol.routes) for m in range(len(R["sch"]["visits"]) - 1)]
        rng.shuffle(tramos)
        for leg in tramos:
            if time.monotonic() > t_fin:
                break
            cand = _con_politica(sol, forced=set(sol.pol.get("forced", ())) ^ {leg})
            c = _obj(inst, cand)
            if c < cost - EPS:
                sol, cost, mejoro, seguir = cand, c, True, True
                break
    tramos = [(v, m) for v, R in enumerate(sol.routes) for m in range(len(R["sch"]["visits"]) - 1)]
    while len(tramos) >= 3 and time.monotonic() < t_fin:
        flips = set(rng.sample(tramos, rng.choice((2, 3))))
        cand = _con_politica(sol, forced=set(sol.pol.get("forced", ())) ^ flips)
        c = _obj(inst, cand)
        if c < cost - EPS:
            sol, cost, mejoro = cand, c, True
            tramos = [(v, m) for v, R in enumerate(sol.routes) for m in range(len(R["sch"]["visits"]) - 1)]
    return sol, cost, mejoro


# ----------------------------------------------------------------------------
# Fase 3: horarios (I)
# ----------------------------------------------------------------------------
def _valores_espera(inst, R, p):
    """Retrasos candidatos de la salida del nodo p (p = 0: depósito): las sumas
    parciales de las esperas posteriores de la ruta, es decir, mover una espera
    existente hasta p (la ruta llega igual al final)."""
    sch = R["sch"]
    k = len(R["route"])
    out = set()
    acc = 0.0
    for q in range(k, p, -1):                       # esperas de los nodos q > p
        w = sch["serv"][q] - sch["arr"][q]
        w += max(0.0, sch["dep"][q] - sch["serv"][q] - inst.s[R["route"][q - 1]])
        if w > 1e-9:
            acc += w
            out.add(acc)
    return sorted(int(round(v)) for v in out if v >= 1)


def _pares_de_esperas(inst, pol, routes, cost, t_fin):
    """Prueba pares de cambios de horario en una misma ruta: (mover una espera
    hasta la posición p) x (cualquier retraso entero de la salida del depósito, o
    mover otra espera hasta q). Devuelve (cost, routes) si mejora, o None."""
    best = None
    for ridx, R in enumerate(routes):
        route = R["route"]
        if not route:
            continue
        k = len(route)
        base = R["sch"]["dep"]
        d0max = 0
        m_try = L.mk(inst, route, {**R["min_dep"], 0: base[0] + 1})
        while m_try is not None and d0max < 400:
            d0max += 1
            m_try = L.mk(inst, route, {**R["min_dep"], 0: base[0] + d0max + 1})
        for p in range(1, k + 1):
            for a in _valores_espera(inst, R, p):
                if time.monotonic() > t_fin:
                    return best
                md1 = dict(R["min_dep"])
                md1[p] = max(md1.get(p, 0.0), base[p] + a)
                m1 = L.mk(inst, route, md1)
                if m1 is None:
                    continue
                # (a) mover espera + retraso de salida
                for d0 in range(1, d0max + 1):
                    if time.monotonic() > t_fin:
                        return best
                    md2 = dict(md1)
                    md2[0] = max(md2.get(0, 0.0), base[0] + d0)
                    m2 = L.mk(inst, route, md2)
                    if m2 is None:
                        break
                    cand = list(routes)
                    cand[ridx] = m2
                    c = _obj(inst, L.Sol(cand, pol))
                    if c < cost - EPS and (best is None or c < best[0]):
                        best = (c, cand)
                # (b) mover espera + mover otra espera
                for q in range(1, k + 1):
                    if q == p:
                        continue
                    for b in _valores_espera(inst, m1, q):
                        if time.monotonic() > t_fin:
                            return best
                        md3 = dict(m1["min_dep"])
                        md3[q] = max(md3.get(q, 0.0), m1["sch"]["dep"][q] + b)
                        m3 = L.mk(inst, route, md3)
                        if m3 is None:
                            continue
                        cand = list(routes)
                        cand[ridx] = m3
                        c = _obj(inst, L.Sol(cand, pol))
                        if c < cost - EPS and (best is None or c < best[0]):
                            best = (c, cand)
        if best is not None:
            return best
    return best


def _pasada_I_horarios(inst, sol, cost, lls, t_fin, max_d=400):
    """Descenso sobre las esperas de las rutas. Para cada ruta y cada posición
    (0 = salida del depósito, en orden aleatorio) prueba los retrasos enteros de la
    salida de ese nodo hasta que la ruta deja de ser factible, con la evaluación
    completa, y se queda con el mejor. Cuando ninguna posición mejora, prueba pares
    de cambios de horario. Las esperas útiles no siempre coinciden con eventos de
    otros vehículos, por eso se prueba cada valor."""
    rng = lls.rng
    routes = list(sol.routes)
    mejoro = False
    seguir = True
    while seguir and time.monotonic() < t_fin:
        seguir = False
        slots = [(r, p) for r, R in enumerate(routes) if R["route"]
                 for p in range(len(R["route"]) + 1)]
        rng.shuffle(slots)
        for ridx, p in slots:
            if time.monotonic() > t_fin:
                break
            R = routes[ridx]
            if p > len(R["route"]):
                continue
            base = R["sch"]["dep"][p]
            best = None
            for d in range(1, max_d + 1):
                if time.monotonic() > t_fin:      # cada evaluación es cara
                    break
                md = dict(R["min_dep"])
                md[p] = max(md.get(p, 0.0), base + d)
                m = L.mk(inst, R["route"], md)
                if m is None:
                    break              # más retraso tampoco es factible
                cand = list(routes)
                cand[ridx] = m
                c = _obj(inst, L.Sol(cand, sol.pol))
                if c < cost - EPS and (best is None or c < best[0]):
                    best = (c, cand)
            if best is not None:
                cost, routes = best
                mejoro = seguir = True
        if not seguir and time.monotonic() < t_fin:
            res = _pares_de_esperas(inst, sol.pol, routes, cost, t_fin)
            if res is not None:
                cost, routes = res
                mejoro = seguir = True
    if mejoro:
        sol = L.Sol(routes, sol.pol)
    return sol, cost, mejoro


def _ordenes(inst, pares, k=80, max_nodos=100000):
    """Los k órdenes factibles de menor distancia que visitan exactamente los pares
    dados (búsqueda en profundidad con poda por ventanas, capacidad, T y por la
    k-ésima mejor distancia encontrada)."""
    import heapq
    P, D = inst.P, inst.D
    mejores = []                     # heap de (-distancia, contador, ruta)
    cnt = [0, 0]

    def dfs(route, t, loc, load, dist, abiertos, pendientes):
        cnt[0] += 1
        if cnt[0] > max_nodos:
            return
        if len(mejores) >= k and dist + D[loc]["LE"] >= -mejores[0][0]:
            return
        if not abiertos and not pendientes:
            if t + D[loc]["LE"] <= inst.T + EPS:
                cnt[1] += 1
                item = (-(dist + D[loc]["LE"]), cnt[1], list(route))
                if len(mejores) < k:
                    heapq.heappush(mejores, item)
                else:
                    heapq.heapreplace(mejores, item)
            return
        for n in list(pendientes) + [p + P for p in abiertos]:
            l2 = inst.loc[n]
            arr = t + D[loc][l2]
            if not route:
                arr = max(arr, inst.a[n])
            sv = max(arr, inst.a[n])
            if sv > inst.b[n] + EPS or load + inst.q[n] > inst.Q:
                continue
            route.append(n)
            if n < P:
                dfs(route, sv + inst.s[n], l2, load + inst.q[n], dist + D[loc][l2],
                    abiertos | {n}, pendientes - {n})
            else:
                dfs(route, sv + inst.s[n], l2, load + inst.q[n], dist + D[loc][l2],
                    abiertos - {n - P}, pendientes)
            route.pop()

    dfs([], 0.0, "LS", 0, 0.0, frozenset(), frozenset(pares))
    return [r for _, _, r in sorted(mejores, key=lambda x: -x[0])]


# ----------------------------------------------------------------------------
# Fase 3b: cambios de rutas evaluados con horarios ajustados (K)
# ----------------------------------------------------------------------------
def _pasada_K_rutas_con_horario(inst, sol, cost, lls, t_fin, top=6, t_i=0.5,
                                n_ordenes=80, max_pares_orden=8):
    """Hay cambios de rutas que solo convienen con esperas bien puestas: con los
    horarios más tempranos parecen peores y ninguna otra pasada los acepta. Esta
    pasada genera cambios de rutas (variantes dentro de cada ruta, los
    n_ordenes órdenes más cortos de cada ruta con hasta max_pares_orden pares, y
    relocate de cada par a sus mejores rutas destino), los ordena por su costo con horarios
    tempranos, y a los que no quedan a más de 1,5 * max(w1, w2) del costo actual les
    ajusta los horarios con la pasada I (t_i segundos cada uno). Acepta el primero
    que, ya ajustado, baja el objetivo."""
    margen = 1.5 * max(inst.w1, inst.w2)
    cands = []
    for ridx, R in enumerate(sol.routes):
        if time.monotonic() > t_fin:
            break
        Lr = R["route"]
        vistos, ms = set(), []
        for new in _variantes_intra(Lr):
            t = tuple(new)
            if new == Lr or t in vistos:
                continue
            vistos.add(t)
            m = L.mk(inst, new)
            if m is not None:
                ms.append(m)
        for m in sorted(ms, key=lambda m: m["rc"])[:top]:
            rs = list(sol.routes)
            rs[ridx] = m
            cands.append(rs)
        pares = [i for i in Lr if i < inst.P]
        if len(pares) <= max_pares_orden:
            for new in _ordenes(inst, pares, n_ordenes):
                if new != Lr and tuple(new) not in vistos:
                    m = L.mk(inst, new)
                    if m is not None:
                        rs = list(sol.routes)
                        rs[ridx] = m
                        cands.append(rs)
    for p, ridx in lls._all_pairs(sol.routes):
        if time.monotonic() > t_fin:
            break
        base = lls._remove(sol.routes, {p})
        if base is None:
            continue
        gone = len(base) < len(sol.routes)
        for opt in lls._options(base, p, () if gone else (ridx,), allow_new=not gone)[:2]:
            tmp = list(base)
            if lls._apply_opt(tmp, opt):
                cands.append(tmp)
    evaluados = []
    for rs in cands:
        if time.monotonic() > t_fin:
            break
        s = L.Sol(rs, sol.pol)
        c = _obj(inst, s)
        if c < cost + margen:
            evaluados.append((c, len(evaluados), s))
    evaluados.sort(key=lambda t: (t[0], t[1]))
    for c, _, s in evaluados:
        if time.monotonic() > t_fin:
            break
        if c < cost - EPS:
            return s, c, True
        s2, c2, _ = _pasada_I_horarios(inst, s, c, lls, min(t_fin, time.monotonic() + t_i))
        if c2 < cost - EPS:
            return s2, c2, True
    return sol, cost, False


# ----------------------------------------------------------------------------
# Fase 4: búsqueda con el tiempo restante (J)
# ----------------------------------------------------------------------------
def _busqueda_ll(inst, sol, cost, lls, t_fin, paciencia, kmax=3):
    """Secuencias de 1 a kmax LLs de la MAHH elegidas al azar. Se acepta si el
    objetivo baja, o si queda igual (los empates dejan desplazarse por mesetas sin
    empeorar nunca). Se detiene si pasan `paciencia` intentos sin mejora estricta."""
    mejoro = False
    sin_mejora = 0
    while time.monotonic() < t_fin and sin_mejora < paciencia:
        s = sol
        for _ in range(lls.rng.randint(1, kmax)):
            s = lls.apply(s, lls.rng.randrange(L.H))
        if L.same_solution(s, sol):    # las LLs no cambiaron nada: mismo costo
            sin_mejora += 1
            continue
        c = _obj(inst, s)
        if c < cost - EPS:
            sol, cost, mejoro, sin_mejora = s, c, True, 0
        else:
            sin_mejora += 1
            if c <= cost + EPS:
                sol = s
    return sol, cost, mejoro


# ----------------------------------------------------------------------------
# Orquestador
# ----------------------------------------------------------------------------
def pulir(inst, sol, time_limit=60.0, pp_frac=0.10, seed=0,
          min_seconds=1.0, max_seconds=15.0, verbose=False, paciencia=5000,
          h_seconds=0.0, cost0=None):
    """Devuelve (Sol mejorada o la de entrada, info). Nunca empeora.
    Presupuesto: max(min_seconds, min(max_seconds, time_limit * pp_frac)).
    h_seconds: tiempo EXTRA (fuera de ese presupuesto) para una pasada H dedicada
    de la política de tripulaciones, al inicio y una sola vez. Con h_seconds > 0 las
    rondas siguientes no vuelven a hacer H; con 0 se comporta como antes (H dentro
    de cada ronda). Si la asignación exacta aplica, H no cambia nada y no se usa.
    cost0: costo ya conocido de `sol` (evita reevaluarla al partir)."""
    t0 = time.monotonic()
    presupuesto = max(min_seconds, min(max_seconds, time_limit * pp_frac))
    t_fin = t0 + presupuesto + max(0.0, h_seconds)

    entrada = sol
    best = sol.copy()               # evaluate puede cambiar sol.routes: no tocar la entrada
    cost0 = best_cost = (_obj(inst, best) if cost0 is None else cost0)
    lls = L.LowLevelHeuristics(inst, random.Random(seed))
    log, mejoras, ronda = [], 0, 0

    def registrar(nombre, antes, despues, it):
        nonlocal mejoras
        if antes - despues > EPS:
            log.append((nombre, antes - despues, it))
            mejoras += 1
            if verbose:
                print("  PP %s ronda %d: %.2f -> %.2f" % (nombre, it, antes, despues))

    def restante():
        return max(0.0, t_fin - time.monotonic())

    if h_seconds > 0 and _politica_importa(inst, best):
        antes = best_cost
        best, best_cost, _ = _pasada_H_politica(
            inst, best, best_cost, lls, time.monotonic() + h_seconds)
        registrar("H", antes, best_cost, 0)

    while restante() > 0:
        ronda += 1
        # fracciones del tiempo restante: la primera ronda explora; las siguientes (tras
        # una mejora de la búsqueda J) solo repasan, para dejar tiempo a J
        f_pul, f_h, f_k, f_i = (0.35, 0.15, 0.7, 0.20) if ronda == 1 else (0.25, 0.05, 0.3, 0.08)
        # 1. pulido determinista hasta convergencia
        best, best_cost, _ = _pulido(inst, best, best_cost, lls,
                                     time.monotonic() + f_pul * restante(), registrar, ronda)
        # 2. política de tripulaciones, salvo que la asignación exacta aplique
        if h_seconds <= 0 and restante() > 0 and _politica_importa(inst, best):
            antes = best_cost
            best, best_cost, _ = _pasada_H_politica(
                inst, best, best_cost, lls, time.monotonic() + f_h * restante())
            registrar("H", antes, best_cost, ronda)
        # 3. cambios de rutas evaluados con horarios ajustados (K)
        if restante() > 0:
            antes = best_cost
            best, best_cost, okk = _pasada_K_rutas_con_horario(
                inst, best, best_cost, lls, time.monotonic() + f_k * restante(), t_i=0.3)
            registrar("K", antes, best_cost, ronda)
            if okk:
                continue                 # mejoró: otra ronda desde el pulido
        # 3b. horarios (I)
        if restante() > 0:
            antes = best_cost
            best, best_cost, _ = _pasada_I_horarios(
                inst, best, best_cost, lls, time.monotonic() + f_i * restante())
            registrar("I", antes, best_cost, ronda)
        # 4. búsqueda con LLs en el tiempo restante (en tramos: si mejora, se vuelve a pulir)
        if restante() > 0:
            antes = best_cost
            best, best_cost, mejoro = _busqueda_ll(
                inst, best, best_cost, lls, time.monotonic() + 0.5 * restante(), paciencia)
            registrar("J", antes, best_cost, ronda)
            if not mejoro and restante() > 0:
                # sin mejora en este tramo: todo el tiempo que queda a la búsqueda
                antes = best_cost
                best, best_cost, mejoro = _busqueda_ll(
                    inst, best, best_cost, lls, t_fin, paciencia)
                registrar("J", antes, best_cost, ronda)
                if not mejoro:
                    break

    # verificación final
    errs = L.final_outputs(inst, best)[3]
    info = {"tiempo_s": time.monotonic() - t0, "obj_inicial": cost0,
            "pasadas": log, "iteraciones": ronda}
    if errs:
        info.update(mejoras=0, obj_final=cost0, nota="revertido: " + "; ".join(errs[:2]))
        return entrada, info
    info.update(mejoras=mejoras, obj_final=best_cost, nota="")
    return best, info