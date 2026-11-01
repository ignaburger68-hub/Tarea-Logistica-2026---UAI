
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solucion_inicial as J

EPS = 1e-9
H = 34
CREW_PROXY_W = 1.0
crews_needed = J.crews_needed



# Programación de rutas
def sched(inst, route, md=None):
    k = len(route)
    if k == 0:
        return None
    D, P = inst.D, inst.P
    locs = ["LS"] + [inst.loc[i] for i in route] + ["LE"]
    n = k + 2
    arr = [0.0] * n
    serv = [0.0] * n
    dep = [0.0] * n
    tS = inst.a[route[0]] - D["LS"][locs[1]]
    if tS < 0.0:
        tS = 0.0
    if md:
        m0 = md.get(0)
        if m0 is not None and m0 > tS:
            tS = m0
    arr[0] = serv[0] = dep[0] = tS
    load = 0
    vd = 0.0
    seen = set()
    for p in range(1, k + 1):
        i = route[p - 1]
        d = D[locs[p - 1]][locs[p]]
        vd += d
        arr[p] = dep[p - 1] + d
        sv = arr[p] if arr[p] > inst.a[i] else inst.a[i]
        if sv > inst.b[i] + EPS:
            return None
        if i >= P and (i - P) not in seen:
            return None
        seen.add(i)
        load += inst.q[i]
        if load > inst.Q or load < 0:
            return None
        serv[p] = sv
        dp = sv + inst.s[i]
        if md:
            m = md.get(p)
            if m is not None and m > dp:
                dp = m
        dep[p] = dp
    d = D[locs[n - 2]][locs[n - 1]]
    vd += d
    arr[n - 1] = serv[n - 1] = dep[n - 1] = dep[n - 2] + d
    if arr[n - 1] > inst.T + EPS:
        return None
    return {"route": list(route), "locs": locs, "arr": arr, "serv": serv,
            "dep": dep, "vdist": vd}


def mk(inst, route, md=None):
    """Construye el dict de una ruta (o None si es infactible)."""
    sch = sched(inst, route, md)
    if sch is None:
        return None
    J.add_visits(inst, sch)
    return {"route": list(route), "min_dep": dict(md) if md else {}, "sch": sch,
            "rc": inst.w3 * sch["vdist"] + inst.w2 * CREW_PROXY_W * crews_needed(inst, sch)}


def empty_route():
    return {"route": [], "min_dep": {}, "sch": {"visits": []}, "rc": 0.0}



def default_pol():
    return {"prefer_handoff": False, "fit": "best", "shuffle": False,
            "seed": 0, "forced": set()}


def assign_crews_v(inst, schs, pol):
    MD = inst.MD
    rng = random.Random(pol.get("seed", 0))
    shuffle = pol.get("shuffle", False)
    prefer_h = pol.get("prefer_handoff", False)
    fit = pol.get("fit", "best")
    forced = pol.get("forced", ())
    vkey = [rng.random() if shuffle else 0.0 for _ in schs]

    legs = []
    for v, sch in enumerate(schs):
        vis = sch["visits"]
        for m in range(len(vis) - 1):
            legs.append((vis[m]["exit"], vkey[v], v, m))
    legs.sort()

    crews = []
    onboard = {}
    leg_driver = {}
    bad = 0

    def ok(c, dep, arr):
        ds = crews[c]["ds"]
        return arr - (dep if ds is None else ds) <= MD + EPS

    def span_of(c, dep, arr):
        ds = crews[c]["ds"]
        return arr - (dep if ds is None else ds)

    def pick(cands):
        if not cands:
            return None
        if fit == "worst":
            return min(cands, key=lambda t: t[0])
        if fit == "random":
            return cands[rng.randrange(len(cands))]
        return max(cands, key=lambda t: t[0])

    for dep, _, v, m in legs:
        vis = schs[v]["visits"]
        arr = vis[m + 1]["entry"]
        loc = vis[m]["loc"]
        pool = onboard.setdefault((v, m), [])

        def handoff():
            res = []
            for (w, mp), lst in onboard.items():
                if (w, mp) == (v, m) or not lst:
                    continue
                wv = schs[w]["visits"]
                if mp >= len(wv) - 1 or wv[mp]["loc"] != loc:
                    continue
                if wv[mp]["entry"] > dep + EPS:
                    continue
                wdep, warr = wv[mp]["exit"], wv[mp + 1]["entry"]
                feas_w = [c for c in lst if ok(c, wdep, warr)]
                for c in lst:
                    if not ok(c, dep, arr):
                        continue
                    if c in feas_w and len(feas_w) == 1:
                        continue
                    res.append((span_of(c, dep, arr), c, (w, mp)))
            return res

        ca = []
        if (v, m) not in forced:
            ca = [(span_of(c, dep, arr), c, None) for c in pool if ok(c, dep, arr)]
        if prefer_h:
            chosen = pick(handoff())
            if chosen is None:
                chosen = pick(ca)
        else:
            chosen = pick(ca)
            if chosen is None:
                chosen = pick(handoff())

        if chosen is None:
            crews.append({"rides": [[v, 0, m]], "ds": None, "de": None})
            c = len(crews) - 1
            pool.append(c)
            if arr - dep > MD + EPS:
                bad += 1
        else:
            _, c, src = chosen
            if src is not None:
                onboard[src].remove(c)
                crews[c]["rides"].append([v, m, m])
                pool.append(c)

        if crews[c]["ds"] is None:
            crews[c]["ds"] = dep
        crews[c]["de"] = arr
        leg_driver[(v, m)] = c

        del onboard[(v, m)]
        nxt = onboard.setdefault((v, m + 1), [])
        for c2 in pool:
            crews[c2]["rides"][-1][2] = m + 1
            nxt.append(c2)

    cdist = 0.0
    for cd in crews:
        for v, a, b in cd["rides"]:
            cdist += sum(schs[v]["legd"][a:b])
    return {"crews": crews, "leg_driver": leg_driver, "ncrews": len(crews),
            "cdist": cdist, "bad": bad}


# Asignación exacta de tripulaciones
EXACT_MAX_LEGS = 22        # sobre este número de tramos se usa solo el voraz
EXACT_MAX_STATES = 200000  # tope de estados explorados por evaluación
EXACT_MAX_MEMO = 2000000   # tope de subconjuntos en la partición (memoria)
EXACT_EXT_LEGS = 45        # salida final: tramos máximos
EXACT_EXT_STATES = 400000  # salida final: tope de estados
COVER_EN_BUSQUEDA = False  # True: cobertura exacta en cada evaluación
COVER_MAX_LEGS = 200           # cobertura exacta: tramos máximos
COVER_MAX_STATES = 2000000     # cobertura exacta: tope de estados
COVER_MAX_NODOS = 200000       # nodos de la búsqueda en la salida final
COVER_NODOS_BUSQUEDA = 1000    # nodos en cada evaluación (COVER_EN_BUSQUEDA)
EXACT_CACHE_MAX = 50000        # resultados guardados en caché
_EXACT_CACHE = {}


def exact_crews(inst, schs, max_legs=None, max_states=None):
    """Asignación óptima de tripulaciones para horarios fijos (hasta
    max_legs tramos)."""
    max_legs = EXACT_MAX_LEGS if max_legs is None else max_legs
    max_states = EXACT_MAX_STATES if max_states is None else max_states
    if sum(len(s["visits"]) - 1 for s in schs) > max_legs:
        return None                 # demasiado grande: no se usa (ni se guarda)
    key = (inst.name, inst.MD, inst.w2, inst.w4, max_legs, max_states,
           tuple(tuple((x["loc"], x["entry"], x["exit"]) for x in s["visits"]) for s in schs))
    if key in _EXACT_CACHE:
        return _EXACT_CACHE[key]
    res = _exact_crews(inst, schs, max_legs, max_states)
    if len(_EXACT_CACHE) >= EXACT_CACHE_MAX:
        _EXACT_CACHE.clear()
    _EXACT_CACHE[key] = res
    return res


def _itinerarios(inst, schs, max_legs, max_states):
    """Enumera los itinerarios posibles de una tripulación y guarda el más
    corto por conjunto de tramos conducidos."""
    MD = inst.MD
    vis = [s["visits"] for s in schs]
    legd = [s["legd"] for s in schs]
    legs = [(v, m) for v in range(len(vis)) for m in range(len(vis[v]) - 1)]
    if not legs or len(legs) > max_legs:
        return None
    lid = {l: i for i, l in enumerate(legs)}
    byloc = defaultdict(list)
    for v, vv in enumerate(vis):
        for k, x in enumerate(vv):
            byloc[x["loc"]].append((v, k))
    best = {}
    seen = {}          # estado -> menor distancia con que se llegó a él
    states = [0]

    def dfs(v, k, dist, mask, ds, just_t, path):
        key = (v, k, mask, ds, just_t)
        old = seen.get(key)
        if old is not None and old <= dist + EPS:
            return
        seen[key] = dist
        states[0] += 1
        if states[0] > max_states:
            raise OverflowError
        if k == len(vis[v]) - 1 and mask:
            if mask not in best or dist < best[mask][0]:
                best[mask] = (dist, list(path))
        if k + 1 < len(vis[v]):
            dep, arr = vis[v][k]["exit"], vis[v][k + 1]["entry"]
            path.append(("pas", v, k))
            dfs(v, k + 1, dist + legd[v][k], mask, ds, False, path)
            path.pop()
            nds = dep if ds is None else ds
            if arr - nds <= MD + EPS:
                path.append(("drv", v, k))
                dfs(v, k + 1, dist + legd[v][k], mask | (1 << lid[(v, k)]), nds, False, path)
                path.pop()
        if not just_t and vis[v][k]["loc"] not in ("LS", "LE"):
            t = vis[v][k]["entry"]
            for (u, j) in byloc[vis[v][k]["loc"]]:
                if (u, j) != (v, k) and vis[u][j]["exit"] >= t - EPS and (u != v or j > k):
                    path.append(("trans", u, j))
                    dfs(u, j, dist, mask, ds, True, path)
                    path.pop()

    try:
        for v in range(len(vis)):
            dfs(v, 0, 0.0, 0, None, False, [("start", v, 0)])
    except OverflowError:
        return None
    return legs, best, vis


def _cr_desde_plan(plan, vis):
    """Arma el cr (mismo formato que assign_crews_v) a partir de los
    itinerarios elegidos."""
    crews, leg_driver, cdist = [], {}, 0.0
    for m, d, path in plan:
        rides = []
        ds = de = None
        for kind, v, k in path:
            if kind in ("start", "trans"):
                rides.append([v, k, k])
            else:
                rides[-1][2] = k + 1
                if kind == "drv":
                    leg_driver[(v, k)] = len(crews)
                    if ds is None:
                        ds = vis[v][k]["exit"]
                    de = vis[v][k + 1]["entry"]
        # unir tramos de pasajero/conducción contiguos
        merged = []
        for r in rides:
            if merged and merged[-1][0] == r[0] and merged[-1][2] == r[1]:
                merged[-1][2] = r[2]
            else:
                merged.append(r)
        crews.append({"rides": merged, "ds": ds, "de": de})
        cdist += d
    return {"crews": crews, "leg_driver": leg_driver, "ncrews": len(crews),
            "cdist": cdist, "bad": 0}


def _exact_crews(inst, schs, max_legs, max_states):
    """Partición óptima de los tramos entre itinerarios, por programación
    dinámica sobre subconjuntos (rápida hasta ~22 tramos)."""
    res = _itinerarios(inst, schs, max_legs, max_states)
    if res is None:
        return None
    legs, best, vis = res
    by_low = [[] for _ in legs]
    for m, (d, p) in sorted(best.items()):
        by_low[(m & -m).bit_length() - 1].append((m, (d, p)))
    memo = {0: (0.0, None)}           # subconjunto -> (costo, itinerario elegido)

    def g(mask):
        if mask in memo:
            return memo[mask][0]
        if len(memo) > EXACT_MAX_MEMO:
            raise OverflowError
        low = (mask & -mask).bit_length() - 1
        res = (float("inf"), None)
        for m, (d, p) in by_low[low]:
            if (m & mask) == m:
                c = g(mask ^ m) + inst.w2 + inst.w4 * d
                if c < res[0]:
                    res = (c, (m, d, p))
        memo[mask] = res
        return res[0]

    full = (1 << len(legs)) - 1
    try:
        cost = g(full)
    except OverflowError:
        return None
    if cost == float("inf"):
        return None
    plan, mask = [], full
    while mask:
        m, d, p = memo[mask][1]
        plan.append((m, d, tuple(p)))
        mask ^= m
    return _cr_desde_plan(plan, vis)


def crews_cover(inst, schs, max_legs=None, max_states=None, max_nodos=None, cota_sup=None):
    """Asignación óptima de tripulaciones como cobertura exacta, por
    ramificación y acotamiento."""
    max_legs = COVER_MAX_LEGS if max_legs is None else max_legs
    max_states = COVER_MAX_STATES if max_states is None else max_states
    max_nodos = COVER_MAX_NODOS if max_nodos is None else max_nodos
    if sum(len(s["visits"]) - 1 for s in schs) > max_legs:
        return None
    key = ("cover", inst.name, inst.MD, inst.w2, inst.w4, max_legs, max_states, max_nodos,
           tuple(tuple((x["loc"], x["entry"], x["exit"]) for x in s["visits"]) for s in schs))
    if key in _EXACT_CACHE:
        return _EXACT_CACHE[key]
    res = _itinerarios(inst, schs, max_legs, max_states)
    cr = None
    if res is not None:
        legs, best, vis = res
        nL = len(legs)
        cols = [(m, inst.w2 + inst.w4 * d, d, tuple(p)) for m, (d, p) in best.items()]
        cubre = [[] for _ in range(nL)]          # tramo -> columnas que lo cubren
        for c in cols:
            m, b = c[0], 0
            while m:
                if m & 1:
                    cubre[b].append(c)
                m >>= 1
                b += 1
        for lst in cubre:
            lst.sort(key=lambda c: c[1] / bin(c[0]).count("1"))
        # costo mínimo por tramo de cada tramo (para la cota inferior)
        minpt = [min((c[1] / bin(c[0]).count("1") for c in lst), default=float("inf"))
                 for lst in cubre]
        if all(lst for lst in cubre):
            mejor = [float("inf") if cota_sup is None else cota_sup + EPS, None]
            nodos = [0]
            full = (1 << nL) - 1

            def cota(libres):
                lb, b = 0.0, 0
                while libres:
                    if libres & 1:
                        lb += minpt[b]
                    libres >>= 1
                    b += 1
                return lb

            def dfs(cubiertos, costo, elegidas):
                nodos[0] += 1
                if nodos[0] > max_nodos:
                    raise OverflowError
                if cubiertos == full:
                    if costo < mejor[0] - EPS:
                        mejor[0], mejor[1] = costo, list(elegidas)
                    return
                libres = full & ~cubiertos
                if costo + cota(libres) >= mejor[0] - EPS:
                    return
                # tramo libre con menos opciones compatibles
                opciones = None
                b, x = 0, libres
                while x:
                    if x & 1:
                        ops = [c for c in cubre[b] if not (c[0] & cubiertos)]
                        if not ops:
                            return
                        if opciones is None or len(ops) < len(opciones):
                            opciones = ops
                            if len(ops) == 1:
                                break
                    x >>= 1
                    b += 1
                for c in opciones:
                    elegidas.append(c)
                    dfs(cubiertos | c[0], costo + c[1], elegidas)
                    elegidas.pop()

            try:
                dfs(0, 0.0, [])
            except OverflowError:
                pass
            except RecursionError:
                pass
            if mejor[1] is not None:
                cr = _cr_desde_plan([(c[0], c[2], c[3]) for c in mejor[1]], vis)
    if len(_EXACT_CACHE) >= EXACT_CACHE_MAX:
        _EXACT_CACHE.clear()
    _EXACT_CACHE[key] = cr
    return cr

def _latest_dep_sch(inst, route, sch, a_pos, b_pos):
    """Salida más tardía desde el nodo a_pos que mantiene la llegada al nodo
    b_pos."""
    locs = sch["locs"]
    t = sch["arr"][b_pos]
    for p in range(b_pos - 1, a_pos - 1, -1):
        latest = t - inst.D[locs[p]][locs[p + 1]]
        if p == a_pos:
            return latest
        i = route[p - 1]
        sv = min(inst.b[i], latest - inst.s[i])
        if sv < inst.a[i] - EPS:
            return None
        t = sv
    return None


def _passenger_paths(vis, legd):
    """Caminos de pasajero dentro de un mismo vehículo (puede bajarse y
    esperar una visita posterior en la misma ubicación)."""
    K = len(vis) - 1
    INF = float("inf")
    pre, pp = [INF] * (K + 1), [None] * (K + 1)
    pre[0] = 0.0
    for k in range(K + 1):
        if k > 0 and pre[k - 1] + legd[k - 1] < pre[k] - 1e-12:
            pre[k], pp[k] = pre[k - 1] + legd[k - 1], ("ride", k - 1)
        for j in range(k + 1, K + 1):
            if vis[j]["loc"] == vis[k]["loc"] and pre[k] < pre[j] - 1e-12:
                pre[j], pp[j] = pre[k], ("wait", k)
    post, pn = [INF] * (K + 1), [None] * (K + 1)
    post[K] = 0.0
    for k in range(K, -1, -1):
        if k < K and post[k + 1] + legd[k] < post[k] - 1e-12:
            post[k], pn[k] = post[k + 1] + legd[k], ("ride", k + 1)
        for j in range(k + 1, K + 1):
            if vis[j]["loc"] == vis[k]["loc"] and post[j] < post[k] - 1e-12:
                post[k], pn[k] = post[j], ("wait", j)
    return pre, pp, post, pn


def _rides_for_block(v, a, b, pp, pn):
    """Viajes de una tripulación que conduce los tramos a..b-1 del vehículo
    v."""
    k = a                              # ida: desde la visita 0 hasta a
    back = []
    while k != 0:
        kind, j = pp[k]
        back.append((kind, j, k))
        k = j
    rides = [[v, 0, 0]]
    for kind, j, k in reversed(back):
        if kind == "ride":
            rides[-1][2] = k
        else:
            rides.append([v, k, k])
    rides[-1][2] = b                   # conduce de a hasta b
    k = b                              # vuelta: desde b hasta la última visita
    while pn[k] is not None:
        kind, j = pn[k]
        if kind == "ride":
            rides[-1][2] = j
        else:
            rides.append([v, j, j])
        k = j
    return rides


def dp_decode(inst, routes):
    """Asignación por programación dinámica por vehículo, con retenciones si
    un turno excede MaxDrivingTime."""
    MD, w2, w4 = inst.MD, inst.w2, inst.w4
    new_routes, crews, leg_driver = [], [], {}
    cdist = 0.0
    for v, R in enumerate(routes):
        sch, route = R["sch"], R["route"]
        vis, legd = sch["visits"], sch["legd"]
        K = len(vis) - 1
        pre, pp, post, pn = _passenger_paths(vis, legd)
        cum = [0.0]
        for d in legd:
            cum.append(cum[-1] + d)
        INF = float("inf")
        f, arg = [INF] * (K + 1), [None] * (K + 1)
        f[0] = 0.0
        for a in range(K):
            if f[a] == INF:
                continue
            for b in range(a + 1, K + 1):
                X = None
                if vis[b]["entry"] - vis[a]["exit"] > MD + EPS:
                    X = _latest_dep_sch(inst, route, sch, vis[a]["i1"], vis[b]["i0"])
                    if X is None or vis[b]["entry"] - X > MD + EPS:
                        if b > a + 1 and X is None:
                            break
                        continue
                c = f[a] + w2 + w4 * (pre[a] + cum[b] - cum[a] + post[b])
                if c < f[b] - 1e-9:
                    f[b], arg[b] = c, (a, X)
        if f[K] == INF:
            return None
        blocks, b = [], K
        while b > 0:
            a, X = arg[b]
            blocks.append((a, b, X))
            b = a
        blocks.reverse()
        md = dict(R["min_dep"])
        for a, b, X in blocks:
            if X is not None:
                pos = vis[a]["i1"]
                md[pos] = max(md.get(pos, 0.0), X)
        R2 = mk(inst, route, md) if md != R["min_dep"] else R
        if R2 is None:
            return None
        vis2 = R2["sch"]["visits"]
        for a, b, X in blocks:
            if vis2[b]["entry"] - vis2[a]["exit"] > MD + EPS:
                return None
            rides = _rides_for_block(v, a, b, pp, pn)
            c = len(crews)
            crews.append({"rides": rides, "ds": vis2[a]["exit"], "de": vis2[b]["entry"]})
            for m in range(a, b):
                leg_driver[(v, m)] = c
            cdist += sum(sum(legd[x:y]) for _, x, y in rides)
        new_routes.append(R2)
    return new_routes, {"crews": crews, "leg_driver": leg_driver, "ncrews": len(crews),
                        "cdist": cdist, "bad": 0}


# Solución, evaluación y solución inicial
class Sol:
    __slots__ = ("routes", "pol")

    def __init__(self, routes, pol=None):
        self.routes = routes
        self.pol = pol if pol is not None else default_pol()

    def copy(self):
        pol = dict(self.pol)
        pol["forced"] = set(self.pol.get("forced", ()))
        return Sol(list(self.routes), pol)  # los dicts de ruta no se mutan


def decode(inst, sol, safe=False, ampliada=False):
    """Calcula las tripulaciones (voraz, exacta y programación dinámica) y
    devuelve (costo, cr) de la mejor."""
    schs = [r["sch"] for r in sol.routes]
    cr = assign_crews_v(inst, schs, sol.pol)
    if not cr["bad"]:  # acortar trayectos como pasajero (nunca empeora)
        crs = J.shorten_passengers(schs, cr)
        if not safe or not J.verify(inst, schs, crs):
            cr = crs
    cost = J.objective(inst, schs, cr) + (1e9 if cr["bad"] else 0.0)
    best_routes = None
    ex = exact_crews(inst, schs)          # hasta EXACT_MAX_LEGS tramos
    if ex is None and (ampliada or COVER_EN_BUSQUEDA):
        ub = None if cr["bad"] else inst.w2 * cr["ncrews"] + inst.w4 * cr["cdist"]
        ex = crews_cover(inst, schs, cota_sup=ub,
                         max_nodos=COVER_MAX_NODOS if ampliada else COVER_NODOS_BUSQUEDA)
    if ex is None and ampliada:
        ex = exact_crews(inst, schs, EXACT_EXT_LEGS, EXACT_EXT_STATES)
    if ex is not None and (not safe or not J.verify(inst, schs, ex)):
        c2 = J.objective(inst, schs, ex)
        if c2 < cost - 1e-9:
            cost, cr = c2, ex
    out = dp_decode(inst, sol.routes)
    if out is not None:
        rs, cr3 = out
        schs3 = [r["sch"] for r in rs]
        crs3 = J.shorten_passengers(schs3, cr3)
        if crs3["cdist"] < cr3["cdist"] - 1e-9 and (not safe or not J.verify(inst, schs3, crs3)):
            cr3 = crs3
        changed = any(a is not b for a, b in zip(rs, sol.routes))
        ex3 = exact_crews(inst, schs3) if changed else None   # exacto con las retenciones
        if changed and ex3 is None and (ampliada or COVER_EN_BUSQUEDA):
            ex3 = crews_cover(inst, schs3, cota_sup=inst.w2 * cr3["ncrews"] + inst.w4 * cr3["cdist"],
                              max_nodos=COVER_MAX_NODOS if ampliada else COVER_NODOS_BUSQUEDA)
        if changed and ex3 is None and ampliada:
            ex3 = exact_crews(inst, schs3, EXACT_EXT_LEGS, EXACT_EXT_STATES)
        if ex3 is not None and J.objective(inst, schs3, ex3) < J.objective(inst, schs3, cr3) - 1e-9 \
                and (not safe or not J.verify(inst, schs3, ex3)):
            cr3 = ex3
        c3 = J.objective(inst, schs3, cr3)
        if c3 < cost - 1e-9 and (not safe or not J.verify(inst, schs3, cr3)):
            cost, cr, best_routes = c3, cr3, rs
    if best_routes is not None:
        sol.routes = best_routes
    return cost, cr


def evaluate(inst, sol):
    return decode(inst, sol)


def same_solution(a, b):
    """True si a y b tienen las mismas rutas y la misma política."""
    return (len(a.routes) == len(b.routes)
            and all(x is y for x, y in zip(a.routes, b.routes))
            and a.pol == b.pol)

def initial_solution(inst):
    """Solución inicial: rutas por inserción más barata."""
    return Sol([mk(inst, r) for r in J.construct_routes(inst)])


# Las 34 heurísticas de bajo nivel 
NAMES = [
    "01 Relocate", "02 Swap de pares", "03 Elim. aleatoria + voraz",
    "04 Shaw + regret-2", "05 Elim. por ubicación", "06 Elim. de los peores",
    "07 Vaciar ruta", "08 Fusión de rutas", "09 Or-opt", "10 Intercambio de colas",
    "11 Swap intra-ruta", "12 Inversión de segmento", "13 Retraso para sincronizar",
    "14 Quitar una espera", "15 Reiniciar ruta", "16 Reiniciar todas las esperas",
    "17 Sincronización múltiple", "18 Desplazar inicio", "19 Priorizar traspasos",
    "20 Orden perturbado", "21 Best/Worst-fit", "22 Corte forzado",
    "23 Selección aleatoria tripul.", "24 Cadena de eyección",
    "25 Elim. por franja de tiempo", "26 Regret-3 insertion", "27 Ruin geográfico",
    "28 Relocate encadenado", "29 Reiniciar política", "30 Retraso máximo de inicio",
    "31 Recortar la cola de una ruta", "32 Atraso conjunto de vehículos",
    "33 Dividir una ruta en dos vehículos", "34 Mover pares y sincronizar",
]


class LowLevelHeuristics:
    def __init__(self, inst, rng, bonus=3.0, slack=5.0):
        self.inst = inst
        self.rng = rng
        self.P = inst.P
        self.bonus = bonus
        self.slack = slack
        P = inst.P
        dvals = [v for row in inst.D.values() for v in row.values()]
        dmax = max(dvals) if dvals and max(dvals) > 0 else 1.0
        tmax = max(inst.b) if max(inst.b) > 0 else 1.0
        self.rel = [[0.0] * P for _ in range(P)]
        for p in range(P):
            for q in range(P):
                self.rel[p][q] = (
                    (inst.D[inst.loc[p]][inst.loc[q]] + inst.D[inst.loc[p + P]][inst.loc[q + P]]) / dmax
                    + (abs(inst.a[p] - inst.a[q]) + abs(inst.a[p + P] - inst.a[q + P])) / tmax
                    + abs(inst.q[p] - inst.q[q]) / max(1, inst.Q))
        self.loc_pairs = defaultdict(list)
        for p in range(P):
            self.loc_pairs[inst.loc[p]].append(p)
            if inst.loc[p + P] != inst.loc[p]:
                self.loc_pairs[inst.loc[p + P]].append(p)
        self.fns = [
            self.h01, self.h02, self.h03, self.h04, self.h05, self.h06, self.h07,
            self.h08, self.h09, self.h10, self.h11, self.h12, self.h13, self.h14,
            self.h15, self.h16, self.h17, self.h18, self.h19, self.h20, self.h21,
            self.h22, self.h23, self.h24, self.h25, self.h26, self.h27, self.h28,
            self.h29, self.h30, self.h31, self.h32, self.h33, self.h34,
        ]

    def apply(self, sol, h):
        s = sol.copy()
        self.fns[h](s)
        s.routes = [r for r in s.routes if r["route"]]
        return s

    def _kmax(self):
        return min(7, max(2, int(0.3 * self.P)))

    def _k(self):
        kmax = min(self._kmax(), self.P)
        return self.rng.randint(2, kmax) if kmax >= 2 else 1

    def _pairs(self, route):
        return [i for i in route if i < self.P]

    def _all_pairs(self, routes):
        return [(p, r) for r, R in enumerate(routes) for p in self._pairs(R["route"])]

    def _without(self, L, p):
        P = self.P
        return [i for i in L if i != p and i != p + P]

    def _remove(self, routes, pset):
        P = self.P
        out = []
        for r in routes:
            if any((i if i < P else i - P) in pset for i in r["route"]):
                new = [i for i in r["route"] if (i if i < P else i - P) not in pset]
                if new:
                    m = mk(self.inst, new)
                    if m is None:
                        return None
                    out.append(m)
            else:
                out.append(r)
        return out

    def _prof(self, R):
        """Perfil de la ruta (tiempos tempranos, holguras) para evaluar
        inserciones."""
        pf = R.get("prof")
        if pf is not None:
            return pf
        inst = self.inst
        route = R["route"]
        k = len(route)
        sch = R["sch"] if not R["min_dep"] else sched(inst, route)
        D, locs = inst.D, sch["locs"]
        arr, serv = sch["arr"], sch["serv"]
        cum = [0] * (k + 2)                 # carga luego del nodo t
        for t in range(1, k + 1):
            cum[t] = cum[t - 1] + inst.q[route[t - 1]]
        lat = [0.0] * (k + 2)               # inicio de servicio más tardío factible
        lat[k + 1] = inst.T
        for t in range(k, 0, -1):
            x = route[t - 1]
            lat[t] = min(inst.b[x], lat[t + 1] - D[locs[t]][locs[t + 1]] - inst.s[x])
        wsuf = [0.0] * (k + 3)              # esperas acumuladas desde el nodo t
        for t in range(k, 0, -1):
            wsuf[t] = wsuf[t + 1] + (serv[t] - arr[t])
        rc0 = inst.w3 * sch["vdist"] + inst.w2 * CREW_PROXY_W * crews_needed(inst, sch)
        pf = (locs, arr, sch["dep"], cum, lat, wsuf, sch["vdist"], sch["dep"][0], rc0)
        R["prof"] = pf
        return pf

    def _route_best(self, ridx, R, p, loc_vis):
        """Mejor inserción del par p en la ruta R."""
        inst, P = self.inst, self.P
        D = inst.D
        a_, b_, s_, q_, loc = inst.a, inst.b, inst.s, inst.q, inst.loc
        route = R["route"]
        k = len(route)
        locs, arr, dep, cum, lat, wsuf, vd0, dep0, rc0 = self._prof(R)
        lp, ld = loc[p], loc[p + P]
        ap, bp, sp, qp = a_[p], b_[p], s_[p], q_[p]
        ad, bd, sd = a_[p + P], b_[p + P], s_[p + P]
        Q, MD, T = inst.Q, inst.MD, inst.T
        w2, w3 = inst.w2 * CREW_PROXY_W, inst.w3
        bon = self.bonus * w3 if (loc_vis is not None and self.bonus > 0) else 0.0
        slack = self.slack
        best = None  # (costo, i, j)

        def has_bonus(l, a, d):
            for (r2, lo, hi) in loc_vis.get(l, ()):
                if r2 != ridx and a - slack <= hi and d + slack >= lo:
                    return True
            return False

        # pickup al inicio: cambia la salida del depósito -> evaluación exacta
        for j in range(k + 1):
            new = [p] + route[:j] + [p + P] + route[j:]
            sch = sched(inst, new)
            if sch is None:
                continue
            cost = w3 * sch["vdist"] + w2 * crews_needed(inst, sch) - rc0
            if bon:
                if has_bonus(lp, sch["arr"][1], sch["dep"][1]):
                    cost -= bon
                if has_bonus(ld, sch["arr"][j + 2], sch["dep"][j + 2]):
                    cost -= bon
            if best is None or cost < best[0]:
                best = (cost, 0, j)

        # pickup después del nodo i >= 1: el prefijo no cambia
        for i in range(1, k + 1):
            li = locs[i]
            arr_p = dep[i] + D[li][lp]
            sv_p = arr_p if arr_p > ap else ap
            if sv_p > bp + EPS or cum[i] + qp > Q:
                continue
            dep_p = sv_p + sp
            cur_loc, cur_t = lp, dep_p
            for j in range(i, k + 1):
                if j > i:
                    if cum[j] + qp > Q:
                        break
                    lj = locs[j]
                    node = route[j - 1]
                    a_n = cur_t + D[cur_loc][lj]
                    sv = a_n if a_n > a_[node] else a_[node]
                    if sv > b_[node] + EPS:
                        break
                    cur_t = sv + s_[node]
                    cur_loc = lj
                arr_d = cur_t + D[cur_loc][ld]
                sv_d = arr_d if arr_d > ad else ad
                if sv_d > bd + EPS:
                    continue
                dep_d = sv_d + sd
                nl = locs[j + 1]
                arr_n = dep_d + D[ld][nl]
                if j < k:
                    nn = route[j]
                    sv_n = arr_n if arr_n > a_[nn] else a_[nn]
                    if sv_n > lat[j + 1] + EPS:
                        continue
                    dl = arr_n - arr[j + 1] - wsuf[j + 1]
                    end_new = arr[k + 1] + (dl if dl > 0.0 else 0.0)
                else:
                    if arr_n > T + EPS:
                        continue
                    end_new = arr_n
                if j == i:
                    dd = D[li][lp] + D[lp][ld] + D[ld][nl] - D[li][nl]
                else:
                    l1, lj2 = locs[i + 1], locs[j]
                    dd = (D[li][lp] + D[lp][l1] - D[li][l1]) + (D[lj2][ld] + D[ld][nl] - D[lj2][nl])
                crews = math.ceil((end_new - dep0) / MD - 1e-9)
                if crews < 1:
                    crews = 1
                cost = w3 * (vd0 + dd) + w2 * crews - rc0
                if bon:
                    if best is not None and cost - 2.0 * bon >= best[0]:
                        continue
                    if has_bonus(lp, arr_p, dep_p):
                        cost -= bon
                    if has_bonus(ld, arr_d, dep_d):
                        cost -= bon
                if best is not None and cost >= best[0]:
                    continue
                best = (cost, i, j)
        if best is None:
            return None
        _, i, j = best
        return best[0], route[:i] + [p] + route[i:j] + [p + P] + route[j:]

    def _best_in(self, base, p):
        """Mejor inserción del par p en la lista base."""
        inst = self.inst
        if not base:
            new = [p, p + self.P]
            sch = sched(inst, new)
            if sch is None:
                return None
            return inst.w3 * sch["vdist"] + inst.w2 * CREW_PROXY_W * crews_needed(inst, sch), new
        R = mk(inst, base)
        if R is None:
            return None
        res = self._route_best(-1, R, p, None)
        if res is None:
            return None
        return res[0] + R["rc"], res[1]

    def _options(self, routes, p, skip=(), allow_new=True):
        """Mejor inserción del par p en cada ruta (y en un vehículo nuevo)."""
        inst, P = self.inst, self.P
        loc_vis = None
        if self.bonus > 0:
            loc_vis = defaultdict(list)
            for ridx, R in enumerate(routes):
                for vis in R["sch"]["visits"]:
                    loc_vis[vis["loc"]].append((ridx, vis["entry"], vis["exit"]))
        opts = []
        for ridx, R in enumerate(routes):
            if ridx in skip or not R["route"]:
                continue
            res = self._route_best(ridx, R, p, loc_vis)
            if res is not None:
                opts.append((res[0], ridx, res[1]))
        if allow_new:
            new = [p, p + P]
            sch = sched(inst, new)
            if sch is not None:
                opts.append((inst.w1 + inst.w3 * sch["vdist"]
                             + inst.w2 * CREW_PROXY_W * crews_needed(inst, sch), len(routes), new))
        opts.sort(key=lambda t: t[0])
        return opts

    def _apply_opt(self, routes, opt):
        _, ridx, new = opt
        r = mk(self.inst, new)
        if r is None:
            return False
        if ridx == len(routes):
            routes.append(r)
        else:
            routes[ridx] = r
        return True

    def _repair_greedy(self, routes, plist, skip=(), allow_new=True):
        for p in plist:
            opts = self._options(routes, p, skip, allow_new)
            if not opts:
                return False
            if not self._apply_opt(routes, opts[0]):
                return False
        return True

    def _repair_regret(self, routes, plist, allow_new=True):
        un = list(plist)
        while un:
            best = None
            for p in un:
                opts = self._options(routes, p, (), allow_new)
                if not opts:
                    return False
                regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else 1e9
                key = (regret, -opts[0][0])
                if best is None or key > best[0]:
                    best = (key, p, opts[0])
            if not self._apply_opt(routes, best[2]):
                return False
            un.remove(best[1])
        return True

    def _destroy_repair(self, sol, pset, repair="greedy"):
        routes = self._remove(sol.routes, pset)
        if routes is None:
            return
        plist = list(pset)
        self.rng.shuffle(plist)
        ok = (self._repair_regret(routes, plist) if repair == "regret"
              else self._repair_greedy(routes, plist))
        if ok:
            sol.routes = routes

    def _rand_pick(self, items, power):
        y = self.rng.random()
        return items.pop(min(int((y ** power) * len(items)), len(items) - 1))

    def h01(self, sol):  # Relocate
        pairs = self._all_pairs(sol.routes)
        if not pairs:
            return
        p, r = self.rng.choice(pairs)
        routes = self._remove(sol.routes, {p})
        if routes is None:
            return
        gone = len(routes) < len(sol.routes)
        opts = self._options(routes, p, () if gone else (r,), allow_new=not gone)
        if opts and self._apply_opt(routes, opts[0]):
            sol.routes = routes

    def h02(self, sol):  # Swap de pares entre dos rutas
        R, rng, inst = sol.routes, self.rng, self.inst
        if len(R) < 2:
            return
        best = None
        for _ in range(8):
            r1, r2 = rng.sample(range(len(R)), 2)
            p1 = rng.choice(self._pairs(R[r1]["route"]))
            p2 = rng.choice(self._pairs(R[r2]["route"]))
            L1 = self._without(R[r1]["route"], p1)
            L2 = self._without(R[r2]["route"], p2)
            a = self._best_in(L2, p1)
            b = self._best_in(L1, p2)
            if a is None or b is None:
                continue
            delta = a[0] + b[0] - R[r1]["rc"] - R[r2]["rc"]
            if best is None or delta < best[0]:
                best = (delta, r1, r2, a[1], b[1])
        if best:
            _, r1, r2, n2, n1 = best
            m1, m2 = mk(inst, n1), mk(inst, n2)
            if m1 is not None and m2 is not None:
                R[r1], R[r2] = m1, m2

    def h03(self, sol):  # Eliminación aleatoria + inserción voraz
        pairs = [p for p, _ in self._all_pairs(sol.routes)]
        if not pairs:
            return
        pset = set(self.rng.sample(pairs, min(self._k(), len(pairs))))
        self._destroy_repair(sol, pset, "greedy")

    def h04(self, sol):  # Shaw + regret-2
        pairs = [p for p, _ in self._all_pairs(sol.routes)]
        if not pairs:
            return
        k = min(self._k(), len(pairs))
        seed = self.rng.choice(pairs)
        removed = [seed]
        remaining = [p for p in pairs if p != seed]
        while len(removed) < k and remaining:
            r = self.rng.choice(removed)
            order = sorted(remaining, key=lambda q: self.rel[r][q])
            chosen = self._rand_pick(order, 6.0)
            removed.append(chosen)
            remaining.remove(chosen)
        self._destroy_repair(sol, set(removed), "regret")

    def h05(self, sol):  # Eliminación por ubicación
        kmax = self._kmax()
        locs = list(self.loc_pairs)
        pset, tries = set(), 0
        while len(pset) < 2 and tries < 4:
            cand = self.loc_pairs[self.rng.choice(locs)]
            pset.update(cand if len(cand) <= kmax else self.rng.sample(cand, kmax))
            tries += 1
        if len(pset) > kmax + 2:
            pset = set(self.rng.sample(sorted(pset), kmax + 2))
        self._destroy_repair(sol, pset, "greedy")

    def h06(self, sol):  # Eliminación de los peores (mayor costo marginal)
        inst = self.inst
        marg = []
        for R in sol.routes:
            for p in self._pairs(R["route"]):
                rest = self._without(R["route"], p)
                if not rest:
                    gain = R["rc"] + inst.w1
                else:
                    m = mk(inst, rest)
                    gain = R["rc"] - m["rc"] if m else 0.0
                marg.append((gain, p))
        if not marg:
            return
        marg.sort(key=lambda t: -t[0])
        k = min(self._k(), len(marg))
        removed = [self._rand_pick(marg, 3.0)[1] for _ in range(k)]
        self._destroy_repair(sol, set(removed), "greedy")

    def h07(self, sol):  # Vaciar la ruta más corta y repartir sus pares
        R = sol.routes
        if len(R) < 2:
            return
        order = sorted(range(len(R)), key=lambda r: len(R[r]["route"]))
        r = self.rng.choice(order[:2])
        pset = set(self._pairs(R[r]["route"]))
        new = self._remove(R, pset)
        if new is None:
            return
        plist = list(pset)
        self.rng.shuffle(plist)
        if self._repair_greedy(new, plist, allow_new=False):
            sol.routes = new

    def h08(self, sol):  # Fusión de dos rutas
        R, inst, rng = sol.routes, self.inst, self.rng
        if len(R) < 2:
            return
        for _ in range(4):
            a, b = rng.sample(range(len(R)), 2)
            merged = None
            for x, y in ((a, b), (b, a)):
                merged = mk(inst, R[x]["route"] + R[y]["route"])
                if merged:
                    break
            if merged is None:
                for x, y in ((a, b), (b, a)):
                    base, ok = list(R[x]["route"]), True
                    for p in self._pairs(R[y]["route"]):
                        ins = self._best_in(base, p)
                        if ins is None:
                            ok = False
                            break
                        base = ins[1]
                    if ok:
                        merged = mk(inst, base)
                        break
            if merged:
                sol.routes = [r for i, r in enumerate(R) if i not in (a, b)] + [merged]
                return

    def _best_variant(self, variants):
        inst = self.inst
        best = None
        for new in variants:
            sch = sched(inst, new)
            if sch is None:
                continue
            rc = inst.w3 * sch["vdist"] + inst.w2 * CREW_PROXY_W * crews_needed(inst, sch)
            if best is None or rc < best[0]:
                best = (rc, new)
        return best

    def h09(self, sol):  # Or-opt
        cand = [i for i, r in enumerate(sol.routes) if len(r["route"]) >= 4]
        if not cand:
            return
        ridx = self.rng.choice(cand)
        L = sol.routes[ridx]["route"]
        n = len(L)
        s = self.rng.randint(1, min(3, n - 1))
        st = self.rng.randint(0, n - s)
        seg, rest = L[st:st + s], L[:st] + L[st + s:]
        variants = (rest[:pos] + seg + rest[pos:] for pos in range(len(rest) + 1) if pos != st)
        best = self._best_variant(variants)
        if best:
            sol.routes[ridx] = mk(self.inst, best[1])

    def _closed_points(self, L):
        P = self.P
        open_, pts = set(), [0]
        for idx, i in enumerate(L):
            if i < P:
                open_.add(i)
            else:
                open_.discard(i - P)
            if not open_:
                pts.append(idx + 1)
        return pts

    def h10(self, sol):  # Intercambio de colas (2-opt*)
        R, inst, rng = sol.routes, self.inst, self.rng
        if len(R) < 2:
            return
        for _ in range(4):
            r1, r2 = rng.sample(range(len(R)), 2)
            L1, L2 = R[r1]["route"], R[r2]["route"]
            n1, n2 = len(L1), len(L2)
            old = R[r1]["rc"] + R[r2]["rc"] + 2 * inst.w1
            best = None
            for i in self._closed_points(L1):
                for j in self._closed_points(L2):
                    if (i == 0 and j == 0) or (i == n1 and j == n2):
                        continue
                    A, B = L1[:i] + L2[j:], L2[:j] + L1[i:]
                    ma = mk(inst, A) if A else None
                    mb = mk(inst, B) if B else None
                    if (A and ma is None) or (B and mb is None):
                        continue
                    delta = ((ma["rc"] + inst.w1) if ma else 0.0) + ((mb["rc"] + inst.w1) if mb else 0.0) - old
                    if best is None or delta < best[0]:
                        best = (delta, ma, mb)
            if best:
                _, ma, mb = best
                keep = [r for i, r in enumerate(R) if i not in (r1, r2)]
                sol.routes = keep + [m for m in (ma, mb) if m]
                return

    def h11(self, sol):  # Swap de dos nodos dentro de una ruta
        cand = [i for i, r in enumerate(sol.routes) if len(r["route"]) >= 4]
        if not cand:
            return
        ridx = self.rng.choice(cand)
        L = sol.routes[ridx]["route"]
        n = len(L)

        def gen():
            for a in range(n - 1):
                for b in range(a + 1, n):
                    new = list(L)
                    new[a], new[b] = new[b], new[a]
                    yield new
        best = self._best_variant(gen())
        if best:
            sol.routes[ridx] = mk(self.inst, best[1])

    def h12(self, sol):  # Inversión de segmento
        cand = [i for i, r in enumerate(sol.routes) if len(r["route"]) >= 4]
        if not cand:
            return
        ridx = self.rng.choice(cand)
        L = sol.routes[ridx]["route"]
        n = len(L)

        def gen():
            for a in range(n - 1):
                for b in range(a + 1, n):
                    yield L[:a] + L[a:b + 1][::-1] + L[b + 1:]
        best = self._best_variant(gen())
        if best:
            sol.routes[ridx] = mk(self.inst, best[1])

    def _sync_candidates(self, routes):
        byloc = defaultdict(list)
        for ridx, R in enumerate(routes):
            n = len(R["route"]) + 2
            for vis in R["sch"]["visits"]:
                if vis["i0"] >= 1 and vis["i1"] <= n - 2:
                    byloc[vis["loc"]].append((ridx, vis))
        cands = []
        for lst in byloc.values():
            for (r1, v1) in lst:
                for (r2, v2) in lst:
                    if r1 == r2:
                        continue
                    gap = v2["entry"] - v1["exit"]
                    if EPS < gap <= self.inst.MD:
                        cands.append((gap, r1, v1["i1"], v2["entry"]))
        cands.sort()
        return cands

    def _sync(self, sol, nmax):
        cands = self._sync_candidates(sol.routes)
        done, tries, used = 0, 0, set()
        while cands and done < nmax and tries < 10:
            tries += 1
            _, r1, pos, target = self._rand_pick(cands, 2.0)
            if r1 in used:
                continue
            R = sol.routes[r1]
            md = dict(R["min_dep"])
            md[pos] = max(md.get(pos, 0.0), target)
            m = mk(self.inst, R["route"], md)
            if m is None:
                continue
            sol.routes[r1] = m
            used.add(r1)
            done += 1

    def h13(self, sol):  # Retraso para sincronizar (1 retraso)
        self._sync(sol, 1)

    def h14(self, sol):  # Quitar una espera
        cand = [i for i, r in enumerate(sol.routes) if r["min_dep"]]
        if not cand:
            return
        ridx = self.rng.choice(cand)
        R = sol.routes[ridx]
        md = dict(R["min_dep"])
        md.pop(self.rng.choice(list(md)))
        m = mk(self.inst, R["route"], md)
        if m:
            sol.routes[ridx] = m

    def h15(self, sol):  # Reiniciar una ruta a tiempos tempranos
        cand = [i for i, r in enumerate(sol.routes) if r["min_dep"]]
        if not cand:
            return
        ridx = self.rng.choice(cand)
        m = mk(self.inst, sol.routes[ridx]["route"])
        if m:
            sol.routes[ridx] = m

    def h16(self, sol):  # Reiniciar todas las esperas
        for ridx, R in enumerate(sol.routes):
            if R["min_dep"]:
                m = mk(self.inst, R["route"])
                if m:
                    sol.routes[ridx] = m

    def h17(self, sol):  # Sincronización múltiple (2 a 3 retrasos a la vez)
        self._sync(sol, self.rng.randint(2, 3))

    def h18(self, sol):  # Desplazar la salida del depósito
        if not sol.routes:
            return
        ridx = self.rng.randrange(len(sol.routes))
        R = sol.routes[ridx]
        base = R["sch"]["dep"][0]
        for delta in (self.rng.uniform(1, 15), self.rng.uniform(1, 6), 1.0):
            md = dict(R["min_dep"])
            md[0] = base + delta
            m = mk(self.inst, R["route"], md)
            if m:
                sol.routes[ridx] = m
                return

    def h19(self, sol):  # Priorizar traspasos
        sol.pol["prefer_handoff"] = not sol.pol.get("prefer_handoff", False)

    def h20(self, sol):  # Orden perturbado (desempates aleatorios)
        sol.pol["shuffle"] = True
        sol.pol["seed"] = self.rng.randrange(1 << 30)

    def h21(self, sol):  # Best-fit <-> worst-fit
        sol.pol["fit"] = "worst" if sol.pol.get("fit", "best") != "worst" else "best"

    def h22(self, sol):  # Corte forzado de una tripulación en un tramo
        if not sol.routes:
            return
        forced = sol.pol["forced"]
        if len(forced) > 10 or (forced and self.rng.random() < 0.3):
            forced.discard(self.rng.choice(sorted(forced)))
        for _ in range(self.rng.randint(1, 3)):
            v = self.rng.randrange(len(sol.routes))
            nleg = len(sol.routes[v]["sch"]["visits"]) - 1
            if nleg >= 1:
                forced.add((v, self.rng.randrange(nleg)))

    def h23(self, sol):  # Selección aleatoria de tripulación
        sol.pol["fit"] = "random"
        sol.pol["seed"] = self.rng.randrange(1 << 30)

    def h24(self, sol):  # Cadena de eyección (p1 -> r2, p2 -> otra ruta)
        R, inst, rng = sol.routes, self.inst, self.rng
        if len(R) < 3:
            return
        r1, r2 = rng.sample(range(len(R)), 2)
        p1 = rng.choice(self._pairs(R[r1]["route"]))
        p2 = rng.choice(self._pairs(R[r2]["route"]))
        L1 = self._without(R[r1]["route"], p1)
        L2 = self._without(R[r2]["route"], p2)
        ins = self._best_in(L2, p1)
        if ins is None:
            return
        tmp = list(R)
        tmp[r1] = mk(inst, L1) if L1 else empty_route()
        tmp[r2] = mk(inst, ins[1])
        if tmp[r1] is None or tmp[r2] is None:
            return
        opts = self._options(tmp, p2, skip=(r2,), allow_new=False)
        if not opts:
            return
        if self._apply_opt(tmp, opts[0]):
            sol.routes = [r for r in tmp if r["route"]]

    def h25(self, sol):
        inst, P = self.inst, self.P
        nodes = []
        for R in sol.routes:
            serv = R["sch"]["serv"]
            for pos, x in enumerate(R["route"], start=1):
                nodes.append((serv[pos], x if x < P else x - P))
        if not nodes:
            return
        pset = set()
        for _ in range(3):
            t0 = self.rng.choice(nodes)[0]
            pset = {p for t, p in nodes if t0 <= t <= t0 + inst.MD}
            if len(pset) >= 2:
                break
        cap = self._kmax() + 2
        if len(pset) > cap:
            pset = set(self.rng.sample(sorted(pset), cap))
        self._destroy_repair(sol, pset, "regret")

    def h26(self, sol):  # Regret-3 insertion
        """Destrucción aleatoria y reinserción con regret-3."""
        pairs = [p for p, _ in self._all_pairs(sol.routes)]
        if not pairs:
            return
        k = min(self._k(), len(pairs))
        removed = set(self.rng.sample(pairs, k))
        routes = self._remove(sol.routes, removed)
        if routes is None:
            return
        un = list(removed)
        while un:
            best = None
            for p in un:
                opts = self._options(routes, p, (), True)
                if not opts:
                    return
                if len(opts) >= 3:
                    regret = opts[2][0] - opts[0][0]
                elif len(opts) == 2:
                    regret = opts[1][0] - opts[0][0]
                else:
                    regret = 1e9
                key = (regret, -opts[0][0])
                if best is None or key > best[0]:
                    best = (key, p, opts[0])
            if not self._apply_opt(routes, best[2]):
                return
            un.remove(best[1])
        sol.routes = routes

    def h27(self, sol):  # Ruin geográfico
        """Ruin geográfico: quita los pares cercanos a un par semilla y los
        reinserta."""
        R, inst, rng = sol.routes, self.inst, self.rng
        pairs = [p for p, _ in self._all_pairs(R)]
        if len(pairs) < 3:
            return
        dvals = []
        for a in pairs:
            for b in pairs:
                if a != b:
                    dvals.append(inst.D[inst.loc[a]][inst.loc[b]])
        if not dvals:
            return
        dvals.sort()
        r = dvals[int(0.3 * (len(dvals) - 1))]
        seed = rng.choice(pairs)
        pset = {p for p in pairs if inst.D[inst.loc[seed]][inst.loc[p]] <= r}
        if len(pset) < 2:
            return
        self._destroy_repair(sol, pset, "greedy")

    def h28(self, sol):  # Relocate encadenado
        """Relocate de r1 a r2 y luego de r2 a r1."""
        R, inst, rng = sol.routes, self.inst, self.rng
        if len(R) < 2:
            return
        r1, r2 = rng.sample(range(len(R)), 2)
        if not R[r1]["route"] or not R[r2]["route"]:
            return
        p1 = rng.choice(self._pairs(R[r1]["route"]))
        L1 = self._without(R[r1]["route"], p1)
        a = self._best_in(R[r2]["route"], p1)
        if a is None:
            return
        tmp2 = a[1]
        if not tmp2:
            return
        p2 = rng.choice(self._pairs(tmp2))
        L2 = self._without(tmp2, p2)
        b = self._best_in(L1, p2) if L1 else (0.0, [p2, p2 + inst.P])
        if b is None:
            return
        m1 = mk(inst, b[1]) if b[1] else empty_route()
        m2 = mk(inst, L2) if L2 else empty_route()
        if m1 is None or m2 is None:
            return
        sol.routes = [r for i, r in enumerate(R) if i not in (r1, r2)]
        if m1["route"]:
            sol.routes.append(m1)
        if m2["route"]:
            sol.routes.append(m2)

    def h29(self, sol):  # Reiniciar política
        """Vuelve la política a la por defecto."""
        sol.pol = default_pol()

    def _max_delay(self, R):
        """Mayor retraso d de la salida del depósito que mantiene factible
        la ruta."""
        inst = self.inst
        route, sch = R["route"], R["sch"]
        arr, dep = sch["arr"], sch["dep"]
        k = len(route)
        best, W = float("inf"), 0.0
        for t in range(1, k + 1):
            x = route[t - 1]
            best = min(best, W + inst.b[x] - arr[t])
            W += dep[t] - arr[t] - inst.s[x]
        best = min(best, W + inst.T - arr[k + 1])
        return max(0.0, best)

    def h30(self, sol):  # Retraso máximo de inicio
        """Retrasa al máximo la salida del depósito de una ruta con esperas,
        sin cambiar su llegada final."""
        inst, rng = self.inst, self.rng
        cand = []
        for ridx, R in enumerate(sol.routes):
            sch = R["sch"]
            if not R["route"] or crews_needed(inst, sch) < 2:
                continue
            route = R["route"]
            wait = 0.0
            for p in range(1, len(route) + 1):
                wait += max(0.0, sch["dep"][p] - sch["arr"][p] - inst.s[route[p - 1]])
            if wait > 1e-6:
                cand.append((ridx, wait))
        if not cand:
            return
        rng.shuffle(cand)
        fallback = None
        for ridx, wait in cand[:5]:
            R = sol.routes[ridx]
            base = R["sch"]["dep"][0]
            c0 = crews_needed(inst, R["sch"])
 
            def with_delay(d):
                md = dict(R["min_dep"])
                md[0] = base + d
                return mk(inst, R["route"], md)
 
            d = min(wait, self._max_delay(R))
            m = with_delay(d) if d > 1e-6 else None
            if m is None and d > 1e-6:
                m = with_delay(max(0.0, d - 1e-7))   # holgura numérica
            if m is None:
                continue
            if crews_needed(inst, m["sch"]) < c0:
                sol.routes[ridx] = m
                return
            if fallback is None:
                fallback = (ridx, m)
        if fallback is not None:          # no bajó tripulaciones: igual acorta la ruta
            sol.routes[fallback[0]] = fallback[1]
    def h31(self, sol):  # Recortar la cola de una ruta
        """Quita 1 a 3 pares del extremo de una ruta si así necesita una
        tripulación menos."""
        inst, rng, P = self.inst, self.rng, self.P
        MD = inst.MD
        cand = []
        for ridx, R in enumerate(sol.routes):
            L = R["route"]
            npairs = len(self._pairs(L))
            if npairs < 2:
                continue
            c = crews_needed(inst, R["sch"])
            if c < 2:
                continue
            over = (R["sch"]["arr"][-1] - R["sch"]["dep"][0]) - (c - 1) * MD
            cand.append((over, ridx, c, npairs))
        if not cand:
            return
        cand.sort()
        _, ridx, c, npairs = self._rand_pick(cand, 2.0)
        R = sol.routes[ridx]
        L = R["route"]
        ends = [True, False]
        rng.shuffle(ends)
        for tail in ends:
            seq = L[::-1] if tail else L
            order = []
            for x in seq:
                p = x if x < P else x - P
                if p not in order:
                    order.append(p)
            for t in range(1, min(3, npairs - 1) + 1):
                pset = set(order[:t])
                new = [x for x in L if (x if x < P else x - P) not in pset]
                m = mk(inst, new)
                if m is None or crews_needed(inst, m["sch"]) >= c:
                    continue
                routes = [r for i, r in enumerate(sol.routes) if i != ridx] + [m]
                plist = list(pset)
                rng.shuffle(plist)
                if self._repair_greedy(routes, plist, skip=(len(routes) - 1,)):
                    sol.routes = routes
                    return

    @staticmethod
    def _dep0(R, t):
        """Copia de las esperas de la ruta R con la salida del depósito en
        t."""
        md = dict(R["min_dep"])
        md[0] = t
        return md

    def _atraso_azar(self, sol, r):
        """Atrasa al azar la salida del depósito de la ruta r, dentro de lo
        factible."""
        R = sol.routes[r]
        dmax = int(self._max_delay(R) + 1e-9)
        if dmax < 1:
            return False
        d = self.rng.randint(1, dmax)
        m = mk(self.inst, R["route"], self._dep0(R, R["sch"]["dep"][0] + d))
        if m is None:
            return False
        sol.routes[r] = m
        return True

    def h32(self, sol):  # Atraso conjunto de vehículos
        """Atrasa al azar la salida de uno o dos vehículos."""
        n = len(sol.routes)
        if n == 0:
            return
        if n >= 2 and self.rng.random() < 0.7:
            idx = self.rng.sample(range(n), 2)
        else:
            idx = [self.rng.randrange(n)]
        for r in idx:
            self._atraso_azar(sol, r)

    def h33(self, sol):  # Dividir una ruta en dos vehículos
        """Divide una ruta en dos vehículos por un punto en que va vacía."""
        inst, rng = self.inst, self.rng
        cand = [i for i, r in enumerate(sol.routes) if len(self._pairs(r["route"])) >= 2]
        if not cand:
            return
        largas = [i for i in cand if crews_needed(inst, sol.routes[i]["sch"]) >= 2]
        ridx = rng.choice(largas if largas else cand)
        Lr = sol.routes[ridx]["route"]
        pts = [i for i in self._closed_points(Lr) if 0 < i < len(Lr)]
        if not pts:
            return
        rng.shuffle(pts)
        best = None
        for i in pts[:3]:
            A, B = mk(inst, Lr[:i]), mk(inst, Lr[i:])
            if A is None or B is None:
                continue
            if best is None or A["rc"] + B["rc"] < best[0]:
                best = (A["rc"] + B["rc"], A, B)
        if best is None:
            return
        sol.routes = [r for k, r in enumerate(sol.routes) if k != ridx] + [best[1], best[2]]


    def h34(self, sol):  # Mover pares y sincronizar
        """Mueve pares entre rutas y sincroniza una de las rutas cambiadas
        con otro vehículo."""
        rng = self.rng
        before = {id(r) for r in sol.routes}
        rng.choice((self.h01, self.h02, self.h10))(sol)
        changed = {i for i, r in enumerate(sol.routes) if id(r) not in before}
        if not changed:
            return
        cands = [c for c in self._sync_candidates(sol.routes) if c[1] in changed]
        for _ in range(min(5, len(cands))):
            _, r1, pos, target = self._rand_pick(cands, 2.0)
            R = sol.routes[r1]
            md = dict(R["min_dep"])
            md[pos] = max(md.get(pos, 0.0), target)
            m = mk(self.inst, R["route"], md)
            if m is not None:
                sol.routes[r1] = m
                return
        self._atraso_azar(sol, rng.choice(sorted(changed)))


# Decodificación final 
def final_outputs(inst, sol):
    _, cr = decode(inst, sol, safe=True, ampliada=True)
    schs = [r["sch"] for r in sol.routes]
    errs = J.verify(inst, schs, cr)
    return schs, cr, J.objective(inst, schs, cr), errs