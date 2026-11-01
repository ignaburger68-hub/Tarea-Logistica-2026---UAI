
import argparse
import heapq
import json
import math
import os
import sys
import time
from collections import defaultdict

EPS = 1e-9



class Instance:
    def __init__(self, path, rounding="none"):
        params, locs, reqs = {}, {}, []
        section = None
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    params[k.strip()] = float(v)
                    continue
                tok = line.split()
                if tok[0] == "L" and "X" in tok:
                    section = "L"
                    continue
                if tok[0] == "R" and "A" in tok:
                    section = "R"
                    continue
                if section == "L":
                    locs[tok[0]] = (float(tok[1]), float(tok[2]))
                elif section == "R":
                    reqs.append(tok)

        self.name = os.path.basename(path)
        self.Q = int(params["Q"])
        self.T = params["T"]
        self.w1 = params["VehicleUseCost"]
        self.w3 = params["VehicleDistanceCost"]
        self.w2 = params["CrewUseCost"]
        self.w4 = params["CrewDistanceCost"]
        self.MD = params["MaxDrivingTime"]

        pick = sorted([t for t in reqs if t[0].startswith("P")], key=lambda t: int(t[0][1:]))
        deli = sorted([t for t in reqs if t[0].startswith("D")], key=lambda t: int(t[0][1:]))
        if len(pick) != len(deli):
            raise ValueError("Distinto número de recogidas y entregas")
        self.P = len(pick)
        rows = pick + deli
        self.rid = [t[0] for t in rows]
        self.loc = [t[1] for t in rows]
        self.a = [float(t[2]) for t in rows]
        self.b = [float(t[3]) for t in rows]
        self.s = [float(t[4]) for t in rows]
        self.q = [int(t[5]) for t in rows]

        def rnd(x):
            if rounding == "ceil":
                return float(math.ceil(x - 1e-9))
            if rounding == "round":
                return float(round(x))
            return x

        self.D = {}
        for l1, (x1, y1) in locs.items():
            self.D[l1] = {}
            for l2, (x2, y2) in locs.items():
                self.D[l1][l2] = 0.0 if l1 == l2 else rnd(math.hypot(x1 - x2, y1 - y2))


# Programación de una ruta de vehículo

def schedule_route(inst, route, min_dep=None):
    """Tiempos más tempranos de una ruta (lista de índices de solicitudes)."""
    D = inst.D
    k = len(route)
    locs = ["LS"] + [inst.loc[i] for i in route] + ["LE"]
    n = k + 2
    arr = [0.0] * n
    serv = [0.0] * n
    dep = [0.0] * n
    tS = max(0.0, inst.a[route[0]] - D["LS"][locs[1]]) if k else 0.0
    arr[0] = serv[0] = dep[0] = tS
    load = 0
    vd = 0.0
    seen = set()
    P = inst.P
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
        if min_dep and p in min_dep and min_dep[p] > dp:
            dp = min_dep[p]
        dep[p] = dp
    d = D[locs[n - 2]][locs[n - 1]]
    vd += d
    arr[n - 1] = serv[n - 1] = dep[n - 1] = dep[n - 2] + d
    if arr[n - 1] > inst.T + EPS:
        return None
    return {"route": list(route), "locs": locs, "arr": arr, "serv": serv,
            "dep": dep, "vdist": vd}


def add_visits(inst, sch):
    """Agrupa nodos consecutivos de una misma ubicación en 'visitas' y
    calcula los tramos (movimientos entre ubicaciones distintas)."""
    locs, arr, dep = sch["locs"], sch["arr"], sch["dep"]
    n = len(locs)
    visits = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and locs[j + 1] == locs[i]:
            j += 1
        visits.append({"loc": locs[i], "i0": i, "i1": j,
                       "entry": arr[i], "exit": dep[j]})
        i = j + 1
    sch["visits"] = visits
    sch["legd"] = [inst.D[visits[m]["loc"]][visits[m + 1]["loc"]]
                   for m in range(len(visits) - 1)]
    return sch


def crews_needed(inst, sch):
    """Tripulaciones mínimas estimadas para cubrir la duración de la ruta."""
    span = sch["arr"][-1] - sch["dep"][0]
    return max(1, math.ceil(span / inst.MD - 1e-9))



# Paso 1: rutas de vehículos (inserción más barata)
def construct_routes(inst):
    """Inserción más barata considerando solo el costo de los vehículos."""
    P = inst.P
    order = sorted(range(P), key=lambda p: (inst.a[p], inst.b[p]))
    routes = []
    for p in order:
        best = None   # (costo, índice de ruta, nueva lista, distancia)
        for ridx in range(len(routes) + 1):
            if ridx < len(routes):
                base, od, fixed = routes[ridx][0], routes[ridx][1], 0.0
            else:
                base, od, fixed = [], 0.0, inst.w1   # abrir un vehículo nuevo
            k = len(base)
            for i in range(k + 1):              # posición de la recogida
                for j in range(i, k + 1):       # posición de la entrega
                    new = base[:i] + [p] + base[i:j] + [p + P] + base[j:]
                    sch = schedule_route(inst, new)
                    if sch is None:
                        continue
                    cost = fixed + inst.w3 * (sch["vdist"] - od)
                    if best is None or cost < best[0]:
                        best = (cost, ridx, new, sch["vdist"])
        if best is None:
            raise RuntimeError("El par %s es infactible por sí solo" % inst.rid[p])
        _, ridx, new, vd = best
        if ridx == len(routes):
            routes.append((new, vd))
        else:
            routes[ridx] = (new, vd)
    return [r for r, _ in routes]



# Paso 2: asignación de tripulaciones (greedy) 
def assign_crews(inst, schs):
    MD = inst.MD
    legs = []
    for v, sch in enumerate(schs):
        vis = sch["visits"]
        for m in range(len(vis) - 1):
            legs.append((vis[m]["exit"], v, m))
    legs.sort()

    crews = []
    onboard = {}
    leg_driver = {}  # (veh, tramo) -> tripulación conductora
    bad = 0

    def ok(c, dep, arr):
        ds = crews[c]["ds"]
        return arr - (dep if ds is None else ds) <= MD + EPS

    def span_of(c, dep, arr):
        ds = crews[c]["ds"]
        return arr - (dep if ds is None else ds)

    for dep, v, m in legs:
        vis = schs[v]["visits"]
        arr = vis[m + 1]["entry"]
        loc = vis[m]["loc"]
        pool = onboard.setdefault((v, m), [])
        best = None  # (tramo_usado, crew, origen)

        # a) tripulación que ya va a bordo del vehículo
        for c in pool:
            if ok(c, dep, arr):
                sp = span_of(c, dep, arr)
                if best is None or sp > best[0]:
                    best = (sp, c, None)

        # b) traspaso desde otro vehículo presente en la misma ubicación
        if best is None:
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
                        continue  # w la necesita para su propio tramo
                    sp = span_of(c, dep, arr)
                    if best is None or sp > best[0]:
                        best = (sp, c, (w, mp))

        # c) tripulación nueva (aborda el vehículo en su nodo de inicio)
        if best is None:
            crews.append({"rides": [[v, 0, m]], "ds": None, "de": None})
            c = len(crews) - 1
            pool.append(c)
            if arr - dep > MD + EPS:
                bad += 1
        else:
            _, c, src = best
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


class PassengerPaths:
    """Caminos más cortos como pasajero sobre los horarios fijos de los
    vehículos."""

    def __init__(self, schs):
        vis = self.vis = [s["visits"] for s in schs]
        legd = self.legd = [s["legd"] for s in schs]
        byloc = {}
        for v, vv in enumerate(vis):
            for k, x in enumerate(vv):
                byloc.setdefault(x["loc"], []).append((v, k))

        def forward(v, k, f):
            if k + 1 < len(vis[v]):
                yield (v, k + 1, 0), legd[v][k]
            if f == 0:
                t = vis[v][k]["entry"]
                for (u, j) in byloc[vis[v][k]["loc"]]:
                    if (u, j) != (v, k) and vis[u][j]["exit"] >= t - 1e-9:
                        yield (u, j, 1), 0.0

        # hacia adelante desde el deposito
        self.fdist, self.fprev = _dijkstra([(v, 0, 0) for v in range(len(vis))],
                                           lambda n: forward(*n))
        # hacia atras desde el final de cada vehiculo
        radj = {}
        for v, vv in enumerate(vis):
            for k in range(len(vv)):
                for f in (0, 1):
                    for m, w in forward(v, k, f):
                        radj.setdefault(m, []).append(((v, k, f), w))
        ends = [(v, len(vv) - 1, f) for v, vv in enumerate(vis) for f in (0, 1)]
        _, self.bnext = _dijkstra(ends, lambda n: radj.get(n, ()))

    def dist(self, rides):
        return sum(sum(self.legd[v][a:b]) for v, a, b in rides)

    def rewrite(self, rides, legs):
        """Reescribe la ida y la vuelta de una tripulación como pasajera por
        caminos más cortos."""
        vis = self.vis
        dl = sorted((vis[v][m]["exit"], v, m) for v, m in legs)
        (_, v1, m1), (_, v2, m2) = dl[0], dl[-1]
        mid, on, closed = [], False, False
        for (v, a, b) in rides:
            if not on and v == v1 and a <= m1 < b:
                on, a = True, m1
            if on:
                if v == v2 and a <= m2 < b:
                    mid.append([v, a, m2 + 1])
                    closed = True
                    break
                mid.append([v, a, b])
        if not closed:
            return None
        starts = [(v1, m1, f) for f in (0, 1) if (v1, m1, f) in self.fdist]
        if not starts:
            return None
        n = min(starts, key=lambda s: self.fdist[s])
        path = []
        while n is not None:
            path.append(n)
            n = self.fprev[n]
        pre = _to_rides(path[::-1])
        path, n = [], (v2, m2 + 1, 0)
        if n not in self.bnext:
            return None
        while n is not None:
            path.append(n)
            n = self.bnext[n]
        post = _to_rides(path)
        out = []
        for r in pre + mid + post:
            if out and out[-1][0] == r[0] and out[-1][2] == r[1]:
                out[-1][2] = r[2]
            elif r[1] < r[2]:
                out.append(list(r))
        if out[0][1] != 0:
            out.insert(0, [out[0][0], 0, 0])
        return out


def _dijkstra(sources, adj):
    dist, link, pq = {}, {}, []
    for s in sources:
        dist[s] = 0.0
        link[s] = None
        pq.append((0.0, s))
    heapq.heapify(pq)
    while pq:
        dd, n = heapq.heappop(pq)
        if dd > dist[n] + 1e-12:
            continue
        for m, w in adj(n):
            nd = dd + w
            if nd < dist.get(m, float("inf")) - 1e-12:
                dist[m] = nd
                link[m] = n
                heapq.heappush(pq, (nd, m))
    return dist, link


def _to_rides(path):
    rides = []
    for a, b in zip(path, path[1:]):
        if a[0] == b[0] and b[1] == a[1] + 1:
            if rides and rides[-1][0] == a[0] and rides[-1][2] == a[1]:
                rides[-1][2] = b[1]
            else:
                rides.append([a[0], a[1], b[1]])
    return rides


def shorten_passengers(schs, cr):
    """Acorta los trayectos como pasajero de cada tripulación."""
    pp = PassengerPaths(schs)
    drives = {}
    for (v, m), c in cr["leg_driver"].items():
        drives.setdefault(c, []).append((v, m))
    new_crews = []
    for c, cd in enumerate(cr["crews"]):
        rides = pp.rewrite(cd["rides"], drives[c]) if c in drives else None
        new_crews.append(dict(cd, rides=rides) if rides is not None else cd)
    out = dict(cr)
    out["crews"] = new_crews
    out["cdist"] = pp.dist([r for cd in new_crews for r in cd["rides"]])
    return out



# Objetivo y verificador
def objective(inst, schs, cr):
    return (inst.w1 * len(schs) + inst.w2 * cr["ncrews"]
            + inst.w3 * sum(s["vdist"] for s in schs) + inst.w4 * cr["cdist"])


def verify(inst, schs, cr):
    """Verifica la solución con los datos crudos."""
    errs = []
    P = inst.P
    seen = set()
    for v, sch in enumerate(schs):
        route, n = sch["route"], len(sch["route"]) + 2
        arr, serv, dep, locs = sch["arr"], sch["serv"], sch["dep"], sch["locs"]
        pos = {}
        for k, i in enumerate(route):
            if i in seen:
                errs.append("solicitud repetida %s" % inst.rid[i])
            seen.add(i)
            pos[i] = k
        for i in route:
            pr = i + P if i < P else i - P
            if pr not in pos:
                errs.append("par incompleto %s" % inst.rid[i])
            elif i < P and pos[pr] < pos[i]:
                errs.append("precedencia violada %s" % inst.rid[i])
        load = 0
        for p in range(1, n - 1):
            i = route[p - 1]
            d = inst.D[locs[p - 1]][locs[p]]
            if abs(arr[p] - (dep[p - 1] + d)) > 1e-6:
                errs.append("tiempo de viaje inconsistente veh %d nodo %d" % (v + 1, p))
            if (serv[p] < arr[p] - 1e-6 or serv[p] < inst.a[i] - 1e-6
                    or serv[p] > inst.b[i] + 1e-6):
                errs.append("ventana violada en %s" % inst.rid[i])
            if dep[p] < serv[p] + inst.s[i] - 1e-6:
                errs.append("salida antes de terminar servicio en %s" % inst.rid[i])
            load += inst.q[i]
            if load < 0 or load > inst.Q:
                errs.append("capacidad violada en %s" % inst.rid[i])
        if arr[-1] > inst.T + 1e-6:
            errs.append("horizonte T excedido veh %d" % (v + 1))
    if len(seen) != 2 * P:
        errs.append("faltan solicitudes por atender")

    # tripulaciones
    driven = defaultdict(list)
    for (v, m), c in cr["leg_driver"].items():
        driven[c].append((v, m))
    for v, sch in enumerate(schs):
        for m in range(len(sch["visits"]) - 1):
            if (v, m) not in cr["leg_driver"]:
                errs.append("tramo sin conductor veh %d tramo %d" % (v + 1, m))
    for c, cd in enumerate(cr["crews"]):
        rides = cd["rides"]
        if not rides:
            errs.append("tripulación %d sin viajes" % (c + 1))
            continue
        if rides[0][1] != 0:
            errs.append("tripulación %d no parte del depósito" % (c + 1))
        v_last, _, b_last = rides[-1]
        if b_last != len(schs[v_last]["visits"]) - 1:
            errs.append("tripulación %d no termina en el fin del vehículo" % (c + 1))
        for k, (v, a, b) in enumerate(rides):
            if a > b:
                errs.append("viaje inválido tripulación %d" % (c + 1))
            if k + 1 < len(rides):
                v2, a2, _ = rides[k + 1]
                vis1, vis2 = schs[v]["visits"][b], schs[v2]["visits"][a2]
                if vis1["loc"] != vis2["loc"]:
                    errs.append("traspaso en ubicaciones distintas tripulación %d" % (c + 1))
                if vis1["entry"] > vis2["exit"] + 1e-6:
                    errs.append("traspaso sin coincidencia temporal tripulación %d" % (c + 1))
        for (v, m) in driven.get(c, []):
            if not any(rv == v and ra <= m < rb for rv, ra, rb in rides):
                errs.append("tripulación %d conduce un tramo sin ir a bordo" % (c + 1))
        if driven.get(c):
            ds = min(schs[v]["visits"][m]["exit"] for v, m in driven[c])
            de = max(schs[v]["visits"][m + 1]["entry"] for v, m in driven[c])
            if de - ds > inst.MD + 1e-6:
                errs.append("tripulación %d excede conducción máxima (%.2f)" % (c + 1, de - ds))
    return errs


# Salida

def to_json(inst, schs, cr, obj):
    vehicles = []
    for v, sch in enumerate(schs):
        names = ["S%d" % (v + 1)] + [inst.rid[i] for i in sch["route"]] + ["E%d" % (v + 1)]
        nodes = [{"nodo": names[p], "ubicacion": sch["locs"][p], "llegada": sch["arr"][p],
                  "inicio_servicio": sch["serv"][p], "salida": sch["dep"][p]}
                 for p in range(len(names))]
        legs = []
        for m in range(len(sch["visits"]) - 1):
            legs.append({"desde": sch["visits"][m]["loc"], "hasta": sch["visits"][m + 1]["loc"],
                         "salida": sch["visits"][m]["exit"], "llegada": sch["visits"][m + 1]["entry"],
                         "conductor": cr["leg_driver"][(v, m)] + 1})
        vehicles.append({"vehiculo": v + 1, "nodos": nodes, "tramos": legs})
    crews = []
    for c, cd in enumerate(cr["crews"]):
        rides = [{"vehiculo": v + 1,
                  "desde": schs[v]["visits"][a]["loc"], "hasta": schs[v]["visits"][b]["loc"]}
                 for v, a, b in cd["rides"]]
        crews.append({"tripulacion": c + 1, "inicio_conduccion": cd["ds"],
                      "fin_conduccion": cd["de"], "viajes": rides})
    return {"instancia": inst.name, "objetivo": obj, "vehiculos_usados": len(schs),
            "tripulaciones_usadas": cr["ncrews"], "vehiculos": vehicles, "tripulaciones": crews}


# Solución inicial completa 
def solve(inst):
    """Rutas por inserción más barata y luego tripulaciones."""
    schs = [add_visits(inst, schedule_route(inst, r)) for r in construct_routes(inst)]
    cr = assign_crews(inst, schs)
    if not cr["bad"]:   # acortar trayectos como pasajero (nunca empeora)
        crs = shorten_passengers(schs, cr)
        if not verify(inst, schs, crs):
            cr = crs
    return schs, cr


def main():
    ap = argparse.ArgumentParser(description="Solución inicial JVCRSP")
    ap.add_argument("instancia")
    ap.add_argument("--rounding", choices=["none", "ceil", "round"], default="ceil",
                    help="redondeo de distancias euclídeas")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    inst = Instance(args.instancia, args.rounding)
    schs, cr = solve(inst)
    errs = verify(inst, schs, cr)
    obj = objective(inst, schs, cr)

    print("Instancia      : %s (%d pares)" % (inst.name, inst.P))
    print("Vehículos      : %d" % len(schs))
    print("Tripulaciones  : %d" % cr["ncrews"])
    print("Dist. vehículos: %.2f" % sum(s["vdist"] for s in schs))
    print("Dist. tripul.  : %.2f" % cr["cdist"])
    print("Objetivo       : %.2f" % obj)
    print("Factible       : %s" % ("SI" if not errs else "NO"))
    for e in errs[:10]:
        print("  error:", e)
    print("Tiempo         : %.2f s" % (time.time() - t0))

    out = args.out or os.path.splitext(args.instancia)[0] + "_solucion.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(to_json(inst, schs, cr, obj), f, indent=2, ensure_ascii=False)
    print("Solución guardada en", out)
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())