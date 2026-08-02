"""Independent Set partition for polish multi-macro batching; two macros may
share an IS only if they share no net AND their bboxes hit no common bin so
both routing and density deltas are disjoint and parallel-safe."""

def build_macro_is_partition(cpu, macros, sizes, *, check_bbox=True,
                              return_stats=False):
    """Greedy IS partition by graph coloring over shared-net (and optionally
    bbox-overlap) conflicts; returns list[list[macro_id]], optionally plus a
    stats dict."""
    macros = list(macros)

    net_to_macros = {}
    for m in macros:
        off = int(cpu.mn_offsets[m])
        end = int(cpu.mn_offsets[m + 1])
        for k in range(off, end):
            ni = int(cpu.mn_net_ids[k])
            net_to_macros.setdefault(ni, []).append(m)

    # Pairwise net-conflict edges: macros sharing any net are neighbors.
    neighbors = {m: set() for m in macros}
    for ni, ms in net_to_macros.items():
        if len(ms) <= 1:
            continue
        ms_unique = set(ms)
        for m1 in ms_unique:
            for m2 in ms_unique:
                if m1 != m2:
                    neighbors[m1].add(m2)

    if check_bbox:
        # AABB hashing on a 64x64 grid: macros sharing any cell are treated as
        # conflicting. Conservative - false positives cost parallelism, not
        # correctness.
        cw_hash = 64
        ch_hash = 64
        # Canvas extent estimate: bracket all macro bboxes plus a 1.0 margin.
        xs = []; ys = []
        for m in macros:
            cx = float(cpu.pos[m, 0]); cy = float(cpu.pos[m, 1])
            hw = float(sizes[m, 0]) * 0.5
            hh = float(sizes[m, 1]) * 0.5
            xs.extend([cx - hw, cx + hw])
            ys.extend([cy - hh, cy + hh])
        x_max = max(xs) + 1.0; x_min = min(xs) - 1.0
        y_max = max(ys) + 1.0; y_min = min(ys) - 1.0
        gw_h = (x_max - x_min) / cw_hash
        gh_h = (y_max - y_min) / ch_hash
        cell_to_macros = {}
        for m in macros:
            cx = float(cpu.pos[m, 0]); cy = float(cpu.pos[m, 1])
            hw = float(sizes[m, 0]) * 0.5
            hh = float(sizes[m, 1]) * 0.5
            cl = max(0, int((cx - hw - x_min) / gw_h))
            cr = min(cw_hash - 1, int((cx + hw - x_min) / gw_h))
            rl = max(0, int((cy - hh - y_min) / gh_h))
            rr = min(ch_hash - 1, int((cy + hh - y_min) / gh_h))
            for r in range(rl, rr + 1):
                for c in range(cl, cr + 1):
                    cell_to_macros.setdefault((r, c), []).append(m)
        for ms in cell_to_macros.values():
            if len(ms) <= 1:
                continue
            ms_unique = set(ms)
            for m1 in ms_unique:
                for m2 in ms_unique:
                    if m1 != m2:
                        neighbors[m1].add(m2)

    colors = {}
    for m in macros:
        used = set(colors[n] for n in neighbors[m] if n in colors)
        c = 0
        while c in used:
            c += 1
        colors[m] = c

    partition_dict = {}
    for m, c in colors.items():
        partition_dict.setdefault(c, []).append(m)
    partition = list(partition_dict.values())
    # Largest IS first - improves downstream cache locality.
    partition.sort(key=len, reverse=True)

    if return_stats:
        sizes_list = [len(g) for g in partition]
        stats = {
            'n_partitions': len(partition),
            'min_size': min(sizes_list) if sizes_list else 0,
            'max_size': max(sizes_list) if sizes_list else 0,
            'mean_size': sum(sizes_list) / len(sizes_list) if sizes_list else 0,
            'total_macros': len(macros),
        }
        return partition, stats
    return partition
