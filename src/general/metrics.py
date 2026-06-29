from collections import deque

DIRS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


def normalize_grid(matrix, lab_size, fill='#'):
    grid = []
    for r in range(lab_size):
        row = list(matrix[r][:lab_size]) if r < len(matrix) else []
        row = row + [fill] * (lab_size - len(row))
        grid.append(row)
    return grid


def in_bounds(pos, lab_size):
    r, c = pos
    return 0 <= r < lab_size and 0 <= c < lab_size


def find_all_positions(grid, ch, lab_size):
    return [(r, c) for r in range(lab_size) for c in range(lab_size) if grid[r][c] == ch]


def check_violation(pred_grid, input_grid, ch, lab_size):
    pred_positions = find_all_positions(pred_grid, ch, lab_size)
    input_positions = find_all_positions(input_grid, ch, lab_size)
    if len(pred_positions) != 1 or not input_positions:
        return True, (pred_positions[0] if len(pred_positions) == 1 else None)
    violated = pred_positions[0] != input_positions[0]
    return violated, pred_positions[0]


def unique_directional_neighbor(grid, pos, lab_size):
    cands = []
    for dr, dc in DIRS.values():
        npos = (pos[0] + dr, pos[1] + dc)
        if in_bounds(npos, lab_size) and grid[npos[0]][npos[1]] in DIRS:
            cands.append(npos)
    return cands[0] if len(cands) == 1 else None


def find_predecessor(grid, input_grid, pos, lab_size):
    cands = []
    for ch, (dr, dc) in DIRS.items():
        npos = (pos[0] - dr, pos[1] - dc)
        if in_bounds(npos, lab_size) and grid[npos[0]][npos[1]] == ch and input_grid[npos[0]][npos[1]] != '#':
            cands.append(npos)
    return cands[0] if len(cands) == 1 else None


def walk_chain_forward(grid, start_pos, lab_size, max_steps):
    visited, seen, cur = [], set(), start_pos
    for _ in range(max_steps):
        if not in_bounds(cur, lab_size) or cur in seen:
            return visited, None
        seen.add(cur)
        ch = grid[cur[0]][cur[1]]
        if ch not in DIRS:
            return visited, cur
        visited.append(cur)
        dr, dc = DIRS[ch]
        cur = (cur[0] + dr, cur[1] + dc)
    return visited, None


def walk_chain_forward_blocked(grid, input_grid, start_pos, lab_size, max_steps):
    cur = start_pos
    for _ in range(max_steps):
        if not in_bounds(cur, lab_size):
            return cur
        ch = grid[cur[0]][cur[1]]
        if ch not in DIRS:
            return cur
        dr, dc = DIRS[ch]
        nxt = (cur[0] + dr, cur[1] + dc)
        if not in_bounds(nxt, lab_size) or input_grid[nxt[0]][nxt[1]] == '#':
            return cur
        cur = nxt
    return cur


def trace_predecessors(grid, input_grid, start_pos, lab_size, max_steps):
    positions, cur = [], start_pos
    for _ in range(max_steps):
        pred = find_predecessor(grid, input_grid, cur, lab_size)
        if pred is None:
            break
        positions.append(pred)
        cur = pred
    return positions


def is_neighbor(a, b):
    if a is None or b is None:
        return False
    return (abs(a[0] - b[0]) == 1 and a[1] == b[1]) or (abs(a[1] - b[1]) == 1 and a[0] == b[0])


def bfs_distance(input_grid, start, end, lab_size):
    if start is None or end is None:
        return -1
    if start == end:
        return 0
    visited = {start}
    q = deque([(start, 0)])
    while q:
        pos, d = q.popleft()
        for dr, dc in DIRS.values():
            npos = (pos[0] + dr, pos[1] + dc)
            if in_bounds(npos, lab_size) and npos not in visited and input_grid[npos[0]][npos[1]] != '#':
                if npos == end:
                    return d + 1
                visited.add(npos)
                q.append((npos, d + 1))
    return -1


def directions_list_to_grid(input_matrix, directions, lab_size):
    grid = normalize_grid(input_matrix, lab_size)
    s_pos = next(iter(find_all_positions(grid, 'S', lab_size)), None)
    if s_pos is None or not directions:
        return grid

    positions = [s_pos]
    cur = s_pos
    for d in directions:
        if d not in DIRS:
            break
        dr, dc = DIRS[d]
        nxt = (cur[0] + dr, cur[1] + dc)
        if not in_bounds(nxt, lab_size):
            break
        positions.append(nxt)
        cur = nxt

    for k in range(1, len(positions) - 1):
        pk = positions[k]
        grid[pk[0]][pk[1]] = directions[k]

    return grid


def decoded_to_directions(decoded):
    rows = decoded_to_matrix(decoded)
    return rows[0] if rows else []

def edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    if n < m:
        a, b = b, a
        n, m = m, n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost_sub = prev[j - 1] + (0 if ai == b[j - 1] else 1)
            cost_del = prev[j] + 1
            cost_ins = curr[j - 1] + 1
            curr[j] = min(cost_sub, cost_del, cost_ins)
        prev = curr
    return prev[m]


def ids_to_directions(ids, id_to_char, extract_type="prompt"):
    token_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    chars = [id_to_char.get(t, '') for t in token_list]

    sol_start, sol_end = -1, -1
    for i, c in enumerate(chars):
        if c in ['<SOLUTION_START>', '<COMPLETION_START>']:
            sol_start = i
        elif c in ['<SOLUTION_END>', '<COMPLETION_END>']:
            sol_end = i
            break

    if extract_type == "solution":
        if sol_end == -1:
            sol_end = len(chars)
        block_chars = chars[sol_start + 1: sol_end]
    else:
        end_idx = sol_start if sol_start != -1 else len(chars)
        block_chars = chars[:end_idx]

    return [c for c in block_chars if c != '\n' and not c.startswith('<') and c != '<PAD>']


def calculate_metrics(input_matrix, solv_matrix, pred_matrix, lab_size):
    input_grid = normalize_grid(input_matrix, lab_size)
    solv_grid = normalize_grid(solv_matrix, lab_size)
    pred_grid = normalize_grid(pred_matrix, lab_size)
    max_steps = lab_size * lab_size + 5
    results = {}

    s_input_pos = next(iter(find_all_positions(input_grid, 'S', lab_size)), None)

    # Início / Final Violado
    start_violated, s_pred_pos = check_violation(pred_grid, input_grid, 'S', lab_size)
    end_violated, e_pred_pos = check_violation(pred_grid, input_grid, 'E', lab_size)
    results['Início Violado'] = "Sim" if start_violated else "Não"
    results['Final Violado'] = "Sim" if end_violated else "Não"

    # Sequência do gabarito (forward)
    gt_first = unique_directional_neighbor(solv_grid, s_input_pos, lab_size) if s_input_pos else None
    gt_visited, gt_landing = walk_chain_forward(solv_grid, gt_first, lab_size, max_steps) if gt_first else ([], None)
    gt_dirs = [solv_grid[p[0]][p[1]] for p in gt_visited]

    # Sequência prevista (forward, completa)
    pred_first = None if start_violated else unique_directional_neighbor(pred_grid, s_pred_pos, lab_size)
    pred_visited, pred_landing = walk_chain_forward(pred_grid, pred_first, lab_size, max_steps) if pred_first else ([], None)
    pred_dirs = [pred_grid[p[0]][p[1]] for p in pred_visited]

    # Acurácia
    all_free = all(input_grid[p[0]][p[1]] != '#' for p in pred_visited)
    is_correct = (not start_violated and not end_violated and pred_first is not None
                  and pred_landing == e_pred_pos and all_free)
    if is_correct:
        is_otima = (pred_dirs == gt_dirs) or (len(pred_dirs) == len(gt_dirs))
        results['Acurácia'] = "Ótima" if is_otima else "Correta"
    else:
        results['Acurácia'] = "Incorreta"

    # Tokens Alterados / Ótimos / Diferentes
    def hamming(a, b):
        return sum(1 for r in range(lab_size) for c in range(lab_size) if a[r][c] != b[r][c])
    results['Tokens Alterados'] = hamming(pred_grid, input_grid)
    results['Tokens Ótimos'] = hamming(solv_grid, input_grid)
    results['Tokens Diferentes'] = hamming(solv_grid, pred_grid)

    # Paredes / Espaços Violados
    paredes_violadas = espacos_violados = 0
    for r in range(lab_size):
        for c in range(lab_size):
            was_wall, is_wall = input_grid[r][c] == '#', pred_grid[r][c] == '#'
            if was_wall and not is_wall:
                paredes_violadas += 1
            elif not was_wall and is_wall:
                espacos_violados += 1
    results['Paredes Violadas'] = paredes_violadas
    results['Espaços Violados'] = espacos_violados

    # Progresso Direto
    if pred_first is None:
        progresso_direto = 0
    else:
        k = 0
        while k < len(gt_dirs) and k < len(pred_dirs) and pred_dirs[k] == gt_dirs[k]:
            k += 1
        bonus = (k == len(gt_dirs) == len(pred_dirs)) and (pred_landing == e_pred_pos) and not end_violated
        progresso_direto = k + (1 if bonus else 0)
    results['Progresso Direto'] = progresso_direto

    # Progresso Inverso
    gt_rev_dirs = list(reversed(gt_dirs))
    if end_violated:
        progresso_inverso = 0
        inv_positions = []
    else:
        inv_positions = trace_predecessors(pred_grid, input_grid, e_pred_pos, lab_size, max_steps)
        inv_dirs = [pred_grid[p[0]][p[1]] for p in inv_positions]
        k = 0
        while k < len(gt_rev_dirs) and k < len(inv_dirs) and inv_dirs[k] == gt_rev_dirs[k]:
            k += 1
        inv_landing = inv_positions[-1] if inv_positions else e_pred_pos
        bonus = (k == len(gt_rev_dirs) == len(inv_dirs)) and is_neighbor(inv_landing, s_pred_pos) and not start_violated
        progresso_inverso = k + (1 if bonus else 0)
    results['Progresso Inverso'] = progresso_inverso

    # Distância Direta / Inversa
    fallback_dist = lab_size ** 2
    if s_pred_pos is None or e_pred_pos is None:
        results['Distância Direta'] = fallback_dist
        results['Distância Inversa'] = fallback_dist
    else:
        dist_first = unique_directional_neighbor(pred_grid, s_pred_pos, lab_size)
        if dist_first is None:
            dd = bfs_distance(input_grid, s_pred_pos, e_pred_pos, lab_size)
        else:
            stop_pos = walk_chain_forward_blocked(pred_grid, input_grid, dist_first, lab_size, max_steps)
            dd = 0 if stop_pos == e_pred_pos else bfs_distance(input_grid, stop_pos, e_pred_pos, lab_size)
        results['Distância Direta'] = dd if dd != -1 else fallback_dist

        dist_inv_positions = trace_predecessors(pred_grid, input_grid, e_pred_pos, lab_size, max_steps)
        if not dist_inv_positions:
            di = bfs_distance(input_grid, s_pred_pos, e_pred_pos, lab_size)
        else:
            stop_pos = dist_inv_positions[-1]
            reached_start = (stop_pos == s_pred_pos) or (dist_first is not None and stop_pos == dist_first)
            di = 0 if reached_start else bfs_distance(input_grid, stop_pos, s_pred_pos, lab_size)
        results['Distância Inversa'] = di if di != -1 else fallback_dist

    # Caminho Único / Conexo
    forward_set = set(pred_visited)
    backward_set = set(inv_positions) if not end_violated else set()
    all_dir_positions = set()
    for ch in DIRS:
        all_dir_positions.update(find_all_positions(pred_grid, ch, lab_size))

    results['Caminho Único'] = "Sim" if (forward_set | backward_set) == all_dir_positions else "Não"
    results['Caminho Conexo'] = "Sim" if (forward_set == all_dir_positions or backward_set == all_dir_positions) else "Não"

    score = {
        'Acurácia': 1.0 if results['Acurácia'] == 'Ótima' else 1.0 if results['Acurácia'] == 'Correta' else 0.0,
        'Tokens Diferentes': 1.0 - min(abs(results['Tokens Diferentes'] / results["Tokens Ótimos"]), 1.0),
        'Progresso Direto': results['Progresso Direto'] / (results['Tokens Ótimos'] + 1),
        'Progresso Inverso': results['Progresso Inverso'] / (results['Tokens Ótimos'] + 1),
        'Distância Direta': (1 / (results['Distância Direta'] + 1)) ** 0.5,
        'Distância Inversa': (1 / (results['Distância Inversa'] + 1)) ** 0.5,
        'Início Violado': 1.0 if results['Início Violado'] == 'Não' else 0.0,
        'Final Violado': 1.0 if results['Final Violado'] == 'Não' else 0.0,
        'Paredes Violadas': (1 / (results['Paredes Violadas'] + 1)),
        'Espaços Violados': (1 / (results['Espaços Violados'] + 1)),
        'Caminho Único': 0.0 if results['Caminho Único'] == 'Não' else 1.0,
        'Caminho Conexo': 0.0 if results['Caminho Conexo'] == 'Não' else 1.0
    }

    results['Corretude'] = (
        1.0 * score['Acurácia'] +
        0.6 * score['Tokens Diferentes'] +
        0.3 * score['Progresso Direto'] +
        0.2 * score['Progresso Inverso'] +
        1.6 * score['Distância Direta'] +
        0.3 * score['Distância Inversa'] +
        1.2 * score['Início Violado'] +
        1.2 * score['Final Violado'] +
        1.5 * score['Paredes Violadas'] +
        1.2 * score['Espaços Violados'] +
        0.6 * score['Caminho Único'] +
        0.3 * score['Caminho Conexo']
    ) / 10.0
    results['Corretude'] = max(0.0, min(int(results['Corretude'] * 1000) / 1000.0, 1.0))
    score['Corretude'] = results['Corretude']

    return results, score

def calculate_direction_metrics(input_matrix, solv_directions, pred_directions, lab_size):
    input_grid = normalize_grid(input_matrix, lab_size)
    solv_grid = directions_list_to_grid(input_matrix, solv_directions, lab_size)
    pred_grid = directions_list_to_grid(input_matrix, pred_directions, lab_size)
    max_steps = lab_size * lab_size + 5
    results = {}

    s_input_pos = next(iter(find_all_positions(input_grid, 'S', lab_size)), None)
    start_violated, s_pred_pos = check_violation(pred_grid, input_grid, 'S', lab_size)
    end_violated, e_pred_pos = check_violation(pred_grid, input_grid, 'E', lab_size)

    gt_first = unique_directional_neighbor(solv_grid, s_input_pos, lab_size) if s_input_pos else None
    gt_visited, gt_landing = walk_chain_forward(solv_grid, gt_first, lab_size, max_steps) if gt_first else ([], None)
    gt_chain_dirs = [solv_grid[p[0]][p[1]] for p in gt_visited]

    pred_first = None if start_violated else unique_directional_neighbor(pred_grid, s_pred_pos, lab_size)
    pred_visited, pred_landing = walk_chain_forward(pred_grid, pred_first, lab_size, max_steps) if pred_first else ([], None)
    pred_chain_dirs = [pred_grid[p[0]][p[1]] for p in pred_visited]

    all_free = all(input_grid[p[0]][p[1]] != '#' for p in pred_visited)
    is_correct = (not start_violated and not end_violated and pred_first is not None
                  and pred_landing == e_pred_pos and all_free)
    if is_correct:
        is_otima = (pred_chain_dirs == gt_chain_dirs) or (len(pred_chain_dirs) == len(gt_chain_dirs))
        results['Acurácia'] = "Ótima" if is_otima else "Correta"
    else:
        results['Acurácia'] = "Incorreta"

    results['Edit Distance'] = edit_distance(list(solv_directions), list(pred_directions))

    results['Tokens Ótimos'] = len(solv_directions)
    results['Tokens Alterados'] = len(pred_directions)
    max_len = max(len(solv_directions), len(pred_directions))
    diff_count = sum(
        1 for i in range(max_len)
        if (solv_directions[i] if i < len(solv_directions) else None) !=
           (pred_directions[i] if i < len(pred_directions) else None)
    )
    results['Tokens Diferentes'] = diff_count

    if pred_first is None:
        progresso_direto = 0
    else:
        k = 0
        while k < len(gt_chain_dirs) and k < len(pred_chain_dirs) and pred_chain_dirs[k] == gt_chain_dirs[k]:
            k += 1
        bonus = (k == len(gt_chain_dirs) == len(pred_chain_dirs)) and (pred_landing == e_pred_pos) and not end_violated
        progresso_direto = k + (1 if bonus else 0)
    results['Progresso Direto'] = progresso_direto

    fallback_dist = lab_size ** 2
    if s_pred_pos is None or e_pred_pos is None:
        results['Distância Direta'] = fallback_dist
    else:
        dist_first = unique_directional_neighbor(pred_grid, s_pred_pos, lab_size)
        if dist_first is None:
            dd = bfs_distance(input_grid, s_pred_pos, e_pred_pos, lab_size)
        else:
            stop_pos = walk_chain_forward_blocked(pred_grid, input_grid, dist_first, lab_size, max_steps)
            dd = 0 if stop_pos == e_pred_pos else bfs_distance(input_grid, stop_pos, e_pred_pos, lab_size)
        results['Distância Direta'] = dd if dd != -1 else fallback_dist

    paredes_violadas = 0
    for r in range(lab_size):
        for c in range(lab_size):
            if input_grid[r][c] == '#' and pred_grid[r][c] != '#':
                paredes_violadas += 1
    results['Paredes Violadas'] = paredes_violadas

    score = {
        'Acurácia': 1.0 if results['Acurácia'] == 'Ótima' else 0.8 if results['Acurácia'] == 'Correta' else 0.0,
        'Edit Distance': 1.0 - min(results['Edit Distance'] / max(results['Tokens Ótimos'], 1), 1.0),
        'Tokens Diferentes': 1.0 - min(results['Tokens Diferentes'] / max(results['Tokens Ótimos'], 1), 1.0),
        'Progresso Direto': results['Progresso Direto'] / (results['Tokens Ótimos']),
        'Distância Direta': (1 / (results['Distância Direta'] + 1)) ** 0.5,
        'Paredes Violadas': 1 / (results['Paredes Violadas'] + 1),
    }

    results['Corretude'] = sum(score.values()) / len(score)
    results['Corretude'] = max(0.0, min(int(results['Corretude'] * 1000) / 1000.0, 1.0))
    score['Corretude'] = results['Corretude']

    return results, score


def decoded_to_matrix(decoded):
    dec = decoded.split('\n')
    res = []
    for line in range(1, len(dec) - 1):
        mline = []
        for i in range(0, len(dec[line]), 2):
            mline.append(dec[line][i])
        res.append(mline)
    return res


def ids_to_matrix(ids, id_to_char, lab_size, extract_type="prompt"):
    token_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    chars = [id_to_char.get(t, '') for t in token_list]

    sol_start, sol_end = -1, -1
    for i, c in enumerate(chars):
        if c in ['<SOLUTION_START>', '<COMPLETION_START>']:
            sol_start = i
        elif c in ['<SOLUTION_END>', '<COMPLETION_END>']:
            sol_end = i
            break

    if extract_type == "solution":
        if sol_end == -1:
            sol_end = len(chars)
        block_chars = chars[sol_start + 1: sol_end]
    else:
        end_idx = sol_start if sol_start != -1 else len(chars)
        block_chars = chars[:end_idx]

    valid_chars = [c for c in block_chars if c != '\n' and not c.startswith('<') and c != '<PAD>']

    rows = []
    for i in range(0, len(valid_chars), lab_size):
        row = valid_chars[i:i + lab_size]
        row = (row + ['#'] * lab_size)[:lab_size]
        rows.append(row)
    while len(rows) < lab_size:
        rows.append(['#'] * lab_size)
    return rows[:lab_size]