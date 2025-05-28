#include <bits/stdc++.h>  // Inclui todas as bibliotecas padrão
#include <atomic>         // Para operações atomicas
#include <mutex>          // Para exclusão mútua entre threads
#include <cstdint>        // Tipos inteiros de tamanho fixo

using namespace std;
using pii = pair<int, int>;
static const int INF = 1e9; // Valor infinito para representar obstáculos
const double INIT_WEIGHT = 1.5; // Peso inicial da heurística
const double WEIGHT_DECAY = 0.998; // Decaimento do peso ao longo do tempo

int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1}; // Movimentos (cima, baixo, esq, dir)

int N, Tlim, W, Ksum;             // Dimensões, tempo limite, interesses, soma K
vector<string> mapa;              // Mapa do grid (caracteres)
vector<pii> interest;             // Posições dos pontos de interesse
vector<int> interestValue;        // Valores dos pontos de interesse
vector<vector<int>> distToExit;   // Distância mínima para saídas
vector<vector<int>> interestDist; // Distância mínima para interesses
int total_traversable;            // Total de células transitáveis

// COMPONENTES PARA PARALELIZAÇÃO
atomic<int> globalBestSat(-INF);  // Melhor satisfação global (thread-safe)
atomic<int> globalExpansions(0);  // Contador de expansões de nós
atomic<double> globalCpu(0.0);    // Tempo de CPU consumido
vector<pii> globalBestPath;       // Melhor caminho encontrado
mutex bestMutex;                  // Mutex para acesso ao melhor caminho

inline int moveCost(char ch) { return (ch == '#' ? INF : (ch == ':' ? 2 : 1));}

// Calcula distâncias mínimas para as saídas usando Dijkstra com priority queue
// Atualiza a matriz distToExit com as distâncias mínimas
void computeExitDistances() {
    distToExit.assign(N, vector<int>(N, INF));
    priority_queue<pair<int, pii>, vector<pair<int, pii>>, greater<>> pq;
    int mid = N / 2;
    vector<pii> doors = {{0, mid}, {N-1, mid}, {mid, 0}, {mid, N-1}};
    for (auto [r, c] : doors) {
        if (mapa[r][c] == '#') continue;
        distToExit[r][c] = 0;
        pq.emplace(0, make_pair(r, c));
    }
    while (!pq.empty()) {
        auto [d, pos] = pq.top(); pq.pop();
        auto [r, c] = pos;
        if (d > distToExit[r][c]) continue;
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
            int cost = moveCost(mapa[nr][nc]);
            if (distToExit[r][c] + cost < distToExit[nr][nc]) {
                distToExit[nr][nc] = distToExit[r][c] + cost;
                pq.emplace(distToExit[nr][nc], make_pair(nr, nc));
            }
        }
    }
}


// Heurística:
    // Valor dos interesses próximos (com bonus para proximidade)
    // Máximo valor restante possível (admissível)
    // Potencial de exploração de células não visitadas
    // Penalidade por revisitas consecutivas 
    // Considera o tempo restante e distância para saída mais próxima
    // Prioriza exploração apenas se houver tempo suficiente para sair
int heuristic(int r, int c, int t, int mask, const bitset<225>& visited_once, const bitset<225>& visited_twice, int consec) {
    int timeLeft = Tlim - t;
    int h = 0;
    int max_remaining_value = 0;

    for (int i = 0; i < W; i++) {
        if (!(mask & (1 << i))) {
            auto [ri, ci] = interest[i];
            int dist = interestDist[r][c];
            int exit_time = distToExit[ri][ci];
            
            if (dist != INF && dist + exit_time <= timeLeft) {
                int value = interestValue[i];
                if (dist <= timeLeft/3) value *= 2;
                h += value;
                max_remaining_value = max(max_remaining_value, interestValue[i]);
            }
        }
    }

    h += max_remaining_value;

    int min_exit_dist = distToExit[r][c];
    int available_exploration_time = timeLeft - min_exit_dist;
    
    int reachable_cells = 0;
    if (available_exploration_time > 0) {
        int potential_new_cells = total_traversable - 
                                 static_cast<int>(visited_once.count()) - 
                                 static_cast<int>(visited_twice.count());
        reachable_cells = min(available_exploration_time, potential_new_cells);
    }
    h += reachable_cells;

    if (consec >= 2) {
        h -= 3 * (consec - 1);
    }

    return h;
}

// Estado do nó na busca
struct State {
    int r, c, t, mask, consec; // Posição (r,c), tempo, máscara de interesses coletados
    bitset<225> visited_once;  // Controlo de visitas
    bitset<225> visited_twice; // Controlo de visitas
    int g, f;                  // Valores g (custo real) e f (g + heurística)
    bool operator<(const State& o) const { return f > o.f; } // Operador < para ordenar na priority_queue
};

// Chave única para o estado (usada no hashmap)
struct Key {
    // Componentes essenciais do estado para comparação
    int r, c, t, mask, consec;
    bitset<225> visited_once;
    bitset<225> visited_twice;
    // Operador == para verificação de igualdade
    bool operator==(const Key& o) const { 
        return tie(r, c, t, mask, consec) == tie(o.r, o.c, o.t, o.mask, o.consec)
            && visited_once == o.visited_once
            && visited_twice == o.visited_twice;
    }
};

// Função auxiliar para combinar hashes
static inline void hash_combine(size_t& seed, size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Função de hash personalizada para a chave
struct KeyHash {
    // Combina hashes de todos componentes do estado
    // Trata bitsets de forma eficiente usando blocos de 64 bits
    size_t operator()(const Key& k) const {
        size_t h = 0;
        
        // Hash primitive fields
        hash_combine(h, hash<int>{}(k.r));
        hash_combine(h, hash<int>{}(k.c));
        hash_combine(h, hash<int>{}(k.t));
        hash_combine(h, hash<int>{}(k.mask));
        hash_combine(h, hash<int>{}(k.consec));

        // Fast bitset hashing using 64-bit blocks
        auto hash_bitset = [](const bitset<225>& bs) {
            size_t hash = 0;
            for (size_t i = 0; i < 225; i += 64) {
                uint64_t chunk = 0;
                for (size_t j = 0; j < 64 && (i + j) < 225; ++j) {
                    chunk |= static_cast<uint64_t>(bs[i + j]) << j;
                }
                hash_combine(hash, std::hash<uint64_t>{}(chunk));
            }
            return hash;
        };

        hash_combine(h, hash_bitset(k.visited_once));
        hash_combine(h, hash_bitset(k.visited_twice));
        
        return h;
    }
};

struct NodeInfo {
    int g;
    Key parent;
    int parentDir;
};

void workerAstar(int doorIdx) {
    // Implementação paralela do A*:
    // Inicializa a partir de uma entrada específica
    // Usa priority_queue para fronteira de busca
    // Mantém hashmap de estados visitados
    // Expande nós considerando movimentos válidos
    // Atualiza melhor caminho encontrado
    // Usa limite 10s ou 1000000 expansões como limite
    auto t0 = chrono::steady_clock::now();
    int expansions = 0;
    priority_queue<State> pq;
    unordered_map<Key, NodeInfo, KeyHash> info;

    int mid = N / 2;
    vector<pii> doors = {{0, mid}, {N-1, mid}, {mid, 0}, {mid, N-1}};
    auto [r0, c0] = doors[doorIdx];
    if (mapa[r0][c0] == '#') return;

    int t_init = 1;
    int mask0 = 0, g0 = 0;
    bitset<225> visited_once0, visited_twice0;
    visited_once0.set(r0 * N + c0);

    // Initial interest collection
    for (int i = 0; i < W; i++) {
        if (interest[i] == make_pair(r0, c0)) {
            mask0 |= (1 << i);
            g0 += interestValue[i];
        }
    }
    g0 += 2;  // Initial cell bonus plus exit bonus

    // Initialization
    double initial_weight = INIT_WEIGHT * pow(WEIGHT_DECAY, t_init);
    int h0 = heuristic(r0, c0, t_init, mask0, visited_once0, visited_twice0, 0);
    int f0 = g0 + int(initial_weight * h0);
    
    Key k0{r0, c0, t_init, mask0, 0, visited_once0, visited_twice0};
    pq.push({r0, c0, t_init, mask0, 0, visited_once0, visited_twice0, g0, f0});
    info[k0] = {g0, k0, -1};

    int bestSat = -INF;
    Key bestKey;

    while (!pq.empty()) {
        if (++expansions > 1000000) break;
        auto now = chrono::steady_clock::now();
        if (chrono::duration<double>(now - t0).count() > 10.0) break;

        State u = pq.top(); pq.pop();
        Key uk{u.r, u.c, u.t, u.mask, u.consec, u.visited_once, u.visited_twice};

        // Early exit check with actual movement cost
        bool isExit = (u.r == 0 && u.c == mid) || (u.r == N-1 && u.c == mid) ||
                     (u.c == 0 && u.r == mid) || (u.c == N-1 && u.r == mid);
        if (isExit) {
            int exit_cost = moveCost(mapa[u.r][u.c]);
            int exit_time = u.t + exit_cost;
            if (exit_time <= Tlim && u.g > bestSat) {
                bestSat = u.g;
                bestKey = uk;
            }
        }

        for (int d = 0; d < 4; d++) {
            int nr = u.r + dr[d], nc = u.c + dc[d];
            if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
            
            char ch = mapa[nr][nc];
            if (ch == '#') continue;
            
            int mc = moveCost(ch), nt = u.t + mc;
            if (nt > Tlim) continue;

            int cell_idx = nr * N + nc;
            if (u.visited_twice.test(cell_idx)) continue;

            // Update visitation tracking
            bitset<225> n_once = u.visited_once;
            bitset<225> n_twice = u.visited_twice;
            bool is_new = !n_once.test(cell_idx) && !n_twice.test(cell_idx);

            if (n_once.test(cell_idx)) {
                n_once.reset(cell_idx);
                n_twice.set(cell_idx);
            } else if (!n_twice.test(cell_idx)) {
                n_once.set(cell_idx);
            }

            // Update interest collection
            int nmask = u.mask;
            int ng = u.g;
            for (int i = 0; i < W; i++) {
                if (!(nmask & (1 << i)) && interest[i] == make_pair(nr, nc)) {
                    nmask |= (1 << i);
                    ng += interestValue[i];
                }
            }

            // Update score components
            if (is_new) ng += 1;
            int nconsec = is_new ? 0 : u.consec + 1;
            ng -= (nconsec >= 2) ? 1 : 0;  // Progressive penalty

            Key nk{nr, nc, nt, nmask, nconsec, n_once, n_twice};
            auto it = info.find(nk);
            if (it == info.end() || ng > it->second.g) {
                int h = heuristic(nr, nc, nt, nmask, n_once, n_twice, nconsec);
                double current_weight = INIT_WEIGHT * pow(WEIGHT_DECAY, u.t);
                int f = ng + int(current_weight * h);
                
                // Prune unpromising states
                if (f > globalBestSat.load(memory_order_relaxed)) {
                    pq.push({nr, nc, nt, nmask, nconsec, n_once, n_twice, ng, f});
                    info[nk] = {ng, uk, d};
                }
            }
        }
    }

    // Path reconstruction
    vector<pii> path;
    if (bestSat > -INF) {
        Key cur = bestKey;
        while (true) {
            path.emplace_back(cur.r, cur.c);
            auto &ni = info[cur];
            if (ni.parentDir < 0) break;
            cur = ni.parent;
        }
        reverse(path.begin(), path.end());
    }

    // Update global best
    double cpu = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    if (bestSat > globalBestSat.load(memory_order_relaxed)) {
        lock_guard<mutex> lk(bestMutex);
        if (bestSat > globalBestSat.load(memory_order_relaxed)) {
            globalBestSat.store(bestSat, memory_order_relaxed);
            globalBestPath = move(path);
            globalExpansions.store(expansions, memory_order_relaxed);
            globalCpu.store(cpu, memory_order_relaxed);
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    struct Inst {
        int ID,N,K,W,T;
        vector<vector<int>> M;
    };

    vector<Inst> instancias = {
        {1,5,6,4,10,{{1,1,1,1,-1},{1,10,1,2,10},{1,-2,10,2,1},{10,10,1,1,1},{-1,1,1,2,-2}}},
        {2,5,5,4,20,{{-1,1,2,2,-1},{1,2,1,1,10},{1,1,10,2,2},{10,2,1,10,1},{-1,2,1,10,-2}}},
        {3,7,10,5,15,{{1,1,1,1,10,-2,1},{1,-2,10,1,1,10,1},{1,10,1,10,1,1,1},{1,1,-2,10,1,2,1},{2,1,10,1,2,1,10},{2,1,1,1,2,10,-3},{1,-1,10,1,1,1,1}}},
        {4,7,10,5,20,{{1,-2,2,2,2,-2,1},{1,1,1,1,1,1,1},{1,10,10,10,10,10,1},{1,1,-2,10,1,2,1},{2,2,2,10,2,10,10},{2,1,2,10,2,10,-3},{1,-1,2,1,1,1,1}}},
        {5,9,19,6,19,{{-2,10,1,1,1,1,-2,1,1},{1,2,-2,10,10,1,1,10,2},{1,1,1,1,1,10,10,1,1},{1,10,10,1,10,-2,1,1,1},{1,1,10,1,1,1,2,10,1},{2,1,1,10,1,2,10,1,1},{10,10,2,2,2,2,10,-3,1},{2,1,1,10,1,2,10,1,10},{-8,10,1,1,1,1,1,1,1}}},
        {6,11,19,8,48,{{1,-5,1,10,1,1,1,1,1,1,1},{1,1,10,1,2,1,10,1,1,1,1},{1,10,10,1,1,1,1,10,1,-3,1},{1,2,-1,10,2,2,1,2,10,1,1},{2,1,1,1,10,2,1,1,2,10,1},{1,2,1,1,1,10,-2,1,2,10,1},{2,1,2,10,1,1,10,1,1,10,1},{1,-1,1,10,1,2,1,1,10,2,1},{1,10,10,10,1,2,-1,10,2,2,1},{1,1,1,1,1,2,10,1,1,1,-4},{1,1,-2,2,1,1,1,2,2,2,1}}},
        {7,13,10,10,120,{{1,1,1,1,1,10,1,2,1,1,1,1,1},{-1,10,1,10,1,2,10,1,10,2,10,10,1},{10,1,1,10,1,2,10,2,10,1,10,-1,1},{1,1,10,1,1,1,-1,1,10,-1,1,10,10},{1,10,1,1,1,10,1,1,10,10,1,1,1},{1,10,-1,10,1,1,10,1,1,1,10,10,1},{1,10,1,1,10,1,1,10,10,1,1,10,1},{10,10,10,-1,1,10,1,1,10,1,1,10,1},{1,2,1,10,2,1,10,1,10,-1,10,1,1},{1,10,2,1,10,2,10,1,10,1,10,1,1},{1,-1,10,1,2,1,10,1,10,1,1,1,10},{1,10,1,10,10,10,10,1,10,1,10,10,-1},{1,1,1,-1,1,1,1,1,10,1,1,1,1}}},
        {8,15,38,15,30,{{-3,1,1,1,1,1,1,1,2,2,2,1,-3,1,-2},{1,1,1,1,1,10,10,10,10,10,1,1,2,1,2},{1,1,1,1,1,2,1,1,1,1,1,1,1,1,-2},{1,1,1,1,1,1,-2,2,1,-3,1,1,1,1,1},{1,1,1,1,1,1,1,2,1,2,1,1,1,1,1},{1,10,1,1,1,2,2,1,2,1,1,1,-1,10,1},{1,10,-3,1,2,1,1,2,1,1,1,1,1,10,1},{1,10,1,1,-2,2,1,1,1,1,2,2,1,10,1},{1,10,1,1,1,1,2,1,1,1,1,1,1,10,2},{1,10,-2,2,1,2,1,2,1,1,1,1,1,10,1},{1,1,1,1,1,1,1,1,1,1,2,1,1,1,2},{-2,1,1,2,1,2,1,1,1,2,1,2,1,-3,1},{1,1,1,1,2,1,1,1,1,1,1,1,1,1,1},{1,2,1,1,1,10,10,10,10,10,1,1,2,1,1},{1,1,-3,1,1,1,1,1,1,1,1,-3,1,1,-4}}},
        {9,15,38,15,45,{{-3,10,1,1,1,1,1,1,2,2,2,10,-6,1,-4},{1,10,1,10,1,10,10,10,10,10,1,1,2,10,2},{1,10,1,10,1,2,1,10,1,10,1,10,1,10,-4},{1,1,1,10,1,10,-4,10,1,-6,1,10,1,1,1},{1,10,10,10,1,10,1,10,1,10,1,10,10,10,1},{1,10,1,1,1,10,2,10,2,10,1,10,-2,10,1},{1,10,-6,10,2,10,1,1,1,10,1,10,1,10,1},{1,10,1,10,-4,10,10,10,10,10,2,2,1,10,1},{1,10,1,10,10,10,2,1,1,10,10,10,1,10,2},{1,10,-4,2,1,1,1,10,1,10,1,1,1,10,1},{1,10,10,10,1,10,1,10,1,10,2,10,10,10,2},{-4,1,1,2,1,10,1,10,1,2,1,2,1,-6,1},{10,10,1,10,2,10,1,10,1,10,1,10,10,10,10},{1,10,1,10,1,10,10,10,10,10,1,10,2,10,1},{1,1,-6,10,1,1,1,1,1,1,1,-6,1,1,-8}}},
        {10,15,76,15,45,{{-6,10,1,1,1,1,1,1,2,2,2,10,-6,1,-4},{1,10,1,10,1,10,10,10,10,10,1,1,2,10,2},{1,10,1,10,1,2,1,10,1,10,1,10,1,10,-4},{1,1,1,10,1,10,-4,10,1,-6,1,10,1,1,1},{1,10,10,10,1,10,1,10,1,10,1,10,10,10,1},{1,10,1,1,1,10,2,10,2,10,1,10,-2,10,1},{1,10,-6,10,2,10,1,1,1,10,1,10,1,10,1},{1,10,1,10,-4,10,10,10,10,10,2,2,1,10,1},{1,10,1,10,10,10,2,1,1,10,10,10,1,10,2},{1,10,-4,2,1,1,1,10,1,10,1,1,1,10,1},{1,10,10,10,1,10,1,10,1,10,2,10,10,10,2},{-4,1,1,2,1,10,1,10,1,2,1,2,1,-6,1},{10,10,1,10,2,10,1,10,1,10,1,10,10,10,10},{1,10,1,10,1,10,10,10,10,10,1,10,2,10,1},{1,1,-6,10,1,1,1,1,1,1,1,-6,1,1,-8}}}
    };
    
    // Precompute interest distances (BFS from all interests)
    interestDist.assign(N, vector<int>(N, INF));
    queue<pair<int, pii>> q;
    for (auto [r,c] : interest) {
        interestDist[r][c] = 0;
        q.emplace(0, make_pair(r, c));
    }
    
    while (!q.empty()) {
        auto [d, pos] = q.front(); q.pop();
        auto [r,c] = pos;
        if (d > interestDist[r][c]) continue;
        
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (nr >= 0 && nr < N && nc >= 0 && nc < N && mapa[nr][nc] != '#') {
                if (interestDist[nr][nc] > d + 1) {
                    interestDist[nr][nc] = d + 1;
                    q.emplace(d + 1, make_pair(nr, nc));
                }
            }
        }
    }

    for (auto &ins : instancias) {
        // Processa múltiplas instâncias:
    // Carrega dados de cada instância
    // Pré-processa distâncias para saídas e interesses
    // Inicia threads para cada porta de entrada
    // Recolhe resultados e exibe estatísticas
    // Executa para as 10 instâncias pré-definidas
    
    // Estrutura Inst armazena configurações de cada problema
    // Pré-computações são refeitas para cada instância
    // Threads são sincronizadas e resultados comparados
        N = ins.N; 
        Tlim = ins.T; 
        Ksum = ins.K;
        mapa.resize(N);
        interest.clear(); 
        interestValue.clear();

        // Parse map and interests
        for (int r = 0; r < N; r++) {
            string s;
            for (int c = 0; c < N; c++) {
                int v = ins.M[r][c];
                if (v == 10) s.push_back('#');
                else if (v == 2) s.push_back(':');
                else if (v == 1) s.push_back('.');
                else {
                    s.push_back('0' + (-v));
                    interest.emplace_back(r, c);
                    interestValue.push_back(-v);
                }
            }
            mapa[r] = s;
        }
        W = interest.size();

        // Calcular traversable cells
        total_traversable = 0;
        for (int r = 0; r < N; r++) 
            for (int c = 0; c < N; c++) 
                if (mapa[r][c] != '#') total_traversable++;

        computeExitDistances();

        // Precalcular interest distances
        interestDist.assign(N, vector<int>(N, INF));
        queue<pair<int, pii>> q;
        for (auto [r,c] : interest) {
            interestDist[r][c] = 0;
            q.emplace(0, make_pair(r, c));
        }

        while (!q.empty()) {
            auto [d, pos] = q.front(); q.pop();
            auto [r,c] = pos;
            if (d > interestDist[r][c]) continue;
            
            for (int k = 0; k < 4; k++) {
                int nr = r + dr[k], nc = c + dc[k];
                if (nr >= 0 && nr < N && nc >= 0 && nc < N && mapa[nr][nc] != '#') {
                    if (interestDist[nr][nc] > d + 1) {
                        interestDist[nr][nc] = d + 1;
                        q.emplace(d + 1, make_pair(nr, nc));
                    }
                }
            }
        }

        {
            lock_guard<mutex> lk(bestMutex);
            globalBestSat = -INF;
            globalBestPath.clear();
            globalExpansions = 0;
            globalCpu = 0.0;
        }

        // Lança as threads e resultados
        vector<thread> threads;
        for (int di = 0; di < 4; di++) 
            threads.emplace_back(workerAstar, di);
        
        for (auto &th : threads) 
            th.join();

        int ideal = Tlim + Ksum;
        int cost = ideal - globalBestSat;
        cout << "\n==== Instância " << ins.ID
             << " (N=" << N << " W=" << W << " K=" << Ksum << " T=" << Tlim << ") ====\n";
        cout << "Melhor sat: " << globalBestSat
             << "  custo: " << cost
             << "  exp: " << globalExpansions
             << "  cpu: " << fixed << setprecision(3) << globalCpu << "s\n";
        cout << "Caminho:\n";
        for (auto &p : globalBestPath)
            cout << "(" << p.first << "," << p.second << ") ";
        cout << "\n";
    }
    return 0;
}