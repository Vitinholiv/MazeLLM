#include <functional>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <queue>
#include <stack>
#include <map>
using namespace std;
namespace fs = std::filesystem;
#define NOTHING 1
#define TESTSETUP

//-------------------- Parameters --------------------

int NUM_MAZES_TO_GENERATE = 1000; // 100000 was used for Train, 1000 was used for Test
string DATA_TYPE = "TEST"; // TRAIN | TEST
string OUTPUT_FILENAME = "dataset.txt";
int SEED = 15813; // 420 was used for Train, 15813 was used for Test

int LAB_MIN_WIDTH = 10 * NOTHING, LAB_MAX_WIDTH = 10 * NOTHING;
int LAB_MIN_HEIGHT = 10 * NOTHING, LAB_MAX_HEIGHT = 10 * NOTHING;
int PADDING_TO_W = 10 * NOTHING, PADDING_TO_H = 10 * NOTHING;
string PADDING_TYPE = "evenly"; // evenly | random | start | end | none

string START_IN_POS = "random"; // random | start | end | top | bottom | left | right 
string END_IN_POS = "random"; // random | start | end | top | bottom | left | right 

string LABYRINTH_TOKENS = "individual"; // individual | wall_encoded | free_edges

string OUTPUT_TO_FORMAT = "completion"; // directions | completion
int MIN_SOLUTION_LENGTH = 25;

map<string, double> CHOSEN_DISTRIBUTION = {
    {"dfs", 0.5},
    {"wilson", 0.2},
    {"percolation", 0.0}, 
    {"percolation_dfs", 0.3}
};

//-------------------- Definitions --------------------

#define pii pair<int,int>
std::mt19937 rng;

int MAZE_MIN_WIDTH = 2*LAB_MIN_WIDTH+1, MAZE_MAX_WIDTH = 2*LAB_MAX_WIDTH+1;
int MAZE_MIN_HEIGHT = 2*LAB_MIN_HEIGHT+1, MAZE_MAX_HEIGHT = 2*LAB_MAX_HEIGHT+1;
int PADDING_TO_WIDTH = 2*PADDING_TO_W+1; int PADDING_TO_HEIGHT = 2*PADDING_TO_H+1;

//---------------------- Helpers ----------------------

vector<string> full_wall_labyrinth(int h, int w){
    string wstr = "";
    for(int i = 0; i < w; i++) wstr += "#";
    vector<string> maze(h, wstr);
    return maze;
}

vector<string> empty_labyrinth(int h, int w){
    string wstr = "";
    for(int i = 0; i < h; i++) wstr += " ";
    vector<string> maze (h, wstr);
    return maze;
}

//-------------------- Generators --------------------

vector<string> generate_labyrinth_dfs(int h, int w){
    vector<string> maze = full_wall_labyrinth(h, w);

    vector<vector<bool>> vis(h, vector<bool>(w, false));
    function<vector<pii>(int,int)> get_neighbors = [&](int l, int c) -> vector<pii> {
        vector<pii> neighs;
        if(l > 1) neighs.push_back({l-2,c});
        if(c > 1) neighs.push_back({l,c-2});
        if(l < h-2) neighs.push_back({l+2,c});
        if(c < w-2) neighs.push_back({l,c+2});
        return neighs;
    };

    stack<pii> stk;
    stk.push({1, 1});
    vis[1][1] = true;
    maze[1][1] = ' ';

    while(!stk.empty()){
        int l = stk.top().first;
        int c = stk.top().second;

        vector<pii> all_neighs = get_neighbors(l, c);
        vector<pii> unvis;
        for(auto& n : all_neighs){
            if(!vis[n.first][n.second]){
                unvis.push_back(n);
            }
        }

        if(!unvis.empty()){
            pii next_cell = unvis[rng() % unvis.size()];
            maze[(l + next_cell.first) / 2][(c + next_cell.second) / 2] = ' ';
            maze[next_cell.first][next_cell.second] = ' ';
            
            vis[next_cell.first][next_cell.second] = true;
            stk.push(next_cell);
        } else {
            stk.pop();
        }
    }
    return maze;
}

vector<string> generate_labyrinth_wilson(int h, int w){
    vector<string> maze = full_wall_labyrinth(h, w);
    vector<vector<bool>> inmaze(h, vector<bool>(w, false));

    auto get_neighbors = [&](int l, int c) -> vector<pii> {
        vector<pii> neighs;
        if(l > 1) neighs.push_back({l-2, c});
        if(c > 1) neighs.push_back({l, c-2});
        if(l < h-2) neighs.push_back({l+2, c});
        if(c < w-2) neighs.push_back({l, c+2});
        return neighs;
    };

    inmaze[1][1] = true;
    maze[1][1] = ' ';

    vector<pii> unvis;
    for(int i = 1; i < h; i += 2){
        for(int j = 1; j < w; j += 2){
            if(i != 1 || j != 1) unvis.push_back({i, j});
        }
    }

    while(!unvis.empty()){
        pii start = unvis.back();
        if(inmaze[start.first][start.second]){
            unvis.pop_back();
            continue;
        }

        map<pii, pii> walks;
        pii curr = start;

        while(!inmaze[curr.first][curr.second]){
            vector<pii> neighs = get_neighbors(curr.first, curr.second);
            pii next_cell = neighs[rng() % neighs.size()];
            walks[curr] = next_cell;
            curr = next_cell;
        }

        curr = start;
        while(!inmaze[curr.first][curr.second]){
            inmaze[curr.first][curr.second] = true;
            maze[curr.first][curr.second] = ' ';
            
            pii next_cell = walks[curr];
            maze[(curr.first + next_cell.first) / 2][(curr.second + next_cell.second) / 2] = ' ';
            
            curr = next_cell;
        }
    }
    return maze;
}

vector<string> generate_labyrinth_percolation(int h, int w, double p = 0.4){
    vector<string> maze = full_wall_labyrinth(h, w);
    
    for(int i = 1; i < h; i += 2){
        for(int j = 1; j < w; j += 2){
            maze[i][j] = ' ';
        }
    }

    uniform_real_distribution<double> dist(0.0, 1.0);
    
    for(int i = 2; i < h - 1; i += 2){
        for(int j = 1; j < w - 1; j += 2){
            if(dist(rng) < p) maze[i][j] = ' ';
        }
    }
    for(int i = 1; i < h - 1; i += 2){
        for(int j = 2; j < w - 1; j += 2){
            if(dist(rng) < p) maze[i][j] = ' ';
        }
    }

    return maze;
}

vector<string> generate_labyrinth_percolation_dfs(int h, int w, double p = 0.15){
    vector<string> maze = generate_labyrinth_dfs(h, w);
    uniform_real_distribution<double> dist(0.0, 1.0);

    for(int i = 1; i < h - 1; i++){
        for(int j = 1; j < w - 1; j++){
            if(i % 2 == 0 && j % 2 == 0) continue;
            if(maze[i][j] == '#'){
                bool vertical_connect = (maze[i-1][j] == ' ' && maze[i+1][j] == ' ');
                bool horizontal_connect = (maze[i][j-1] == ' ' && maze[i][j+1] == ' ');

                if(vertical_connect || horizontal_connect){
                    if(dist(rng) < p){
                        maze[i][j] = ' ';
                    }
                }
            }
        }
    }
    return maze;
}

//------------------------ Solving ------------------------

vector<pii> get_candidates(const vector<string>& maze, string type){
    vector<pii> c;
    int h = maze.size(), w = maze[0].size();
    for(int r = 1; r < h; r += 2){
        for(int c_idx = 1; c_idx < w; c_idx += 2){
            if(maze[r][c_idx] != ' ') continue;
            if(type == "start" && (r!=1 || c_idx!=1)) continue;
            if(type == "end" && (r!=h-2 || c_idx!=w-2)) continue;
            if(type == "top" && r!=1) continue;
            if(type == "bottom" && r!=h-2) continue;
            if(type == "left" && c_idx!=1) continue;
            if(type == "right" && c_idx!=w-2) continue;
            c.push_back({r, c_idx});
        }
    }
    return c.empty() ? vector<pii>{{1, 1}} : c;
}

vector<pii> find_path(const vector<string>& maze, pii start, pii end){
    int h = maze.size(), w = maze[0].size();
    vector<vector<pii>> parent(h, vector<pii>(w, {-1, -1}));
    queue<pii> q;
    
    q.push(start); parent[start.first][start.second] = start;
    int dr[] = {-1, 1, 0, 0}, dc[] = {0, 0, -1, 1};

    while(!q.empty()){
        pii u = q.front(); q.pop();
        if(u == end) break;
        for(int i=0; i<4; i++){
            int nr = u.first+dr[i], nc = u.second+dc[i];
            if(nr>0 && nr<h && nc>0 && nc<w && maze[nr][nc]!='#' && parent[nr][nc].first==-1){
                parent[nr][nc] = u;
                q.push({nr, nc});
            }
        }
    }
    if(parent[end.first][end.second].first == -1) return {};

    vector<pii> path;
    for(pii curr = end; curr != start; curr = parent[curr.first][curr.second]) path.push_back(curr);
    path.push_back(start);
    reverse(path.begin(), path.end());
    return path;
}

bool set_start_end_and_solve(vector<string>& maze, vector<pii>& path){
    vector<pii> starts = get_candidates(maze, START_IN_POS);
    vector<pii> ends = get_candidates(maze, END_IN_POS);

    for(int tries = 0; tries < 25; tries++){
        pii S = starts[rng() % starts.size()];
        pii E = ends[rng() % ends.size()];
        if(S == E) continue;

        path = find_path(maze, S, E);
        if(path.size() >= MIN_SOLUTION_LENGTH){
            maze[S.first][S.second] = 'S';
            maze[E.first][E.second] = 'E';
            return true;
        }
    }
    return false;
}

//-------------------- Post-Processing --------------------

void apply_padding(vector<string>& maze, vector<pii>& path){
    if(PADDING_TYPE == "none") return;
    int h = maze.size(), w = maze[0].size();
    int pad_h = max(0, PADDING_TO_HEIGHT - h);
    int pad_w = max(0, PADDING_TO_WIDTH - w);
    
    if(pad_h == 0 && pad_w == 0) return;

    int off_r = 0, off_c = 0;

    if(PADDING_TYPE == "end"){
        off_r = 0; off_c = 0;
    } 
    else if(PADDING_TYPE == "start"){
        off_r = pad_h; off_c = pad_w;
    } 
    else if(PADDING_TYPE == "evenly"){
        off_r = pad_h / 2; off_c = pad_w / 2;
    } 
    else if(PADDING_TYPE == "random"){
        uniform_int_distribution<int> dr(0, pad_h/2);
        uniform_int_distribution<int> dc(0, pad_w/2);
        off_r = dr(rng) * 2; off_c = dc(rng) * 2;
    }

    off_r -= (off_r % 2);
    off_c -= (off_c % 2);

    vector<string> padded = full_wall_labyrinth(PADDING_TO_HEIGHT, PADDING_TO_WIDTH);
    for(int r = 0; r < h; r++){
        for(int c = 0; c < w; c++){
            padded[r + off_r][c + off_c] = maze[r][c];
        }
    }
    maze = padded;

    for(auto& p : path){
        p.first += off_r;
        p.second += off_c;
    }
}

//-------------------- Representations --------------------

string represent_labyrinth(const vector<string>& m){
    int h = m.size();
    if(h == 0) return "<LABYRINTH_START>\n<LABYRINTH_END>";
    int w = m[0].size();
    
    string result = "<LABYRINTH_START>\n";

    if(LABYRINTH_TOKENS == "individual"){
        for(int r = 0; r < h; r++){
            for(int c = 0; c < w; c++){
                result += m[r][c];
                if(c < w - 1) result += " ";
            }
            if(r < h - 1) result += "\n";
        }
    } 
    else if(LABYRINTH_TOKENS == "wall_encoded"){
        for(int r = 1; r < h; r += 2){
            for(int c = 1; c < w; c += 2){
                char ch = m[r][c];
                if(ch == ' ') ch = '.';
                
                int U = 0, D = 0, L = 0, R = 0;
                if(m[r-1][c] != '#') U = 1;
                if(m[r+1][c] != '#') D = 1;
                if(m[r][c-1] != '#') L = 1;
                if(m[r][c+1] != '#') R = 1;
                
                result += string(1, ch) + to_string(U) + to_string(D) + to_string(L) + to_string(R);
                if(c < w - 2) result += " "; 
            }
            if(r < h - 2) result += "\n";
        }
    } 
    else if(LABYRINTH_TOKENS == "free_edges"){
        result += "<ADJLIST_START>\n";
        bool first = true;
        for(int r = 1; r < h; r += 2){
            for(int c = 1; c < w; c += 2){
                int log_r = r / 2;
                int log_c = c / 2;
                
                if(c + 2 < w && m[r][c+1] != '#'){
                    if(!first) result += " ; ";
                    result += "(" + to_string(log_r) + "," + to_string(log_c) + ") <-> (" + to_string(log_r) + "," + to_string(log_c+1) + ")";
                    first = false;
                }
                if(r + 2 < h && m[r+1][c] != '#'){
                    if(!first) result += " ; ";
                    result += "(" + to_string(log_r) + "," + to_string(log_c) + ") <-> (" + to_string(log_r+1) + "," + to_string(log_c) + ")";
                    first = false;
                }
            }
        }
        result += "\n<ADJLIST_END>\n";
        
        string s_pos = "", e_pos = "";
        for(int r = 1; r < h; r += 2){
            for(int c = 1; c < w; c += 2){
                if(m[r][c] == 'S') s_pos = "(" + to_string(r/2) + "," + to_string(c/2) + ")";
                if(m[r][c] == 'E') e_pos = "(" + to_string(r/2) + "," + to_string(c/2) + ")";
            }
        }
        if(!s_pos.empty()) result += "<ORIGIN> " + s_pos + "\n";
        if(!e_pos.empty()) result += "<TARGET> " + e_pos + "\n";
    }

    if(LABYRINTH_TOKENS != "free_edges") result += "\n";
    result += "<LABYRINTH_END>";
    return result;
}

string format_solution(vector<string> maze, const vector<pii>& path){
    if(path.empty()) return "<NO_SOLUTION>";

    bool is_abstract = (LABYRINTH_TOKENS == "wall_encoded" || LABYRINTH_TOKENS == "free_edges");
    vector<pii> logic_path;
    for(auto p : path) if(p.first % 2 != 0 && p.second % 2 != 0) logic_path.push_back(p);
    const vector<pii>& active_path = is_abstract ? logic_path : path;

    if(OUTPUT_TO_FORMAT == "completion"){
        if(LABYRINTH_TOKENS == "free_edges"){
            string res = "<SOLUTION_START>\n";
            for(size_t i = 0; i < logic_path.size() - 1; i++){
                res += "(" + to_string(logic_path[i].first/2) + "," + to_string(logic_path[i].second/2) + ") <-> (" + 
                       to_string(logic_path[i+1].first/2) + "," + to_string(logic_path[i+1].second/2) + ")";
                if(i < logic_path.size() - 2) res += " ; ";
            }
            return res + "\n<SOLUTION_END>";
        }

        for(size_t i = 1; i < path.size() - 1; i++){
            int dr = path[i+1].first - path[i].first;
            int dc = path[i+1].second - path[i].second;
            char dir = '*';
            if(dr > 0) dir = 'D'; else if(dr < 0) dir = 'U';
            else if(dc > 0) dir = 'R'; else if(dc < 0) dir = 'L';
            maze[path[i].first][path[i].second] = dir;
        }

        string res = "<SOLUTION_START>\n";
        if(LABYRINTH_TOKENS == "wall_encoded"){
            int h = maze.size(), w = maze[0].size();
            for(int r = 1; r < h; r += 2){
                for(int c = 1; c < w; c += 2){
                    char ch = maze[r][c];
                    if(ch == ' ') ch = '.';
                    
                    int U = 0, D = 0, L = 0, R = 0;
                    if(maze[r-1][c] != '#') U = 1;
                    if(maze[r+1][c] != '#') D = 1;
                    if(maze[r][c-1] != '#') L = 1;
                    if(maze[r][c+1] != '#') R = 1;
                    
                    res += string(1, ch) + to_string(U) + to_string(D) + to_string(L) + to_string(R);
                    if(c < w - 2) res += " "; 
                }
                if(r < h - 2) res += "\n";
            }
        }
        else {
            for(int r = 0; r < maze.size(); r++){
                for(int c = 0; c < maze[0].size(); c++){
                    res += maze[r][c];
                    if(c < maze[0].size() - 1) res += " ";
                }
                if(r < maze.size() - 1) res += "\n";
            }
        }

        return res + "\n<SOLUTION_END>";
    } 
    else if(OUTPUT_TO_FORMAT == "directions"){
        string res = "<SOLUTION_START>\n";
        for(size_t i = 0; i < active_path.size() - 1; i++){
            int dr = active_path[i+1].first - active_path[i].first;
            int dc = active_path[i+1].second - active_path[i].second;
            if(dr > 0) res += "D"; else if(dr < 0) res += "U";
            else if(dc > 0) res += "R"; else if(dc < 0) res += "L";
            if(i < active_path.size() - 2) res += " ";
        }
        return res + "\n<SOLUTION_END>";
    }
    return "<ERROR: FORMAT UNKNOWN>";
}

//-------------------- Pipeline Central --------------------

void generate_dataset_file(map<string, double> gen_probs = {}){
    rng.seed(SEED);
    cout << "Architecture: " << LABYRINTH_TOKENS << " | Solution Format: " << OUTPUT_TO_FORMAT << " | Seed: " << SEED << endl;

    if(LABYRINTH_TOKENS == "individual"){
        OUTPUT_FILENAME = "Simple_" + OUTPUT_FILENAME;
    } else if(LABYRINTH_TOKENS == "wall_encoded"){
        OUTPUT_FILENAME = "WallEncoded_" + OUTPUT_FILENAME;
    } else if(LABYRINTH_TOKENS == "free_edges"){
        OUTPUT_FILENAME = "Edgelist_" + OUTPUT_FILENAME;
    }

    string type_dir = DATA_TYPE;
    string format_dir = OUTPUT_TO_FORMAT;
    
    transform(type_dir.begin(), type_dir.end(), type_dir.begin(), ::tolower);
    transform(format_dir.begin(), format_dir.end(), format_dir.begin(), ::tolower);

    fs::path dir_path = fs::path(".") / type_dir / format_dir;
    if(!fs::exists(dir_path)){
        fs::create_directories(dir_path);
    }

    fs::path base_file(OUTPUT_FILENAME);
    string stem = base_file.stem().string();
    string ext = base_file.extension().string();
    
    fs::path final_path = dir_path / base_file;

    if(fs::exists(final_path)){
        int version = 1;
        fs::path temp_path = dir_path / (stem + "_" + to_string(version) + ext);
        while(fs::exists(temp_path)){
            version++;
            temp_path = dir_path / (stem + "_" + to_string(version) + ext);
        }
        final_path = temp_path;
        cout << "WARNING: Already existing file. Saving as: " << final_path.string() << endl;
    }

    ofstream file(final_path);
    if(!file.is_open()){
        cerr << "Erro ao abrir o arquivo " << final_path.string() << endl;
        return;
    }

    vector<string> methods = {"dfs", "wilson", "percolation", "percolation_dfs"};
    vector<double> weights;
    
    if(gen_probs.empty()){
        weights = {1.0, 1.0, 1.0, 1.0};
    } else {
        for(const string& m : methods){
            weights.push_back(gen_probs.count(m) ? gen_probs[m] : 0.0);
        }
    }
    discrete_distribution<int> generator_dist(weights.begin(), weights.end());

    int success_count = 0;
    int attempts = 0;

    while(success_count < NUM_MAZES_TO_GENERATE){
        attempts++;
        
        int h_cells = LAB_MIN_HEIGHT + (rng() % (LAB_MAX_HEIGHT - LAB_MIN_HEIGHT + 1));
        int w_cells = LAB_MIN_WIDTH + (rng() % (LAB_MAX_WIDTH - LAB_MIN_WIDTH + 1));
        int h = 2 * h_cells + 1;
        int w = 2 * w_cells + 1;

        int method_idx = generator_dist(rng);
        string chosen_method = methods[method_idx];
        vector<string> maze;
        
        if(chosen_method == "dfs") maze = generate_labyrinth_dfs(h, w);
        else if(chosen_method == "wilson") maze = generate_labyrinth_wilson(h, w);
        else if(chosen_method == "percolation") maze = generate_labyrinth_percolation(h, w, 0.4);
        else maze = generate_labyrinth_percolation_dfs(h, w, 0.20);

        vector<pii> path;
        bool has_solution = set_start_end_and_solve(maze, path);
        
        if(!has_solution || path.size() < MIN_SOLUTION_LENGTH) continue;
        apply_padding(maze, path);
        
        string maze_str = represent_labyrinth(maze);
        string sol_str = format_solution(maze, path);
        
        if(sol_str.find("<ERROR") != string::npos){
            cerr << "Critical Error: " << sol_str << endl;
            break; 
        }
        file << maze_str << "\n" << sol_str;
        success_count++;
        if(success_count < NUM_MAZES_TO_GENERATE) file << "\n\n";

        if(success_count % (NUM_MAZES_TO_GENERATE/10 > 0 ? NUM_MAZES_TO_GENERATE/10 : 1) == 0){
            cout << "Progress: " << success_count << "/" << NUM_MAZES_TO_GENERATE << endl;
        }
    }

    file.close();
    cout << "Generation finalized. " << success_count << " labyrinths genereated.\n";
}

//---------------------- Execution ----------------------
#ifdef TESTSETUP
int main(){
    SEED = 15813;
    NUM_MAZES_TO_GENERATE = 1000;
    LAB_MIN_WIDTH = 10 * 1; LAB_MAX_WIDTH = 10 * 1;
    LAB_MIN_HEIGHT = 10 * 1; LAB_MAX_HEIGHT = 10 * 1;
    PADDING_TO_W = 10 * 1; PADDING_TO_H = 10 * 1;
    PADDING_TYPE = "evenly";
    START_IN_POS = "random";
    END_IN_POS = "random";
    LABYRINTH_TOKENS = "individual";
    MIN_SOLUTION_LENGTH = 25;
    DATA_TYPE = "TEST";

    OUTPUT_FILENAME = "dataset_21.txt";
    OUTPUT_TO_FORMAT = "completion";
    generate_dataset_file(CHOSEN_DISTRIBUTION);
    OUTPUT_TO_FORMAT = "directions";
    generate_dataset_file(CHOSEN_DISTRIBUTION);
    
    OUTPUT_FILENAME = "dataset_25.txt";
    LAB_MIN_HEIGHT += 2; LAB_MAX_HEIGHT += 2;
    LAB_MIN_WIDTH += 2; LAB_MAX_WIDTH += 2;
    PADDING_TO_H += 2; PADDING_TO_W += 2;
    MIN_SOLUTION_LENGTH = 30;
    OUTPUT_TO_FORMAT = "completion";
    generate_dataset_file(CHOSEN_DISTRIBUTION);
    OUTPUT_TO_FORMAT = "directions";
    generate_dataset_file(CHOSEN_DISTRIBUTION);

    OUTPUT_FILENAME = "dataset_29.txt";
    LAB_MIN_HEIGHT += 2; LAB_MAX_HEIGHT += 2;
    LAB_MIN_WIDTH += 2; LAB_MAX_WIDTH += 2;
    PADDING_TO_H += 2; PADDING_TO_W += 2;
    MIN_SOLUTION_LENGTH = 36;
    OUTPUT_TO_FORMAT = "completion";
    generate_dataset_file(CHOSEN_DISTRIBUTION);
    OUTPUT_TO_FORMAT = "directions";
    generate_dataset_file(CHOSEN_DISTRIBUTION);

    OUTPUT_FILENAME = "dataset_33.txt";
    LAB_MIN_HEIGHT += 2; LAB_MAX_HEIGHT += 2;
    LAB_MIN_WIDTH += 2; LAB_MAX_WIDTH += 2;
    PADDING_TO_H += 2; PADDING_TO_W += 2;
    MIN_SOLUTION_LENGTH = 44;
    OUTPUT_TO_FORMAT = "completion";
    generate_dataset_file(CHOSEN_DISTRIBUTION);
    OUTPUT_TO_FORMAT = "directions";
    generate_dataset_file(CHOSEN_DISTRIBUTION);

    OUTPUT_FILENAME = "dataset_37.txt";
    LAB_MIN_HEIGHT += 2; LAB_MAX_HEIGHT += 2;
    LAB_MIN_WIDTH += 2; LAB_MAX_WIDTH += 2;
    PADDING_TO_H += 2; PADDING_TO_W += 2;
    MIN_SOLUTION_LENGTH = 55;
    OUTPUT_TO_FORMAT = "completion";
    generate_dataset_file(CHOSEN_DISTRIBUTION);
    OUTPUT_TO_FORMAT = "directions";
    generate_dataset_file(CHOSEN_DISTRIBUTION);

    LAB_MIN_WIDTH = 4 * 1; LAB_MAX_WIDTH = 10 * 1;
    LAB_MIN_HEIGHT = 4 * 1; LAB_MAX_HEIGHT = 10 * 1;
    PADDING_TO_W = 10 * 1; PADDING_TO_H = 10 * 1;
    MIN_SOLUTION_LENGTH = 18;
    OUTPUT_FILENAME = "dataset_9_to_21.txt";
    OUTPUT_TO_FORMAT = "completion";
    generate_dataset_file(CHOSEN_DISTRIBUTION);
    OUTPUT_TO_FORMAT = "directions";
    generate_dataset_file(CHOSEN_DISTRIBUTION);

    LAB_MIN_WIDTH = 10 * 1; LAB_MAX_WIDTH = 10 * 1;
    LAB_MIN_HEIGHT = 10 * 1; LAB_MAX_HEIGHT = 10 * 1;
    PADDING_TO_W = 10 * 1; PADDING_TO_H = 10 * 1;

    CHOSEN_DISTRIBUTION = {
        {"dfs", 1.0},
        {"wilson", 0.0},
        {"percolation", 0.0}, 
        {"percolation_dfs", 0.0}
    };
    OUTPUT_FILENAME = "dfs_21.txt";
    OUTPUT_TO_FORMAT = "completion";
    generate_dataset_file(CHOSEN_DISTRIBUTION);
    OUTPUT_TO_FORMAT = "directions";
    generate_dataset_file(CHOSEN_DISTRIBUTION);

    CHOSEN_DISTRIBUTION = {
        {"dfs", 0.0},
        {"wilson", 1.0},
        {"percolation", 0.0}, 
        {"percolation_dfs", 0.0}
    };
    OUTPUT_FILENAME = "wilson_21.txt";
    OUTPUT_TO_FORMAT = "completion";
    generate_dataset_file(CHOSEN_DISTRIBUTION);
    OUTPUT_TO_FORMAT = "directions";
    generate_dataset_file(CHOSEN_DISTRIBUTION);

    CHOSEN_DISTRIBUTION = {
        {"dfs", 0.0},
        {"wilson", 0.0},
        {"percolation", 0.0}, 
        {"percolation_dfs", 1.0}
    };
    OUTPUT_FILENAME = "percolation_dfs_21.txt";
    OUTPUT_TO_FORMAT = "completion";
    generate_dataset_file(CHOSEN_DISTRIBUTION);
    OUTPUT_TO_FORMAT = "directions";
    generate_dataset_file(CHOSEN_DISTRIBUTION);

    return 0;
}
#else
int main(){
    generate_dataset_file(CHOSEN_DISTRIBUTION);
    return 0;
}
#endif