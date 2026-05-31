#include <bits/stdc++.h>
using namespace std;
#define pii pair<int,int>
#define PROMPT_SKIP cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n'); continue
#define rep(i,a,b) for(int __i = a; __i <= b; __i++)

enum SaveMode {
    COMPLETION,
    DIRECTIONS
};

enum Padding {
    NONE,
    MAXIMIZE
};

struct Prompter {
    string join(const vector<string>& v, const string s){
        string restr = "";
        for(int _i = 0; _i < v.size(); _i++){
            restr += "[" + to_string(_i+1) + "] " + v[_i];
            if(_i < v.size()-1) restr += s;
        }
        return restr;
    }
    int prompt_for_int(const string& message) {
        int res;
        while(true){
            cout << message << ": ";
            if(cin >> res){
                return res;
            } else {
                PROMPT_SKIP;
            }
        }
    }
    string prompt_options(const string& message, const vector<string>& opts){
        int res;
        while(true){
            cout << message << " (" << join(opts," | ") << "): ";
            if(cin >> res){
                if(res > opts.size() || res <= 0) continue;
                return opts[res-1];
            } else {
                PROMPT_SKIP;
            }
        }
    }
    string get_timestamp(){
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        stringstream ss;
        ss << put_time(&tm, "%Y%m%d_%H%M%S");
        return ss.str();
    }
};

class MazeGenerator {
protected:
    pii n; vector<string> grid, solution; mt19937 rng;
public:
    MazeGenerator(){ random_device rd; rng = mt19937(rd()); }
    virtual ~MazeGenerator() = default;
    virtual void generate(pii size) = 0;
    virtual void random_generation() = 0;
    virtual void solve() = 0;
    virtual string textify(int type, int padding) = 0;

    string stringify(vector<string>& g, pair<int,string> pad = {0,""}){
        string slv = "", fullrow = "", lrow = "", rrow = "";

        int df = max(n.first,pad.first) - n.first, ds = max(n.second,pad.first) - n.second;
        int dfl = randint(0,df), dfr = df - dfl;
        int dsl = randint(0,ds), dsr = ds - dsl;

        rep(z,1,max(n.second,pad.first)) fullrow += pad.second;
        rep(z,1,dsl) lrow += pad.second;
        rep(z,1,dsr) rrow += pad.second;

        rep(z,1,dfl) slv += fullrow + "\n";
        for(const string& row : g){
            slv += lrow + row + rrow + "\n";
        }
        rep(z,1,dfr) slv += fullrow + "\n";

        return slv;
    }

    void save(int num, const string& filename, int format = COMPLETION, int padding = MAXIMIZE) {
        string filepath = filename + ".txt";
        ofstream outfile(filepath);
        if(!outfile.is_open()){
            cerr << "Failed to open " << filepath << endl;
            return;
        }
        cout << "Building " << num << " mazes..." << endl;
        for(int i = 0; i < num; i++) {
            random_generation(); 
            outfile << textify(format,padding);
            if(i < num - 1) outfile << "\n";
        }
        outfile.close();
        cout << "Saved to " << filepath << endl;
    }

    int randint(int lower, int upper){
        uniform_int_distribution<int> _randy(lower, upper);
        return _randy(rng);
    }

    double randreal(double lower, double upper){
        uniform_real_distribution<double> _randy(lower,upper);
        return _randy(rng);
    }

};

class PathCarver : public MazeGenerator {
private:
    int minSize = 5, maxSize = 20; double sparsity = 0.1;
    pii start, end; string directions;
    vector<pii> neighborhood(int l, int c) {
        return {{l,c-1}, {l,c+1}, {l-1,c}, {l+1,c}};
    }
    bool isValidCarve(int l, int c) {
        if(l <= 0 || l >= n.first - 1 || c <= 0 || c >= n.second - 1) return false;
        if(grid[l][c] == ' ') return false; 
        
        int pathNeighbors = 0;
        for(pii u : neighborhood(l,c)){
            if(grid[u.first][u.second] == ' '){
                pathNeighbors++;
            }
        }
        return pathNeighbors == 1;
    }
public:
    PathCarver(int maxSz = 20, double probInbranch = 0.1) : maxSize(maxSz), sparsity(probInbranch) {}
    void generate(pii size) override {
        n = size; grid.assign(n.first, string(n.second, '#'));
        start = {randint(1,n.first-2),randint(1,n.second-2)}, end;

        vector<pii> active = {start};
        grid[start.first][start.second] = ' ';
        vector<pii> open;
        open.push_back(start);

        while(!active.empty()){
            int idx = active.size() - 1;
            if(randreal(0,1) < sparsity){
                idx = randint(0,active.size()-1);
            }

            pii curr = active[idx];
            vector<pii> dirs = neighborhood(curr.first, curr.second);
            shuffle(dirs.begin(), dirs.end(), rng);

            bool moved = false;
            for(pii nxt : dirs){
                if(isValidCarve(nxt.first, nxt.second)){
                    grid[nxt.first][nxt.second] = ' ';
                    active.push_back(nxt);
                    open.push_back(nxt);
                    break;
                }
            }
            if(!moved){
                active.erase(active.begin() + idx);
            }
        }

        end = open[randint(1, open.size() - 1)];
        grid[start.first][start.second] = 'S';
        grid[end.first][end.second] = 'E';
    }

    void solve() override {
        vector<vector<pii>> parent(n.first, vector<pii>(n.second, {-1,-1}));
        queue<pii> q; q.push(start);
        parent[start.first][start.second] = start;

        int dl[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};

        while(!q.empty()){
            pii curr = q.front(); q.pop();
            if(curr == end) break;

            for(int i = 0; i < 4; ++i){
                int nl = curr.first + dl[i], nc = curr.second + dc[i];
                bool inside = (nl >= 0 && nl < n.first && nc >= 0 && nc < n.second);
                if(inside && (grid[nl][nc] == ' ' || grid[nl][nc] == 'E') && parent[nl][nc].first == -1){
                    parent[nl][nc] = curr;
                    q.push({nl, nc});
                }
            }
        }

        vector<pii> path; pii step = end; directions = "";
        while(step != start){
            path.push_back(step);
            step = parent[step.first][step.second];
        }
        path.push_back(start);
        reverse(path.begin(), path.end());

        solution = grid;
        for(size_t i = 0; i < path.size() - 1; ++i){
            pii curr = path[i];
            pii next = path[i+1];

            char dir = ' ';
            if(next.second > curr.second) dir = 'R';
            else if(next.second < curr.second) dir = 'L';
            else if(next.first > curr.first) dir = 'D';
            else if(next.first < curr.first) dir = 'U';

            directions += dir;
            if(curr != start){
                solution[curr.first][curr.second] = dir;
            }
        }
    }

    string textify(int type, int padding) override {
        if(padding == NONE){
            string unsolved = stringify(grid);
            if(type == DIRECTIONS){
                return unsolved + "&\n" + directions + "\n";
            } else if(type == COMPLETION){
                string solved = stringify(solution);
                return unsolved + "&\n" + solved;
            }
        } else if(padding == MAXIMIZE){
            string unsolved = stringify(grid,{maxSize,"#"});
            if(type == DIRECTIONS){
                return unsolved + "&\n" + directions + "\n";
            } else if(type == COMPLETION){
                string solved = stringify(solution,{maxSize,"#"});
                return unsolved + "&\n" + solved;
            }
        }
        return "";
    }
    
    void random_generation() override {
        generate({randint(minSize, maxSize), randint(minSize, maxSize)});
        solve();
    }
};

int main(){
    Prompter p;
    
    vector<string> maze_types = { "Path Carver" };
    string selected = p.prompt_options("Labyrinth Type", maze_types);

    MazeGenerator* generator = nullptr;
    if(selected == "Path Carver"){
        generator = new PathCarver();

        vector<string> fopts = {"Completion", "Directions"};
        vector<string> popts = {"None", "Maximize"};

        string format = p.prompt_options("Solution Type", fopts);
        string pad = p.prompt_options("Padding Type", popts);

        SaveMode smode = (format == "Directions") ? DIRECTIONS : COMPLETION;
        Padding pmode = (pad == "None") ? NONE : MAXIMIZE;
        
        int qt = p.prompt_for_int("Amount");
        string filename = "Pathcarver_" + format + "_" + pad + "_" + p.get_timestamp();
        generator->save(qt, filename, smode, pmode);
    }
    if (generator != nullptr) delete generator;

    return 0;
}