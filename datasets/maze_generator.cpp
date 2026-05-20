#include <bits/stdc++.h>
using namespace std;
#define K *1000

// Parâmetros
const int MIN_SIZE = 5;
const int MAX_SIZE = 20;
const int NUM_MAZES = 100;
const int GRID_W = 20;
const int GRID_H = 20;

// Código
struct Point {
    int x, y;
};

int dx1[] = {0, 0, -1, 1};
int dy1[] = {-1, 1, 0, 0};

bool isValidCarve(const vector<string>& grid, int x, int y, int maze_w, int maze_h) {
    if (x <= 0 || x >= maze_w - 1 || y <= 0 || y >= maze_h - 1) return false;
    if (grid[y][x] == ' ') return false; 
    
    int path_neighbors = 0;
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx1[i];
        int ny = y + dy1[i];
        if (grid[ny][nx] == ' ') {
            path_neighbors++;
        }
    }
    return path_neighbors == 1;
}

string generate_solved_maze(mt19937& rng) {
    vector<string> grid(GRID_H, string(GRID_W, '#'));
    
    uniform_int_distribution<int> dist_w(MIN_SIZE, MAX_SIZE);
    uniform_int_distribution<int> dist_h(MIN_SIZE, MAX_SIZE);
    int maze_w = dist_w(rng); 
    int maze_h = dist_h(rng); 
    
    vector<Point> active;
    active.push_back({1, 1});
    grid[1][1] = ' ';
    
    uniform_real_distribution<double> chance(0.0, 1.0);
    
    while (!active.empty()) {
        int idx = active.size() - 1; 
        if (chance(rng) < 0.10) {
            uniform_int_distribution<int> rand_idx(0, active.size() - 1);
            idx = rand_idx(rng);
        }
        
        Point curr = active[idx];
        vector<int> dirs = {0, 1, 2, 3};
        shuffle(dirs.begin(), dirs.end(), rng);
        
        bool moved = false;
        for (int i : dirs) {
            int nx = curr.x + dx1[i];
            int ny = curr.y + dy1[i];
            
            if (isValidCarve(grid, nx, ny, maze_w, maze_h)) {
                grid[ny][nx] = ' ';
                active.push_back({nx, ny});
                moved = true;
                break;
            }
        }
        
        if (!moved) {
            active.erase(active.begin() + idx);
        }
    }

    Point start = {1, 1};
    Point end = {-1, -1};
    
    for (int y = maze_h - 2; y > 0 && end.y == -1; --y) {
        for (int x = maze_w - 2; x > 0 && end.x == -1; --x) {
            if (grid[y][x] == ' ') {
                end = {x, y};
            }
        }
    }
    
    vector<string> unsolved_grid = grid;
    unsolved_grid[start.y][start.x] = 'S';
    if (end.x != -1) unsolved_grid[end.y][end.x] = 'E';
    
    string unsolved_output = "";
    for (int y = 0; y < GRID_H; ++y) unsolved_output += unsolved_grid[y] + "\n";
    
    if (start.x == end.x && start.y == end.y) {
        return unsolved_output + "&\n" + unsolved_output;
    }

    vector<vector<Point>> parent(GRID_H, vector<Point>(GRID_W, {-1, -1}));
    queue<Point> q;
    q.push(start);
    parent[start.y][start.x] = start;
    
    while (!q.empty()) {
        Point curr = q.front();
        q.pop();
        
        if (curr.x == end.x && curr.y == end.y) break;
        
        for (int i = 0; i < 4; ++i) {
            int nx = curr.x + dx1[i];
            int ny = curr.y + dy1[i];
            
            if (grid[ny][nx] == ' ' && parent[ny][nx].x == -1) {
                parent[ny][nx] = curr;
                q.push({nx, ny});
            }
        }
    }

    vector<Point> path;
    Point step = end;
    while (!(step.x == start.x && step.y == start.y)) {
        path.push_back(step);
        step = parent[step.y][step.x];
    }
    path.push_back(start);
    reverse(path.begin(), path.end());
    
    for (size_t i = 0; i < path.size() - 1; ++i) {
        Point curr = path[i];
        Point next = path[i+1];
        
        char dir = ' ';
        if (next.x > curr.x) dir = 'R';
        else if (next.x < curr.x) dir = 'L';
        else if (next.y > curr.y) dir = 'D';
        else if (next.y < curr.y) dir = 'U';
        
        grid[curr.y][curr.x] = dir;
    }
    grid[end.y][end.x] = 'E';

    string solved_output = "";
    for (int y = 0; y < GRID_H; ++y) {
        solved_output += grid[y] + "\n";
    }
    return unsolved_output + "&\n" + solved_output;
}

int main(){
    string filename = "mazes.txt";
    cout << "Starting process to build " << NUM_MAZES << " mazes." << endl;
    
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Failed to open " << filename << " for writing." << endl;
        return 1;
    }
    
    random_device rd;
    mt19937 rng(rd());
    
    for (int i = 0; i < NUM_MAZES; ++i) {
        outfile << generate_solved_maze(rng);
        if (i < NUM_MAZES - 1) {
            outfile << "\n"; 
        }
        if ((i+1) % ((NUM_MAZES)/10) == 0) {
            cout << "Generated " << (i + 1) << " mazes..." << endl;
        }
    }
    
    outfile.close();
    cout << "Done! Saved to " << filename << endl;
    return 0;
}