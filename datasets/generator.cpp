#include <bits/stdc++.h>
using namespace std;
#define pii pair<int,int>
#define PROMPT_SKIP cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n'); continue

// Solution Format
#define COMPLETION 0
#define DIRECTIONS 1

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
};

class MazeGenerator {
protected:
    pii n; vector<string> grid;
    string solution;
public:
    virtual ~MazeGenerator() = default;
    virtual void generate(pii size) = 0;
    virtual void random_generation() = 0;
    virtual void solve() = 0;
    virtual string textify(int type) = 0;

    void save(int num, const string& filename, int format = COMPLETION) {
        string filepath = "datasets/" + filename + ".txt";
        ofstream outfile(filepath);
        if(!outfile.is_open()){
            cerr << "Failed to open " << filepath << endl;
            return;
        }
        cout << "Building " << num << " mazes..." << endl;
        for(int i = 0; i < num; i++) {
            random_generation(); 
            outfile << textify(format);
            if(i < num - 1) outfile << "\n";
        }
        outfile.close();
        cout << "Saved to " << filepath << endl;
    }
};

int main(){
    Prompter p;
}